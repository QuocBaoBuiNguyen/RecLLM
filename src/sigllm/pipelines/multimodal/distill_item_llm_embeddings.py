import argparse
import os
from types import SimpleNamespace

import pandas as pd
import torch
from sigllm.common.utils import resolve_hf_model_path
from transformers import AutoModelForCausalLM, AutoTokenizer

from sigllm.common.logging_utils import NotebookLogger

DISTILL_TEMPLATE = (
    "The movie is described by the following metadata. {item_text} "
    "Summarize the movie's characteristics for recommendation."
)

LOGGER = NotebookLogger.rich_logger("sigllm.distill_item_llm")

def parse_args():
    parser = argparse.ArgumentParser(description="Distill per-item LLM semantic embeddings")
    parser.add_argument("--rec-cfg", default=None, help="Path to the recommendation config file (YAML).")
    parser.add_argument("--rec-ckpt", default=None, help="Stage-3 Step-1 runner checkpoint ")
    parser.add_argument("--options", nargs="+", default=None)
    parser.add_argument("--llm-model", required=True, help="HF path/dir of the base LLM")
    parser.add_argument("--data-pkl", required=True, help="Preprocessed pickle with columns iid,title,genres (e.g. training_ood2.pkl).")
    parser.add_argument("--item-num", type=int, required=True, help="Number of items to process (for testing).")
    parser.add_argument("--output", required=True, help="Output path for the distilled embeddings (pickle).")
    parser.add_argument("--lora-adapter-dir", default=None, help="Path to LoRA adapter dir (if any).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for embedding extraction.")
    parser.add_argument("--max-length", type=int, default=64, help="Max length for tokenization.")
    parser.add_argument("--padding-index", type=int, default=0, help="Index of the padding token (default 0).")
    parser.add_argument(
        "--space",
        choices=["input", "last_hidden"],
        default="last_hidden",
        help=(
            "Target space of the distilled vectors. 'input': mask-mean of the LLM's input "
            "token embeddings over the raw item text — the space soft tokens are injected "
            "into; use this file for Stage-1/2 alignment (w_llm). 'last_hidden': last-layer "
            "hidden state at the final prompt token — the LLM's output space; use this file "
            "for the warm token."
        ),
    )
    return parser.parse_args()

def parse_genres(genres) -> list[str]:
    return sorted({g.strip() for g in str(genres).split("|") if g.strip()})

def build_item_texts(data_pkl: str, item_num: int, padding_index: int) -> dict[int, str]:
    df = pd.read_pickle(data_pkl)
    required = {"iid", "title", "genres"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns in {data_pkl}: {missing}")

    item_texts: dict[int, str] = {}
    # Select columns with a list (avoid KeyError from using a tuple key)
    for iid, title, genres in df[["iid", "title", "genres"]].drop_duplicates(subset="iid").itertuples(index=False):
        iid = int(iid)
        if iid == padding_index:
            continue
        parsed = parse_genres(genres)
        genre_str = ", ".join(parsed) if parsed else "Unknown"
        item_texts[iid] = f"Title: {title}. Genres: {genre_str}."

    covered = [i for i in range(item_num) if i in item_texts]

    LOGGER.info(f"Build item texts: %d/%d item id covered (missing ids get zero vector).", len(covered), item_num)
    return item_texts

def _load_base_or_adapter(args):

    if not args.llm_model:
        raise ValueError("Missing --llm-model argument.")
    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model,
        use_fast=False,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    source = "base"

    if args.lora_adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_adapter_dir)
        source = "adapter"
        LOGGER.info("Loaded LoRA adapter from %s", args.lora_adapter_dir)
    model.eval()
    return model, tokenizer, source

def _derive_user_item_num(data_pkl: str, item_num: int):
    data_dir = os.path.dirname(os.path.abspath(data_pkl))
    users, items = 0, item_num
    for name in ("train_ood2.pkl", "valid_ood2.pkl", "test_ood2.pkl"):
        p = os.path.join(data_dir, name)
        if os.path.isfile(p):
            df = pd.read_pickle(p)
            users = max(users, int(df["uid"].max()) + 1)
            items = max(items, int(df["iid"].max()) + 1)
    return max(users, 1), max(items, item_num)

def _load_finetuned_recllm(args):

    from sigllm.common.config import Config
    from sigllm.models.multimodal.qformer_rec_llm import QRecLLM

    cfg = Config(SimpleNamespace(cfg_path=args.rec_cfg, options=args.options))
    model_cfg = cfg.model_cfg

    model_cfg.tuning_step = 1
    model_cfg.ckpt = args.rec_ckpt

    if model_cfg.get("qformer_config") is not None:
        model_cfg.qformer_config.warm_token = False
        model_cfg.qformer_config.item_llm_emb_path = None

    step1 = cfg.run_cfg.get("qformer_stage3_step1") if cfg.run_cfg is not None else None
    if step1 is not None and step1.get("prompt_path"):
        model_cfg.prompt_path = step1.prompt_path

    user_num, item_num = _derive_user_item_num(args.data_pkl, args.item_num)
    model_cfg.rec_config.user_num = int(user_num)
    model_cfg.rec_config.item_num = int(item_num)

    model = QRecLLM.from_config(model_cfg)
    model.eval()

    LOGGER.info(
        "Build fine-tuned RecLLM for distillation (LoRA loaded from %s).", args.rec_ckpt
    )
    return model.llm_model, model.llm_tokenizer, "finetuned"

@torch.no_grad()
def _distill_table(model, tokenizer, item_texts, item_num, batch_size, max_length, padding_index, space):
    tokenizer.padding_size = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hidden_size = int(model.config.hidden_size)
    device = next(model.parameters()).device

    table = torch.zeros(item_num, hidden_size, dtype=torch.float32)

    ids = sorted(item_texts)

    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start:start + batch_size]
        if space == "input":
            # Raw item text only: the instruction template is constant across
            # items and would dominate a mask-mean over input embeddings.
            prompts = [item_texts[i] for i in batch_ids]
        else:
            prompts = [DISTILL_TEMPLATE.format(item_text=item_texts[i]) for i in batch_ids]
        tokens = tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        ).to(device)

        if space == "input":
            token_emb = model.get_input_embeddings()(tokens.input_ids) # [B, T, H]
            mask = tokens.attention_mask.unsqueeze(-1).to(token_emb.dtype) # [B, T, 1]
            pooled = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = pooled.float().cpu() # [B, H]
        else:
            outputs = model(**tokens, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.hidden_states[-1] # [B, T, H]
            last_idx = tokens.attention_mask.sum(dim=1) - 1 # [B]
            batch_idx = torch.arange(last_hidden.size(0), device=device)
            pooled = last_hidden[batch_idx, last_idx].float().cpu() # [B, H]

        for row, iid in enumerate(batch_ids):
            table[iid] = pooled[row]

        if (start // batch_size) % 20 == 0:
            LOGGER.info("Distilled %d/%d items...", min(start + batch_size, len(ids)), len(ids))
    return table, hidden_size

def distill(args) -> None:

    finetuned = bool(args.rec_cfg and args.rec_ckpt)

    if finetuned:
        model, tokenizer, source = _load_finetuned_recllm(args)
        llm_name = args.rec_ckpt
    else:
        if bool(args.rec_cfg) != bool(args.rec_ckpt):
            raise ValueError("--rec-cfg and --rec-ckpt must be given together.")
        model, tokenizer, source = _load_base_or_adapter(args)
        llm_name = args.llm_model

        if source == "base":
            LOGGER.warning(
                "Distilling from Base LLM. SeLLa distills from the Step-1 finetuned models"
            )

    item_texts = build_item_texts(args.data_pkl, args.item_num, args.padding_index)
    table, hidden_size = _distill_table(
        model, tokenizer, item_texts, args.item_num, args.batch_size, args.max_length, args.padding_index, args.space
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            "item_llm_emb": table,
            "item_num": args.item_num,
            "hidden_size": hidden_size,
            "llm_model": llm_name,
            "source": source,
            "space": args.space,
            "normalized": False,
        },
        args.output,
    )

    LOGGER.info("Saved item_llm_emb table shape %s to %s", table.shape, args.output)

def main():
    distill(parse_args())

if __name__ == "__main__":
    main()