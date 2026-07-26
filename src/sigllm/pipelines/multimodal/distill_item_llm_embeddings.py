import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from sigllm.datasets.data_preprocessing import LOGGER
from transformers import AutoModelForCausalLM, AutoTokenizer

from sigllm.common.logging_utils import NotebookLogger
from sigllm.common.utils import resolve_hf_model_path

DISTILL_TEMPLATE = (
    "The movie is described by the following metadata. {item_text} "
    "Summarize the movie's characteristics for the purpose of recommending it to a user."
)

def parse_args():
    parser = argparse.ArgumentParser(description="Distill per-item LLM semantic embeddings")
    parser.add_argument("--llm-model", required=True, help="HF path/dir of the base LLM")
    parser.add_argument("--data-pkl", required=True, help="Preprocessed pickle with columns iid,title,genres (e.g. training_ood2.pkl).")
    parser.add_argument("--item-num", type=int, required=True, help="Number of items to process (for testing).")
    parser.add_argument("--output", required=True, help="Output path for the distilled embeddings (pickle).")
    parser.add_argument("--lora-adapter-dir", default=None, help="Path to LoRA adapter dir (if any).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for embedding extraction.")
    parser.add_argument("--max-length", type=int, default=64, help="Max length for tokenization.")
    parser.add_argument("--padding-index", type=int, default=0, help="Index of the padding token (default 0).")
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

@torch.no_grad()
def distill(args) -> None:
    model_path, local_files_only = resolve_hf_model_path(args.llm_model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        output_hidden_states=True,
        local_files_only=local_files_only,
    )
    if args.lora_adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_adapter_dir)
        LOGGER.info("Loaded LoRA adapter from %s", args.lora_adapter_dir)
    model.eval()

    hidden_size = int(model.config.hidden_size)
    device = next(model.parameters()).device

    item_texts = build_item_texts(args.data_pkl, args.item_num, args.padding_index)
    table = torch.zeros(args.item_num, hidden_size, dtype=torch.float32)

    ids = sorted(item_texts)

    for start in range(0, len(ids), args.batch_size):
        batch_ids = ids[start:start + args.batch_size]
        prompts = [DISTILL_TEMPLATE.format(item_text=item_texts[iid]) for iid in batch_ids]
        tokens = tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=True,
        ).to(device)

        outputs = model(**tokens, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]
        last_idx = tokens.attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(last_hidden.size(0), device=device)
        pooled = last_hidden[batch_idx, last_idx].float().cpu()

        for row, iid in enumerate(batch_ids):
            table[iid] = pooled[row]

        if (start // args.batch_size) % 20 == 0:
            LOGGER.info("Distilled %d/%d items...", min(start + args.batch_size, len(ids)), len(ids))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            "item_llm_emb": table,
            "item_num": args.item_num,
            "hidden_size": hidden_size,
            "llm_model": args.llm_model,
            "normalized": False,
        },
        args.output,
    )

    LOGGER.info("Saved item_llm_emb table shape %s to %s", table.shape, args.output)

def main():
    distill(parse_args())

if __name__ == "__main__":
    main()