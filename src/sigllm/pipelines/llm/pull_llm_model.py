import os

import torch


def pull_model(
    model_path: str = "unsloth/Qwen2-7B-bnb-4bit",
    save_dir: str = "./ckpt/llm/qwen2-7b-unsloth-4bit",
):
    """Pre-warm HuggingFace cache with Unsloth's pre-quantized Qwen2-7B.

    Unsloth ships ``unsloth/Qwen2-7B-bnb-4bit`` — weights are already serialized
    in NF4 4-bit format (~5GB on disk vs ~14GB for the BF16 base). This call
    triggers the download into the HF cache so subsequent
    ``FastLanguageModel.from_pretrained`` calls (Stage 2/3) load from disk
    instead of re-downloading.

    Saving a local copy is optional — ``FastLanguageModel.from_pretrained`` can
    point straight at the HF model id. We snapshot to ``save_dir`` so the
    config can hold a stable local path that survives HF cache clears.
    """
    from unsloth import FastLanguageModel

    print(f"Pulling Unsloth model from {model_path} ...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,           # let Unsloth auto-detect (bf16 on Ampere+, fp16 elsewhere)
        load_in_4bit=True,
    )

    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving snapshot to {save_dir} ...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("Done!")
    return save_dir


def smoke_test_model(
    model_dir: str,
    prompt: str = "Q: What is the largest animal?\nA:",
    max_new_tokens: int = 32,
):
    """Load the saved 4-bit model and run one quick generation test.

    NOTE: Unsloth's fast-generate path currently has a bug with
    ``inputs_embeds=`` (issue #3309). Plain ``input_ids`` generation works
    fine, which is what this smoke test uses. SigLLM training/eval do not
    call ``generate`` — they use ``forward`` only — so the bug does not
    affect the training pipeline.
    """
    from unsloth import FastLanguageModel

    print(f"Running smoke test from {model_dir} ...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    generation_output = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
    print(tokenizer.decode(generation_output[0], skip_special_tokens=True))


if __name__ == "__main__":
    saved_dir = pull_model()
    smoke_test_model(saved_dir)
