import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def pull_model(model_path="lmsys/vicuna-7b-v1.5", save_dir="./ckpt/llm/vicuna_7b"):
    """
    Download and save the configured causal LM and tokenizer.
    """
    print(f"Pulling model from {model_path}...")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": "cpu"}  # Load to CPU for saving
    )
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving model and tokenizer to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Done!")
    return save_dir


def smoke_test_model(model_dir, prompt="Q: What is the largest animal?\nA:", max_new_tokens=32):
    """Load the saved model and run one quick generation test."""

    use_cuda = torch.cuda.is_available()
    device_map = {"": 0} if use_cuda else {"": "cpu"}
    dtype = torch.float16 if use_cuda else torch.float32

    print(f"Running smoke test from {model_dir} on {'cuda' if use_cuda else 'cpu'}...")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map=device_map,
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    generation_output = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
    print(tokenizer.decode(generation_output[0], skip_special_tokens=True))

if __name__ == "__main__":
    # Default to Vicuna-7B to match the original paper setup more closely.
    saved_dir = pull_model()
    smoke_test_model(saved_dir)
