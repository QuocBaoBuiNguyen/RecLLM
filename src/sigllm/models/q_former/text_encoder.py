import torch
import torch.nn as nn
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel, LlamaTokenizer, LlamaForCausalLM

class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = int(self.model.config.hidden_size)

    def forward(self, text_list, device, max_len=64):
        tokens = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)

        out = self.model(**tokens, return_dict=True)
        h = out.last_hidden_state
        mask = tokens.attention_mask

        input_mask_expanded = mask.unsqueeze(-1).expand(h.size()).float()
        sum_embeddings = torch.sum(h * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask

        return h, pooled


class LlamaEmbeddingTextEncoder(nn.Module):
    """
    Text encoder that represents text directly in the LLaMA input-embedding
    space. It returns token embeddings and mean-pooled embeddings with hidden
    size equal to the downstream LLaMA hidden size.
    """

    def __init__(self, model_name=None, tokenizer=None, embedding_layer=None, hidden_size=None, torch_dtype=None):
        super().__init__()

        if tokenizer is None:
            if model_name is None:
                raise ValueError("model_name is required when tokenizer is not provided.")
            tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=False)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        self.tokenizer = tokenizer

        model_config = None
        if embedding_layer is None:
            if model_name is None:
                raise ValueError("model_name is required when embedding_layer is not provided.")
            load_kwargs = {}
            if torch_dtype is not None:
                load_kwargs["torch_dtype"] = torch_dtype
            loaded_model = LlamaForCausalLM.from_pretrained(model_name, **load_kwargs)
            embedding_layer = loaded_model.get_input_embeddings()
            hidden_size = int(loaded_model.config.hidden_size)
            model_config = loaded_model.config

        self.embedding_layer = embedding_layer
        self.hidden_size = int(hidden_size or embedding_layer.embedding_dim)
        self.model = SimpleNamespace(config=model_config or SimpleNamespace(hidden_size=self.hidden_size))

    def forward(self, text_list, device, max_len=64):
        tokens = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
        ).to(device)

        embedding_device = self.embedding_layer.weight.device
        input_ids = tokens.input_ids.to(embedding_device)
        h = self.embedding_layer(input_ids).to(device=device, dtype=torch.float32)
        mask = tokens.attention_mask

        input_mask_expanded = mask.unsqueeze(-1).expand(h.size()).to(h.dtype)
        sum_embeddings = torch.sum(h * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask

        return h, pooled
