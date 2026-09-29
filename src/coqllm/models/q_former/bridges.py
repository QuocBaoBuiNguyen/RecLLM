"""CF -> soft-token bridges, selected by ``model.qformer_config.bridge``.

- ``qformer``  : the BLIP-2 Q-Former (``HFQFormerAdapter``). Default; the
  headline CoQLLM path. Constructor arguments are forwarded unchanged, so
  selecting it through this factory is behaviour-identical to building the
  adapter directly.
- ``attn_mlp`` : ablation control for the Q-Former. It keeps the same token
  budget (``num_queries`` tokens per item), width, depth and number of heads,
  but removes the two things that make a Q-Former a Q-Former: the learned
  query bank and the cross-attention to the CF vector.

Why the control is built this way: in CoQLLM each item is encoded from a single
CF vector, so the Q-Former cross-attends to exactly one key (KV = 1) and the
softmax is always 1. Differentiation across its query tokens therefore comes
from the learned queries plus self-attention/FFN depth. The control produces its
tokens as ``num_queries`` different linear views of the CF vector
(``token_gen``, **bias-free**: a per-token bias would be a learned constant per
token, i.e. a disguised query bank) and mixes them with plain self-attention.
Self-attention is never applied to a single CF token, which would be degenerate.

Only Stage 2 (generative) and Stage 3 apply to ``attn_mlp``: it has no text
stream, so the Stage-1 ITC/ITM/ITG objectives are undefined for it.
"""

from typing import Optional

import torch
import torch.nn as nn

from coqllm.models.q_former.hf_qformer_adapter import HFQFormerAdapter

BRIDGE_TYPES = ("qformer", "attn_mlp")


class AttnMLPBridge(nn.Module):
    """Attention-MLP bridge: CF vector -> ``num_queries`` tokens -> self-attention.

    Exposes the same interface the pipeline uses on ``HFQFormerAdapter``:
    ``encode_cf``, ``forward``, ``out_proj``, ``d_model``, ``output_dim``,
    ``num_queries`` and (optional) zero-initialised ``user_proj``.
    """

    def __init__(
        self,
        d_cf: int,
        d_model: int = 768,
        num_queries: int = 8,
        num_heads: int = 8,
        num_layers: int = 4,
        output_dim: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.0,
        user_conditioned: bool = False,
        d_user: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_cf = int(d_cf)
        self.d_model = int(d_model)
        self.num_queries = int(num_queries)
        self.output_dim = int(output_dim) if output_dim is not None else self.d_model

        # num_queries distinct linear views of the CF vector. bias=False on purpose
        # (see module docstring): no per-token learned constant.
        self.token_gen = nn.Linear(self.d_cf, self.num_queries * self.d_model, bias=False)
        # Counterpart of the LayerNorm the Q-Former applies to its query embeddings.
        # Its affine params are shared across tokens, so this adds no per-token constant.
        self.input_ln = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=int(intermediate_size or 4 * self.d_model),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,  # post-LN, like the BERT-style Q-Former layers
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=int(num_layers), enable_nested_tensor=False
        )
        self.out_proj = (
            nn.Identity() if self.output_dim == self.d_model else nn.Linear(self.d_model, self.output_dim)
        )

        # Same user-conditioning mechanism as HFQFormerAdapter, applied to the
        # generated tokens at the same position (before attention).
        self.user_conditioned = bool(user_conditioned)
        if self.user_conditioned:
            self.d_user = int(d_user) if d_user is not None else self.d_cf
            self.user_proj = nn.Linear(self.d_user, self.d_model)

        self.apply(self._init_weights)
        if self.user_conditioned:
            # Zero-init: a no-op at step 0, grows only if the CTR signal asks for it.
            nn.init.zeros_(self.user_proj.weight)
            nn.init.zeros_(self.user_proj.bias)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # BERT-style init, matching the Q-Former's initializer_range.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode_cf(self, cf_vec: torch.Tensor, user_cf: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``[B, d_cf]`` -> ``[B, num_queries, d_model]`` (``out_proj`` not applied)."""
        if cf_vec.dim() != 2:
            raise ValueError(f"Expected cf_vec to have shape [B, d_cf], got {tuple(cf_vec.shape)}")
        batch_size = cf_vec.size(0)
        tokens = self.token_gen(cf_vec).view(batch_size, self.num_queries, self.d_model)
        if self.user_conditioned and user_cf is not None:
            if user_cf.dim() != 2 or user_cf.size(0) != batch_size:
                raise ValueError(
                    f"Expected user_cf shape [{batch_size}, d_user], got {tuple(user_cf.shape)}"
                )
            tokens = tokens + self.user_proj(user_cf).unsqueeze(1)
        tokens = self.input_dropout(self.input_ln(tokens))
        return self.encoder(tokens)

    def forward(self, cf_vec: torch.Tensor, user_cf: Optional[torch.Tensor] = None) -> torch.Tensor:
        """LLM-feeding mode: ``[B, num_queries, output_dim]``."""
        return self.out_proj(self.encode_cf(cf_vec, user_cf=user_cf))


def build_bridge(
    bridge: str,
    *,
    d_cf: int,
    d_model: int,
    num_queries: int,
    num_heads: int,
    num_layers: int,
    output_dim: int,
    user_conditioned: bool = False,
    d_user: Optional[int] = None,
    **qformer_kwargs,
) -> nn.Module:
    """Build the bridge named by ``bridge``.

    ``qformer_kwargs`` (text model name, instruction length, init flags, ...) are
    passed through to ``HFQFormerAdapter`` untouched and ignored by ``attn_mlp``.
    """
    bridge = (bridge or "qformer").lower()
    if bridge == "qformer":
        return HFQFormerAdapter(
            d_cf=d_cf,
            d_model=d_model,
            num_queries=num_queries,
            num_heads=num_heads,
            num_layers=num_layers,
            output_dim=output_dim,
            user_conditioned=user_conditioned,
            d_user=d_user,
            **qformer_kwargs,
        )
    if bridge == "attn_mlp":
        return AttnMLPBridge(
            d_cf=d_cf,
            d_model=d_model,
            num_queries=num_queries,
            num_heads=num_heads,
            num_layers=num_layers,
            output_dim=output_dim,
            user_conditioned=user_conditioned,
            d_user=d_user,
        )
    raise ValueError(f"Unknown bridge '{bridge}'. Expected one of {BRIDGE_TYPES}.")


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    # Sanity check (torch only): python -m coqllm.models.q_former.bridges
    torch.manual_seed(0)
    bridge = AttnMLPBridge(d_cf=256, user_conditioned=True, d_user=256)
    cf, user = torch.randn(4, 256), torch.randn(4, 256)
    out = bridge(cf, user_cf=user)
    assert out.shape == (4, 8, 768), out.shape
    assert bridge.token_gen.bias is None
    # Zero-init user_proj => user conditioning is a no-op at initialisation.
    bridge.eval()
    assert torch.allclose(bridge(cf, user_cf=user), bridge(cf)), "user_proj must start as a no-op"
    # Tokens of one item must differ (distinct linear views, no copy of one vector).
    assert not torch.allclose(out[0, 0], out[0, 1])
    print(f"AttnMLPBridge OK: out={tuple(out.shape)}, params={count_parameters(bridge):,}")
