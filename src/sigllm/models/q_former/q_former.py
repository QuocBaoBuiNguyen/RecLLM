import torch
import torch.nn as nn
from torch.nn import Parameter


class QFormer(nn.Module):
    """Align item/history CF embeddings via learned queries and cross-attention."""

    def __init__(
        self,
        d_cf: int,
        d_model: int,
        num_queries: int = 8,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.q = Parameter(torch.randn(num_queries, d_model))
        self.proj_cf = nn.Linear(d_cf, d_model)

        self.attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
            )
            for _ in range(num_layers)
        ])
        self.ln1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.attn_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        self.ffn_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

    def forward(self, cf_vec: torch.Tensor, ins_token_emb: torch.Tensor) -> torch.Tensor:
        B = cf_vec.size(0)
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        cf_tok = self.proj_cf(cf_vec).unsqueeze(1)

        # InstructBLIP-style conditioning: instruction tokens share the
        # Q-Former self-attention stream with learned queries, while the CF
        # token is the cross-attention source.
        x = torch.cat([q, ins_token_emb], dim=1)
        query_count = q.size(1)
        kv = cf_tok
        for attn, ffn, ln1, ln2, attn_dropout, ffn_dropout in zip(
            self.attn,
            self.ffn,
            self.ln1,
            self.ln2,
            self.attn_dropout,
            self.ffn_dropout,
        ):
            y, _ = attn(x, kv, kv)
            x = ln1(x + attn_dropout(y))
            z = ffn(x)
            x = ln2(x + ffn_dropout(z))
        return x[:, :query_count]
