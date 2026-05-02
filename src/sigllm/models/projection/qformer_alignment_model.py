import torch
import torch.nn as nn
import torch.nn.functional as F

class QRecInstructAlignmentModel(nn.Module):
    """Instruction-conditioned alignment with injected encoders."""

    def __init__(self, mf, qformer, text_encoder) -> None:
        super().__init__()
        self.mf = mf
        self.qformer = qformer
        self.text_encoder = text_encoder

        d = text_encoder.model.config.hidden_size
        # TEMP_DISABLED_USER_CF: user CF is not part of the temporary QFormer/LLM path.
        # self.p_user = nn.Linear(d, d)
        self.p_item = nn.Linear(d, d)
        self.p_text = nn.Linear(d, d)

    def pool_queries(self, q_tokens: torch.Tensor) -> torch.Tensor:
        # Kept only for the old pooled-QFormer path. Do not use in the ILM-style path.
        return q_tokens.mean(dim=1)

    def ins_tokens(self, ins_list, device):
        h, pooled = self.text_encoder(ins_list, device, max_len=48)
        return h

    def text_vec(self, text_list, device):
        _, pooled = self.text_encoder(text_list, device, max_len=64)
        return self.p_text(pooled)

    def enc_user(self, u_ids, ins_tok_emb):
        # TEMP_DISABLED_USER_CF: old stage-1 path encoded the user CF vector.
        # u_cf = self.mf.user_encoder(u_ids)
        # u_q = self.qformer(u_cf, ins_tok_emb)
        # return self.p_user(self.pool_queries(u_q))
        raise RuntimeError("User CF encoding is temporarily disabled.")

    def enc_item(self, i_ids, ins_tok_emb):
        i_cf = self.mf.item_encoder(i_ids)
        i_q = self.qformer(i_cf, ins_tok_emb)
        # Old path pooled all Q-Former query tokens into one vector too early:
        # return self.p_item(self.pool_queries(i_q))
        return self.p_item(i_q)

    def forward(self, u_idx: torch.Tensor, i_idx: torch.Tensor, ins_list: list[str], text_list: list[str]):
        device = u_idx.device
        ins_tok_emb = self.ins_tokens(ins_list, device)
        # TEMP_DISABLED_USER_CF: keep old call nearby for easy rollback.
        # user_z = self.enc_user(u_idx, ins_tok_emb)
        user_z = None
        item_z = self.enc_item(i_idx, ins_tok_emb)
        text_z = self.text_vec(text_list, device)
        return {"user": user_z, "item": item_z, "text": text_z}

    @staticmethod
    def l2norm(x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

    @staticmethod
    def select_query_by_text(q_tokens: torch.Tensor, text_vec: torch.Tensor):
        """
        Select the query token nearest to the matching text CLS vector.

        Args:
            q_tokens: item query tokens with shape [B, Q, D]
            text_vec: matching text vectors with shape [B, D]

        Returns:
            selected_queries: [B, D]
            selected_indices: [B]
        """
        if q_tokens.dim() != 3:
            raise ValueError(f"Expected q_tokens shape [B, Q, D], got {tuple(q_tokens.shape)}")
        if text_vec.dim() != 2:
            raise ValueError(f"Expected text_vec shape [B, D], got {tuple(text_vec.shape)}")
        if q_tokens.size(0) != text_vec.size(0) or q_tokens.size(-1) != text_vec.size(-1):
            raise ValueError(
                "Shape mismatch between q_tokens and text_vec: "
                f"q_tokens={tuple(q_tokens.shape)}, text_vec={tuple(text_vec.shape)}"
            )

        q_norm = QRecInstructAlignmentModel.l2norm(q_tokens)
        text_norm = QRecInstructAlignmentModel.l2norm(text_vec)
        scores = torch.einsum("bqd,bd->bq", q_norm, text_norm)
        selected_indices = scores.argmax(dim=1)
        batch_indices = torch.arange(q_tokens.size(0), device=q_tokens.device)
        selected_queries = q_tokens[batch_indices, selected_indices]
        return selected_queries, selected_indices

    @staticmethod
    def loss_user_item(u_vec: torch.Tensor, i_pos_vec: torch.Tensor, i_neg_vecs: torch.Tensor, tau: float = 0.07):
        u = QRecInstructAlignmentModel.l2norm(u_vec)
        pos = QRecInstructAlignmentModel.l2norm(i_pos_vec)
        neg = QRecInstructAlignmentModel.l2norm(i_neg_vecs)

        pos_logit = (u * pos).sum(-1, keepdim=True) / tau
        neg_logit = (u.unsqueeze(1) * neg).sum(-1) / tau

        logits = torch.cat([pos_logit, neg_logit], dim=1)
        labels = torch.zeros(u.size(0), dtype=torch.long, device=u.device)
        return F.cross_entropy(logits, labels)

    @staticmethod
    def loss_item_text(i_q_tokens: torch.Tensor, t_vec: torch.Tensor, tau: float = 0.07):
        # ILM-style: keep all query outputs until the item-text loss, then select
        # the query nearest to the matching text CLS vector.
        i_selected, _ = QRecInstructAlignmentModel.select_query_by_text(i_q_tokens, t_vec)
        i = QRecInstructAlignmentModel.l2norm(i_selected)
        t = QRecInstructAlignmentModel.l2norm(t_vec)
        logits = (i @ t.T) / tau
        labels = torch.arange(i.size(0), device=i.device)
        return F.cross_entropy(logits, labels)

    @staticmethod
    def loss_item_text_symmetric(i_vec: torch.Tensor, t_vec: torch.Tensor, tau: float = 0.07):
        if i_vec.dim() == 3:
            i_vec, _ = QRecInstructAlignmentModel.select_query_by_text(i_vec, t_vec)

        i = QRecInstructAlignmentModel.l2norm(i_vec)
        t = QRecInstructAlignmentModel.l2norm(t_vec)

        logits = (i @ t.T) / tau
        labels = torch.arange(i.size(0), device=i.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2
