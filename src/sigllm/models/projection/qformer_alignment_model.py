import torch
import torch.nn as nn
import torch.nn.functional as F

class QRecInstructAlignmentModel(nn.Module):
    """ILM-style instruction-conditioned item alignment model."""

    def __init__(self, mf, qformer, text_encoder) -> None:
        super().__init__()
        self.mf = mf
        self.qformer = qformer
        self.text_encoder = text_encoder

        d = text_encoder.hidden_size
        self.p_text = nn.Linear(d, d)

    def pool_queries(self, q_tokens: torch.Tensor) -> torch.Tensor:
        return q_tokens.mean(dim=1)

    def ins_tokens(self, ins_list, device):
        h, pooled = self.text_encoder(ins_list, device, max_len=48)
        return h

    def text_vec(self, text_list, device):
        _, pooled = self.text_encoder(text_list, device, max_len=64)
        return self.p_text(pooled)

    def enc_user(self, u_ids, ins_tok_emb):
        u_cf = self.mf.user_encoder(u_ids)
        return self.qformer(u_cf, ins_tok_emb)

    def enc_item(self, i_ids, ins_tok_emb):
        i_cf = self.mf.item_encoder(i_ids)
        return self.qformer(i_cf, ins_tok_emb)

    def forward(self, u_idx: torch.Tensor, i_idx: torch.Tensor, ins_list: list[str], text_list: list[str]):
        device = u_idx.device
        ins_tok_emb = self.ins_tokens(ins_list, device)
        user_z = self.enc_user(u_idx, ins_tok_emb)
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
    def select_pair_by_similarity(left_tokens: torch.Tensor, right_tokens: torch.Tensor):
        """
        Select the closest query-query pair for each positive item-item pair.

        ILM first chooses the representations from the positive pair, then uses
        those selected representations in the in-batch contrastive objective.
        """
        if left_tokens.dim() != 3 or right_tokens.dim() != 3:
            raise ValueError(
                "Expected pair inputs to have shape [B, Q, D], "
                f"got left={tuple(left_tokens.shape)}, right={tuple(right_tokens.shape)}"
            )
        if left_tokens.shape != right_tokens.shape:
            raise ValueError(
                "Expected paired query tensors to have the same shape, "
                f"got left={tuple(left_tokens.shape)}, right={tuple(right_tokens.shape)}"
            )

        left = QRecInstructAlignmentModel.l2norm(left_tokens)
        right = QRecInstructAlignmentModel.l2norm(right_tokens)
        scores = torch.einsum("bqd,brd->bqr", left, right)
        flat_indices = scores.flatten(start_dim=1).argmax(dim=1)
        right_query_count = right_tokens.size(1)
        left_indices = flat_indices // right_query_count
        right_indices = flat_indices % right_query_count
        batch_indices = torch.arange(left_tokens.size(0), device=left_tokens.device)
        return (
            left_tokens[batch_indices, left_indices],
            right_tokens[batch_indices, right_indices],
            left_indices,
            right_indices,
        )

    @staticmethod
    def inbatch_logits(left_vec: torch.Tensor, right_vec: torch.Tensor, tau: float = 0.07):
        if left_vec.dim() != 2 or right_vec.dim() != 2:
            raise ValueError(
                "Expected selected embeddings to have shape [B, D], "
                f"got left={tuple(left_vec.shape)}, right={tuple(right_vec.shape)}"
            )
        if left_vec.shape != right_vec.shape:
            raise ValueError(
                "Expected selected embeddings to have the same shape, "
                f"got left={tuple(left_vec.shape)}, right={tuple(right_vec.shape)}"
            )

        left = QRecInstructAlignmentModel.l2norm(left_vec)
        right = QRecInstructAlignmentModel.l2norm(right_vec)
        return (left @ right.T) / tau

    @staticmethod
    def multiquery_inbatch_logits(left_tokens: torch.Tensor, right_tokens: torch.Tensor, tau: float = 0.07):
        """Compatibility helper: max similarity across all query pairs."""
        if left_tokens.dim() == 2:
            left_tokens = left_tokens.unsqueeze(1)
        if right_tokens.dim() == 2:
            right_tokens = right_tokens.unsqueeze(1)
        if left_tokens.dim() != 3 or right_tokens.dim() != 3:
            raise ValueError(
                "Expected pair inputs to have shape [B, Q, D] or [B, D], "
                f"got left={tuple(left_tokens.shape)}, right={tuple(right_tokens.shape)}"
            )
        if left_tokens.size(-1) != right_tokens.size(-1):
            raise ValueError(
                "Pair embedding dimension mismatch: "
                f"left={tuple(left_tokens.shape)}, right={tuple(right_tokens.shape)}"
            )

        left = QRecInstructAlignmentModel.l2norm(left_tokens)
        right = QRecInstructAlignmentModel.l2norm(right_tokens)
        query_logits = torch.einsum("bqd,crd->bcqr", left, right) / tau
        return query_logits.amax(dim=(-1, -2))

    @staticmethod
    def loss_inbatch(
        left_vec: torch.Tensor,
        right_vec: torch.Tensor,
        tau: float = 0.07,
        symmetric: bool = False,
    ):
        logits = QRecInstructAlignmentModel.inbatch_logits(left_vec, right_vec, tau=tau)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_left = F.cross_entropy(logits, labels)
        if not symmetric:
            return loss_left
        loss_right = F.cross_entropy(logits.T, labels)
        return (loss_left + loss_right) / 2

    @staticmethod
    def loss_multiquery_inbatch_symmetric(
        left_tokens: torch.Tensor,
        right_tokens: torch.Tensor,
        tau: float = 0.07,
    ):
        left_selected, right_selected, _, _ = QRecInstructAlignmentModel.select_pair_by_similarity(
            left_tokens, right_tokens
        )
        return QRecInstructAlignmentModel.loss_inbatch(
            left_selected, right_selected, tau=tau, symmetric=True
        )

    @staticmethod
    def loss_item_item_ilm(left_tokens: torch.Tensor, right_tokens: torch.Tensor, tau: float = 0.07):
        left_selected, right_selected, _, _ = QRecInstructAlignmentModel.select_pair_by_similarity(
            left_tokens, right_tokens
        )
        return QRecInstructAlignmentModel.loss_inbatch(
            left_selected, right_selected, tau=tau, symmetric=False
        )

    @staticmethod
    def multiquery_inbatch_top1(left_tokens: torch.Tensor, right_tokens: torch.Tensor, tau: float = 0.07):
        if left_tokens.dim() == 3 and right_tokens.dim() == 3:
            left_tokens, right_tokens, _, _ = QRecInstructAlignmentModel.select_pair_by_similarity(
                left_tokens, right_tokens
            )
            logits = QRecInstructAlignmentModel.inbatch_logits(left_tokens, right_tokens, tau=tau)
        else:
            logits = QRecInstructAlignmentModel.multiquery_inbatch_logits(left_tokens, right_tokens, tau=tau)
        labels = torch.arange(logits.size(0), device=logits.device)
        return (logits.argmax(dim=1) == labels).float().mean()

    @staticmethod
    def loss_item_text(i_q_tokens: torch.Tensor, t_vec: torch.Tensor, tau: float = 0.07):
        i_selected, _ = QRecInstructAlignmentModel.select_query_by_text(i_q_tokens, t_vec)
        return QRecInstructAlignmentModel.loss_inbatch(
            i_selected, t_vec, tau=tau, symmetric=False
        )

    @staticmethod
    def loss_item_text_symmetric(i_vec: torch.Tensor, t_vec: torch.Tensor, tau: float = 0.07):
        if i_vec.dim() == 3:
            i_vec, _ = QRecInstructAlignmentModel.select_query_by_text(i_vec, t_vec)
        return QRecInstructAlignmentModel.loss_inbatch(
            i_vec, t_vec, tau=tau, symmetric=True
        )
