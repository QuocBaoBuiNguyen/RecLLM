import logging
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn import Parameter

try:
    from transformers import (
        AutoTokenizer,
        BertModel,
        InstructBlipQFormerConfig,
        InstructBlipQFormerModel,
    )
except (ImportError, ModuleNotFoundError):  # pragma: no cover - depends on runtime env
    AutoTokenizer = None
    BertModel = None
    InstructBlipQFormerConfig = None
    InstructBlipQFormerModel = None

LOGGER = logging.getLogger(__name__)


class HFQFormerAdapter(nn.Module):
    """Wrapper around Hugging Face InstructBLIP Q-Former.

    Exposes multiple forward modes used by the SigLLM training stages:

    - ``forward(cf_vec, text)`` — joint forward returning query hidden states
      with ``out_proj`` applied. Used by Stage 2 / Stage 3 when ``text`` is the
      InstructBLIP-style task instruction.
    - ``encode_cf(cf_vec)`` — queries only, no text branch input. Used by
      Stage 1 ITC where the CF and text streams are kept uni-modal.
    - ``encode_text(text)`` — text only, no queries, no cross-attention. Used
      by Stage 1 ITC and as a CLS pool for downstream contrastive losses.
    - ``forward_multimodal(cf_vec, text, causal_text)`` — joint forward
      returning both query and text hidden states. ``causal_text=True`` masks
      text→text attention causally for ITG; ``False`` is the default
      bidirectional mode used by ITM and by ``forward``.
    - ``forward_interaction(target_cf, history_cf, history_mask)`` —
      interaction-aware mode: queries cross-attend a joint sequence of
      ``[target, history_1, ..., history_L]`` rather than a single CF token,
      so the user-target-history compatibility is encoded inside Q-Former
      itself rather than left to the downstream LLM. Used at Stage 3 step 2
      when ``model.qformer_config.interaction_aware=True`` in config.

    The recommendation signal enters through ``encoder_hidden_states`` (cross-
    attention to a single CF token, or to the L+1 sequence in interaction
    mode); the text stream enters through ``input_ids`` (self-attention with
    the learned queries).
    """

    def __init__(
        self,
        d_cf: int,
        d_model: int,
        num_queries: int = 8,
        num_heads: int = 8,
        num_layers: int = 2,
        output_dim: Optional[int] = None,
        dropout: float = 0.0,
        intermediate_size: Optional[int] = None,
        cross_attention_frequency: int = 2,
        initializer_range: float = 0.02,
        qformer_text_model_name: str = "bert-base-uncased",
        max_instruction_length: int = 48,
        init_from_pretrained_text: bool = True,
        instruction_aware: bool = True,
        max_interaction_length: int = 11,
    ):
        super().__init__()
        self.instruction_aware = bool(instruction_aware)
        # Length budget for forward_interaction's encoder input
        # (target + max_history). ML-1M max history is 10 → 11 slots total.
        self.max_interaction_length = int(max_interaction_length)

        if AutoTokenizer is None or InstructBlipQFormerConfig is None or InstructBlipQFormerModel is None:
            raise ModuleNotFoundError(
                "transformers with InstructBLIP support is required to use HFQFormerAdapter. "
                "Please install or upgrade transformers in the runtime environment."
            )

        self.d_cf = d_cf
        self.d_model = d_model
        self.num_queries = num_queries
        self.output_dim = int(output_dim) if output_dim is not None else d_model
        self.max_instruction_length = int(max_instruction_length)
        self.qformer_tokenizer = AutoTokenizer.from_pretrained(
            qformer_text_model_name,
            truncation_side="right",
        )

        self.q = Parameter(torch.randn(1, num_queries, d_model))
        self.proj_cf = nn.Linear(d_cf, d_model)
        self.out_proj = nn.Identity() if self.output_dim == d_model else nn.Linear(d_model, self.output_dim)

        # Learnable positional embedding for the encoder sequence in
        # forward_interaction. Slot 0 = target item; slots 1..max_history =
        # history items in temporal order (oldest → newest). Small-scale
        # init follows BLIP-2 / BERT positional embedding convention.
        self.interaction_pos_embedding = Parameter(
            torch.empty(self.max_interaction_length, d_model)
        )
        nn.init.normal_(self.interaction_pos_embedding, mean=0.0, std=initializer_range)

        config = InstructBlipQFormerConfig(
            vocab_size=len(self.qformer_tokenizer),
            hidden_size=d_model,
            encoder_hidden_size=d_model,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size or (4 * d_model),
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            cross_attention_frequency=cross_attention_frequency,
            initializer_range=initializer_range,
        )
        self.qformer = InstructBlipQFormerModel(config)
        self.vocab_size = config.vocab_size

        if init_from_pretrained_text:
            self._init_text_branch_from_pretrained_bert(qformer_text_model_name)

    def _init_text_branch_from_pretrained_bert(self, bert_model_name: str) -> int:
        """Copy embeddings, self-attention, and FFN weights from a pretrained
        BERT into the Q-Former (BLIP-2 / InstructBLIP convention).

        Cross-attention layers keep their random init since BERT has no
        cross-attention. The Q-Former names self-attention modules
        ``encoder.layer.i.attention.attention.*`` while BERT uses
        ``encoder.layer.i.attention.self.*``; we remap accordingly. When the
        Q-Former has fewer layers than BERT, only the first ``num_layers``
        of BERT are copied. Tensors with mismatched shapes (e.g. when
        ``hidden_size`` or ``num_heads`` is configured differently from
        BERT-base) are skipped, leaving them at their random init.

        Returns the number of tensors successfully transferred.
        """
        if BertModel is None:
            LOGGER.warning(
                "transformers.BertModel not available; skipping pretrained "
                "text-branch init. Q-Former text side will start from random."
            )
            return 0

        try:
            bert = BertModel.from_pretrained(bert_model_name)
        except Exception as exc:  # network / cache miss
            LOGGER.warning(
                "Could not load pretrained BERT '%s' for Q-Former text-branch "
                "init (%s). Falling back to random init.",
                bert_model_name,
                exc,
            )
            return 0

        bert_state = bert.state_dict()
        target_state = self.qformer.state_dict()

        loaded = 0
        skipped_shape = 0
        for q_key, q_tensor in target_state.items():
            bert_key = q_key.replace(".attention.attention.", ".attention.self.")
            if bert_key not in bert_state:
                continue
            bert_tensor = bert_state[bert_key]
            if bert_tensor.shape != q_tensor.shape:
                skipped_shape += 1
                continue
            target_state[q_key] = bert_tensor.clone()
            loaded += 1

        self.qformer.load_state_dict(target_state, strict=True)
        del bert

        LOGGER.info(
            "Initialized Q-Former text branch from %s: %d tensors loaded, "
            "%d shape-mismatch skipped, %d Q-Former tensors total.",
            bert_model_name,
            loaded,
            skipped_shape,
            len(target_state),
        )
        return loaded

    def load_state_dict(self, state_dict, strict: bool = True):
        """Accept both adapter-native keys and keys where the inner HF
        Q-Former prefix was stripped by external loading code."""
        if not isinstance(state_dict, dict):
            return super().load_state_dict(state_dict, strict=strict)

        remapped_state_dict = dict(state_dict)
        expected_keys = set(super().state_dict().keys())

        has_prefixed_qformer_keys = any(
            isinstance(k, str) and k.startswith("qformer.") for k in remapped_state_dict
        )
        if not has_prefixed_qformer_keys:
            fixed_state_dict = {}
            remapped_any_key = False
            for key, value in remapped_state_dict.items():
                if isinstance(key, str) and f"qformer.{key}" in expected_keys:
                    fixed_state_dict[f"qformer.{key}"] = value
                    remapped_any_key = True
                else:
                    fixed_state_dict[key] = value
            if remapped_any_key:
                remapped_state_dict = fixed_state_dict

        return super().load_state_dict(remapped_state_dict, strict=strict)

    @property
    def text_word_embeddings(self) -> nn.Module:
        """Q-Former's text token embedding table.

        Returned so callers (e.g. ITG language modelling head) can tie weights
        to the Q-Former tokenizer's vocabulary.
        """
        return self.qformer.embeddings.word_embeddings

    def _normalize_text_input(self, text: Union[str, list], batch_size: int) -> list:
        if isinstance(text, str):
            return [text] * batch_size
        text = list(text)
        if len(text) != batch_size:
            raise ValueError(f"Expected {batch_size} text strings, got {len(text)}")
        return text

    def _tokenize(self, text_list: list, max_len: int, device):
        tokens = self.qformer_tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        return tokens.input_ids.to(device), tokens.attention_mask.to(device)

    def _project_cf(self, cf_vec: torch.Tensor):
        if cf_vec.dim() != 2:
            raise ValueError(f"Expected cf_vec to have shape [B, d_cf], got {tuple(cf_vec.shape)}")
        encoder_hidden_states = self.proj_cf(cf_vec).unsqueeze(1)
        encoder_attention_mask = torch.ones(
            cf_vec.size(0), 1, dtype=torch.long, device=cf_vec.device
        )
        return encoder_hidden_states, encoder_attention_mask

    def _build_causal_joint_mask(
        self,
        batch_size: int,
        query_count: int,
        text_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """3D attention mask for ITG: queries see queries (bidirectional),
        text positions see queries + previous text tokens (causal)."""

        device = text_attention_mask.device
        text_len = text_attention_mask.size(1)
        seq = query_count + text_len

        mask = torch.zeros(batch_size, seq, seq, dtype=torch.long, device=device)
        mask[:, :query_count, :query_count] = 1
        mask[:, query_count:, :query_count] = 1
        causal = torch.tril(torch.ones(text_len, text_len, dtype=torch.long, device=device))
        mask[:, query_count:, query_count:] = causal.unsqueeze(0).expand(batch_size, -1, -1)

        col_pad = text_attention_mask.unsqueeze(1).expand(batch_size, seq, text_len)
        mask[:, :, query_count:] = mask[:, :, query_count:] * col_pad
        row_pad = text_attention_mask.unsqueeze(-1)
        mask[:, query_count:, :] = mask[:, query_count:, :] * row_pad
        return mask

    def encode_cf(self, cf_vec: torch.Tensor) -> torch.Tensor:
        """Queries-only forward over a CF (collaborative filtering) vector.

        The recommendation signal enters via cross-attention to a single CF
        token; there is no text-side input. Returns query hidden states of
        shape ``[B, num_queries, d_model]``. ``out_proj`` is not applied
        here; callers decide whether they want the projected (LLM-feeding)
        or raw (contrastive) representation.
        """
        batch_size = cf_vec.size(0)
        query_tokens = self.q.expand(batch_size, -1, -1)
        query_attention_mask = torch.ones(
            batch_size, query_tokens.size(1), dtype=torch.long, device=cf_vec.device
        )
        encoder_hidden_states, encoder_attention_mask = self._project_cf(cf_vec)

        outputs = self.qformer(
            input_ids=None,
            attention_mask=query_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def encode_text(
        self,
        text: Union[str, list],
        max_length: Optional[int] = None,
    ):
        """Text-only forward — no queries, no cross-attention.

        Returns ``(text_hidden, text_cls)`` where ``text_hidden`` is
        ``[B, T, d_model]`` and ``text_cls`` is ``text_hidden[:, 0]``.
        """
        if isinstance(text, str):
            text_list = [text]
        else:
            text_list = list(text)
        device = next(self.qformer.parameters()).device
        input_ids, attention_mask = self._tokenize(
            text_list, max_length or self.max_instruction_length, device
        )
        outputs = self.qformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            query_embeds=None,
            encoder_hidden_states=None,
            return_dict=True,
        )
        text_hidden = outputs.last_hidden_state
        return text_hidden, text_hidden[:, 0]

    def forward_multimodal(
        self,
        cf_vec: torch.Tensor,
        text: Union[str, list],
        causal_text: bool = False,
        max_text_length: Optional[int] = None,
    ):
        """Joint forward returning ``(query_hidden, text_hidden, text_ids, text_mask)``.

        ``causal_text=True`` enables a causal mask on the text→text attention
        block (used by ITG); the default is bidirectional (used by ITM and by
        the LLM-feeding ``forward``).
        """
        if cf_vec.dim() != 2:
            raise ValueError(f"Expected cf_vec to have shape [B, d_cf], got {tuple(cf_vec.shape)}")

        batch_size = cf_vec.size(0)
        text_list = self._normalize_text_input(text, batch_size)

        query_tokens = self.q.expand(batch_size, -1, -1)
        query_count = query_tokens.size(1)

        text_ids, text_attention_mask = self._tokenize(
            text_list, max_text_length or self.max_instruction_length, cf_vec.device
        )

        encoder_hidden_states, encoder_attention_mask = self._project_cf(cf_vec)
        query_attention_mask = torch.ones(
            batch_size, query_count, dtype=torch.long, device=cf_vec.device
        )

        if causal_text:
            joint_attention_mask = self._build_causal_joint_mask(
                batch_size, query_count, text_attention_mask
            )
        else:
            joint_attention_mask = torch.cat([query_attention_mask, text_attention_mask], dim=1)

        outputs = self.qformer(
            input_ids=text_ids,
            attention_mask=joint_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )

        sequence_hidden = outputs.last_hidden_state
        query_hidden = sequence_hidden[:, :query_count]
        text_hidden = sequence_hidden[:, query_count:]
        return query_hidden, text_hidden, text_ids, text_attention_mask

    def forward(self, cf_vec: torch.Tensor, instruction) -> torch.Tensor:
        """LLM-feeding mode: queries cross-attend to ``cf_vec`` while the text
        stream consumes ``instruction``. Returns query hidden states with
        ``out_proj`` applied: ``[B, num_queries, output_dim]``.

        When ``self.instruction_aware`` is False, the instruction text is
        dropped and this reduces to ``encode_cf`` + ``out_proj`` — a vanilla
        BLIP-2 forward without instruction routing. This is the ablation
        baseline isolating the contribution of the instruction-aware design.
        """

        if not self.instruction_aware:
            query_hidden = self.encode_cf(cf_vec)
            return self.out_proj(query_hidden)

        query_hidden, _, _, _ = self.forward_multimodal(
            cf_vec, instruction, causal_text=False
        )
        return self.out_proj(query_hidden)

    def forward_interaction(
        self,
        target_cf: torch.Tensor,
        history_cf: torch.Tensor,
        history_mask: torch.Tensor,
        instruction=None,
    ) -> torch.Tensor:
        """Interaction-aware mode: queries jointly cross-attend to the
        sequence ``[target_cf, history_cf_1, ..., history_cf_L]``. Optionally
        instruction-aware: when ``instruction`` is provided and
        ``self.instruction_aware=True``, instruction tokens are concatenated
        with queries in the self-attention path while the queries cross-attend
        to the multi-element encoder memory — this is the faithful
        InstructBLIP design adapted for recommendation (multi-element memory
        replaces image patches, instruction guides query routing).

        Unlike per-item mode (``encode_cf`` / ``forward``), the queries see
        the entire user-item-history context in a single forward pass and
        the output ``[B, num_queries, output_dim]`` encodes target-history
        compatibility directly — rather than leaving that reasoning to the
        downstream LLM's self-attention over many per-item soft tokens.

        Args
        ----
        target_cf
            ``[B, d_cf]`` — target item's CF embedding.
        history_cf
            ``[B, L, d_cf]`` — L history item CF embeddings, padded with
            zero/sentinel rows where ``history_mask`` is 0.
        history_mask
            ``[B, L]`` — binary mask: 1 where the history slot is a real
            (non-padded) item, 0 where the slot is padding. The target slot
            is always considered valid and is not part of this mask.
        instruction
            Optional. Single string (broadcast to batch) or list of B strings.
            When provided AND ``self.instruction_aware=True``, instruction
            tokens concat with queries in self-attention (full InstructBLIP
            design over the multi-element memory). When None or
            ``self.instruction_aware=False``, falls back to vanilla
            queries-only over the multi-element memory.

        Returns
        -------
        ``[B, num_queries, output_dim]`` interaction-aware query hidden
        states with ``out_proj`` applied.
        """
        if target_cf.dim() != 2:
            raise ValueError(
                f"Expected target_cf shape [B, d_cf], got {tuple(target_cf.shape)}"
            )
        if history_cf.dim() != 3:
            raise ValueError(
                f"Expected history_cf shape [B, L, d_cf], got {tuple(history_cf.shape)}"
            )
        if history_mask.dim() != 2:
            raise ValueError(
                f"Expected history_mask shape [B, L], got {tuple(history_mask.shape)}"
            )

        batch_size = target_cf.size(0)
        history_len = history_cf.size(1)
        seq_len = history_len + 1  # target slot + L history slots

        if seq_len > self.max_interaction_length:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_interaction_length="
                f"{self.max_interaction_length}. Increase max_interaction_length "
                f"or truncate history."
            )

        # 1. Stack target + history: [B, L+1, d_cf]; target at position 0.
        sequence_cf = torch.cat([target_cf.unsqueeze(1), history_cf], dim=1)

        # 2. Project to d_model and add positional embedding.
        encoder_hidden = self.proj_cf(sequence_cf)
        encoder_hidden = encoder_hidden + self.interaction_pos_embedding[:seq_len].unsqueeze(0)

        # 3. Build attention mask: target always valid (1), history slots
        # follow history_mask.
        target_mask = torch.ones(
            batch_size, 1, dtype=history_mask.dtype, device=history_mask.device
        )
        encoder_attention_mask = torch.cat([target_mask, history_mask], dim=1)

        # 4. Queries + (optional) instruction tokens.
        query_tokens = self.q.expand(batch_size, -1, -1)
        query_count = query_tokens.size(1)
        query_attention_mask = torch.ones(
            batch_size, query_count, dtype=torch.long, device=target_cf.device
        )

        use_instruction = self.instruction_aware and instruction is not None
        if use_instruction:
            # Tokenize instruction and concat in self-attention (matches
            # forward_multimodal pattern but with multi-element encoder memory).
            text_list = self._normalize_text_input(instruction, batch_size)
            text_ids, text_attention_mask = self._tokenize(
                text_list, self.max_instruction_length, target_cf.device
            )
            joint_attention_mask = torch.cat(
                [query_attention_mask, text_attention_mask], dim=1
            )
            outputs = self.qformer(
                input_ids=text_ids,
                attention_mask=joint_attention_mask,
                query_embeds=query_tokens,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=True,
            )
            # Slice query positions only (drop instruction tokens at output).
            sequence_hidden = outputs.last_hidden_state
            query_hidden = sequence_hidden[:, :query_count]
            return self.out_proj(query_hidden)

        # Fallback: queries-only over multi-element memory (no instruction).
        outputs = self.qformer(
            input_ids=None,
            attention_mask=query_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        return self.out_proj(outputs.last_hidden_state)
