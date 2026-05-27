
import logging
import random
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import os

from sigllm.common.logging_utils import NotebookLogger
from sigllm.common.registry import registry
from sigllm.models.multimodal.base.rec_base_model import Rec2Base
from sigllm.models.q_former.hf_qformer_adapter import HFQFormerAdapter

LOGGER = NotebookLogger.rich_logger("sigllm.rec_base_model")

def log_step(title: str, detail: Optional[str] = None) -> None:
    """Emit a compact log line with optional detail string."""

    message = title if detail is None else f"{title} | {detail}"
    LOGGER.info(message)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def tensor_stat_string(name: str, tensor: Optional[torch.Tensor]) -> str:
    if tensor is None:
        return f"{name}=None"
    if tensor.numel() == 0:
        return f"{name}=empty shape={tuple(tensor.shape)}"

    detached = tensor.detach().float()
    mean_val = detached.mean().item()
    std_val = detached.std(unbiased=False).item()
    norm_val = detached.norm(dim=-1).mean().item() if detached.dim() >= 2 else detached.norm().item()
    return (
        f"{name}: shape={tuple(detached.shape)}, "
        f"mean={mean_val:.4f}, std={std_val:.4f}, mean_l2={norm_val:.4f}"
    )


@registry.register_model("mini_gpt4rec_v2")
class QRecLLM(Rec2Base):
    """
    QFormer + InstructBLIP for recommendation.
    """ 
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4rec.yaml",
    }    
    
    # TEMP_DISABLED_USER_CF: old prompt order included a user soft-token slot.
    # PLACEHOLDERS_FOR_EMBED = ["<UserID>", "<ItemIDList>", "<TargetItemID>"]
    PLACEHOLDERS_FOR_EMBED = ["<ItemIDList>", "<TargetItemID>"]

    # Item-text instructions for the Q-Former. Must match the distribution
    # the Q-Former was trained on in stage 1 (see
    # QFormerAlignmentBuilder.TEMPL_ITEM_TEXT). The verbose stage 2 prompt
    # MUST NOT be passed here — it gets truncated to max_instruction_length
    # tokens and would carry no per-item signal.
    QFORMER_ITEM_INSTRUCTIONS = [
        "Represent this movie for recommendation using its title and genres.",
        "Align this movie metadata with its collaborative filtering representation.",
        "Given the movie metadata, extract recommendation-relevant item features.",
        "Use the title and genres to describe this movie in the item embedding space.",
        "Map this movie's textual attributes to its collaborative recommendation signal.",
        "Identify the movie preferences implied by its title and genre metadata.",
        "Create a language-aligned representation of this movie for recommendation.",
        "Summarize this movie as an item a recommender system can compare.",
        "Based on the title and genres, represent what kind of users may like this movie.",
        "Encode the semantic information of this movie for item-language alignment.",
        "Use a few metadata cues to align this movie with behavioral item signals.",
        "Produce a recommendation-aware representation from this movie description.",
    ]

    def __init__(
        self,
        rec_model="MF",
        rec_config=None,
        pretrained_rec=None,
        pretrained_qformer=None,
        pretrained_llm_proj=None,
        freeze_rec=True,
        llm_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=1024,
        end_sym='\n',
        proj_token_num=1, # the number of tokens that the user/item embedding projected to
        num_queries=8,
        num_heads=8,
        num_layers=2,
        qformer_d_model=768,
        qformer_output_dim=None,
        qformer_text_model_name="bert-base-uncased",
        max_instruction_length=48,
        freeze_proj=False,
        ablate_soft_tokens=False,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=("q_proj", "v_proj"),
        lora_dropout=0.05,
        tuning_step=None,
    ):
        super().__init__()

        self.proj_token_num = proj_token_num
        self._has_logged_trainable_stats = False
        self._flow_log_steps = 0
        self._max_flow_log_steps = 3
        self._has_logged_prompt_injection_stats = False
        self._eval_pred_log_count = 0
        self._max_eval_pred_log_batches = 20

        self.ablate_soft_tokens = bool(ablate_soft_tokens)
        if self.ablate_soft_tokens:
            log_step(
                "ABLATION ACTIVE",
                "ablate_soft_tokens=True → target_llm and interacted_llm_flat "
                "will be zeroed before injection (Information flow log will show "
                "target_llm mean/std=0).",
            )

        self.use_lora = bool(use_lora)
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_target_modules = tuple(lora_target_modules)
        self.lora_dropout = float(lora_dropout)
        self.tuning_step = tuning_step

        log_step("Running MiniGPT4Rec_v2 initialization")

        self.rec_model_type = rec_model

        # Initialize components
        self._init_rec_model(rec_model, rec_config, pretrained_rec, freeze_rec)
        self._init_llm_model(llm_model)
        self._init_qformer(
            d_cf=rec_config.embedding_size,
            d_model=qformer_d_model,
            num_queries=num_queries,
            num_heads=num_heads,
            num_layers=num_layers,
            qformer_output_dim=qformer_output_dim,
            pretrained_qformer=pretrained_qformer,
            freeze_qformer=False,
            qformer_text_model_name=qformer_text_model_name,
            max_instruction_length=max_instruction_length,
        )
        self._init_projection(proj_token_num, freeze_proj, pretrained_llm_proj)
        self._init_prompts(prompt_path, prompt_template, max_txt_len, end_sym)
        self._apply_tuning_step_policy()

    def _init_rec_model(self, rec_model, rec_config, pretrained_rec, freeze_rec):
        log_step("Loading Rec_model")
        self.rec_encoder = self.init_rec_encoder(rec_model, rec_config)
        
        if self.rec_encoder is not None and pretrained_rec != "not_have":
            self.rec_encoder.load_state_dict(torch.load(pretrained_rec, map_location="cpu"))
            log_step("Successfully loaded the pretrained model")
        
        if freeze_rec and self.rec_encoder is not None:
            for name, param in self.rec_encoder.named_parameters():
                param.requires_grad = False
            self.rec_encoder = self.rec_encoder.eval()
            self.rec_encoder.train = disabled_train
            log_step("Freeze rec encoder")

        log_step("Loading Rec_model Done")

    def _init_llm_model(self, llm_path):
        log_step(f"Loading LLM: {llm_path}")
        model_path = llm_path if llm_path else "./content/ckpt/llm/base"

        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True,
        )
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self._resolve_soft_token_placeholder()
        log_step(
            "Loading LLM Done",
            f"hidden_size={self.llm_model.config.hidden_size}, "
            f"pad_token_id={self.llm_tokenizer.pad_token_id}, "
            f"soft_token_id={self._soft_token_id} ('{self._soft_token_str}')",
        )

        if self.use_lora:
            self._attach_lora()

    def _resolve_soft_token_placeholder(self):
        tok = self.llm_tokenizer
        if tok.unk_token_id is not None:
            self._soft_token_str = tok.unk_token
            self._soft_token_id = tok.unk_token_id
            return

        skip_ids = {tok.eos_token_id, tok.pad_token_id, tok.bos_token_id}
        skip_ids.discard(None)

        hardcoded = (
            "<|extra_0|>", "<|reserved_0|>", "<|fim_pad|>",
            "<|object_ref_start|>", "<|object_ref_end|>",
            "<|box_start|>", "<|box_end|>",
            "<|quad_start|>", "<|quad_end|>",
            "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
            "<|image_pad|>", "<|video_pad|>",
            "<|im_start|>",
        )
        for candidate in hardcoded:
            ids = tok(candidate, add_special_tokens=False).input_ids
            if len(ids) == 1 and ids[0] not in skip_ids:
                self._soft_token_str = candidate
                self._soft_token_id = ids[0]
                return

        added = getattr(tok, "added_tokens_decoder", None) or {}
        for token_id, added_token in added.items():
            if token_id in skip_ids:
                continue
            content = getattr(added_token, "content", str(added_token))
            ids = tok(content, add_special_tokens=False).input_ids
            if len(ids) == 1 and ids[0] == token_id:
                self._soft_token_str = content
                self._soft_token_id = token_id
                return

        log_step(
            "Soft-token fallback",
            "no unk_token and no safe single-token candidate; using eos_token "
            "as soft-slot placeholder. Soft slots will COLLIDE with padding if "
            "pad_token == eos_token — Step 2 may corrupt embeddings silently.",
        )
        self._soft_token_str = tok.eos_token
        self._soft_token_id = tok.eos_token_id

    def _attach_lora(self):
        from peft import LoraConfig, TaskType, get_peft_model

        log_step(
            "Attaching LoRA to LLM",
            f"r={self.lora_r}, alpha={self.lora_alpha}, "
            f"target_modules={list(self.lora_target_modules)}, dropout={self.lora_dropout}",
        )
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=list(self.lora_target_modules),
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.llm_model = get_peft_model(self.llm_model, lora_config)
        log_step(
            "LoRA attached",
            f"trainable LoRA params={count_trainable_parameters(self.llm_model)}",
        )

    def _apply_tuning_step_policy(self):
        step = self.tuning_step
        if step is None:
            return

        if int(step) == 1:
            for p in self.qformer.parameters():
                p.requires_grad = False
            for p in self.llm_proj.parameters():
                p.requires_grad = False
            self.qformer.eval()
            self.qformer.train = disabled_train
            self.llm_proj.eval()
            self.llm_proj.train = disabled_train
            log_step(
                "Tuning step 1",
                "LoRA trainable; Q-Former, projection, MF and base LLM all frozen.",
            )

        elif int(step) == 2:
            # CoLLM Equation (5), Ω = ϕ variant: Q-Former + projection trainable,
            # LoRA + base LLM + MF frozen. Q-Former was briefly frozen here as an
            # α2 experiment (theory: 8-layer Q-Former too large for 33k samples);
            # plateaued at uAUC ~0.694, below baseline 0.708, because llm_proj
            # alone (~2.8M params) lacks capacity to fix the CIE channel. Unfrozen
            # again to let Q-Former co-adapt with projection on the recommendation
            # task — matches the original CoLLM recipe.
            if hasattr(self.llm_model, "peft_config"):
                for n, p in self.llm_model.named_parameters():
                    if "lora_" in n:
                        p.requires_grad = False
            for p in self.qformer.parameters():
                p.requires_grad = True
            self.qformer.train()
            for p in self.llm_proj.parameters():
                p.requires_grad = True
            self.llm_proj.train()
            log_step(
                "Tuning step 2",
                "Q-Former + projection trainable; LoRA, base LLM and MF frozen.",
            )

        else:
            log_step("Tuning step", f"unrecognized value '{step}', no policy applied")

    def _init_qformer(
        self,
        d_cf,
        d_model,
        num_queries,
        num_heads,
        num_layers,
        qformer_output_dim,
        pretrained_qformer: str,
        freeze_qformer: bool,
        qformer_text_model_name: str,
        max_instruction_length: int,
    ):
        log_step("Loading QFormer")
        log_step(
            "Using Q-Former tokenizer for instructions",
            f"tokenizer={qformer_text_model_name}, hidden_size={d_model}",
        )

        self.qformer = HFQFormerAdapter(
            d_cf=d_cf,
            d_model=d_model,
            num_queries=num_queries,
            num_heads=num_heads,
            num_layers=num_layers,
            output_dim=qformer_output_dim or d_model,
            qformer_text_model_name=qformer_text_model_name,
            max_instruction_length=max_instruction_length,
            init_from_pretrained_text=False,
        ).to(self.device)

        if pretrained_qformer and pretrained_qformer != "not_have":
            ckpt = torch.load(pretrained_qformer, map_location="cpu")
            state_dict = ckpt
            if isinstance(state_dict, dict) and any(k.startswith("qformer.") for k in state_dict.keys()):
                state_dict = {k.replace("qformer.", "", 1): v for k, v in state_dict.items()}
            self.qformer.load_state_dict(state_dict, strict=True)
            log_step("Successfully loaded QFormer checkpoint", pretrained_qformer)

        # 3) freeze / train tiếp
        if freeze_qformer:
            for p in self.qformer.parameters():
                p.requires_grad = False
            self.qformer.eval()
            self.qformer.train = disabled_train
            log_step("Freeze QFormer")
        else:
            for p in self.qformer.parameters():
                p.requires_grad = True
            self.qformer.train()
            log_step("Train QFormer in stage 3")

        log_step("Loading QFormer Done")
        return self.qformer

    def _init_projection(self, proj_token_num, freeze_proj, pretrained_llm_proj=None):
        """
        Stage 3 projection: map Q-Former output tokens -> LLM hidden tokens.
        Input  : qformer_out [B, Q, d_q]
        Output : llm_tokens  [B, Q, H]

        Matches InstructBLIP: a single ``nn.Linear`` from Q-Former hidden size
        to LLM hidden size, applied per token. If ``pretrained_llm_proj``
        points to a state dict (e.g. from Stage 2 generative pretraining), it
        is loaded before any freezing.
        """
        log_step("Loading Projection (QFormer -> LLM)")

        if self.qformer is None:
            raise ValueError("qformer is None. Please init/load Q-Former before init projection.")
        if not hasattr(self.qformer, "q"):
            raise ValueError("qformer.q (learned query tokens) is required to infer num_queries.")
        if self.llm_model is None:
            raise ValueError("llm_model is None. Please init LLM backbone before init projection.")

        d_q = self.qformer.output_dim
        Q = int(self.qformer.q.shape[-2])
        H = int(self.llm_model.config.hidden_size)

        # luôn sync theo Q-Former để tránh lệch số <unk> khi inject
        self.proj_token_num = Q
        if proj_token_num is not None and int(proj_token_num) != Q:
            log_step("WARNING",
                    f"proj_token_num({proj_token_num}) != qformer.num_queries({Q}). "
                    f"Using Q={Q} to keep injection consistent.")

        self.llm_proj = nn.Sequential(
            nn.Linear(d_q, H),
            nn.LayerNorm(H),
        )
        nn.init.normal_(self.llm_proj[0].weight, std=0.02)
        nn.init.zeros_(self.llm_proj[0].bias)
        nn.init.constant_(self.llm_proj[1].weight, H ** -0.5)
        nn.init.zeros_(self.llm_proj[1].bias)

        if pretrained_llm_proj and pretrained_llm_proj != "not_have" and os.path.exists(pretrained_llm_proj):
            state_dict = torch.load(pretrained_llm_proj, map_location="cpu")
            self.llm_proj.load_state_dict(state_dict, strict=True)
            log_step("Loaded Stage 2 projection", pretrained_llm_proj)

        if freeze_proj:
            for p in self.llm_proj.parameters():
                p.requires_grad = False
            self.llm_proj.eval()
            self.llm_proj.train = disabled_train
            log_step("Freeze llm_proj")

        log_step("Loading Projection Done",
                f"d_q={d_q}, H={H}, Q={self.proj_token_num}")

    def _log_trainable_module_stats(self):
        if self._has_logged_trainable_stats:
            return

        llm_total = (
            count_trainable_parameters(self.llm_model) if self.llm_model is not None else 0
        )
        lora_total = 0
        if self.llm_model is not None and hasattr(self.llm_model, "peft_config"):
            lora_total = sum(
                p.numel()
                for n, p in self.llm_model.named_parameters()
                if "lora_" in n and p.requires_grad
            )
        stats = [
            f"rec_encoder={count_trainable_parameters(self.rec_encoder) if self.rec_encoder is not None else 0}",
            f"qformer={count_trainable_parameters(self.qformer) if self.qformer is not None else 0}",
            f"llm_proj={count_trainable_parameters(self.llm_proj) if hasattr(self, 'llm_proj') else 0}",
            f"llm_model={llm_total}",
            f"llm_lora={lora_total}",
        ]
        log_step("Trainable parameter counts", ", ".join(stats))
        self._has_logged_trainable_stats = True

    def _log_information_flow(self, user_q, target_q, user_llm, target_llm, merged_flat):
        if self._flow_log_steps >= self._max_flow_log_steps:
            return

        log_step(
            "Information flow",
            " | ".join(
                [
                    tensor_stat_string("user_q", user_q),
                    tensor_stat_string("target_q", target_q),
                    tensor_stat_string("user_llm", user_llm),
                    tensor_stat_string("target_llm", target_llm),
                    tensor_stat_string("merged_embs", merged_flat),
                ]
            ),
        )
        self._flow_log_steps += 1

    def _init_prompts(self, prompt_path, prompt_template, max_txt_len, end_sym):
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.has_print_prompt = False

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            # TEMP_DISABLED_USER_CF: keep old prompts in the file with this marker,
            # but do not sample them while user CF is disabled.
            filted_prompts = [
                raw_prompt for raw_prompt in raw_prompts
                if raw_prompt.strip() and not raw_prompt.lstrip().startswith("# DISABLED_USER_CF")
            ]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            log_step(f"Load {len(self.prompt_list)} training prompts")
            log_step(f"Prompt List: \n{self.prompt_list}")
        else:
            self.prompt_list = []

    def _sample_prompt(self):
        return random.choices(
            self.prompt_list,
            weights=[5] * (len(self.prompt_list) - 1) + [1],
            k=1,
        )[0]

    def set_mode(self, mode):
        '''
        mode \in ['v1','v2',None]
        '''
        self.run_mode_ = mode

    def to_be_trained(self):
        # TEMP_DISABLED_USER_CF: old trainable placeholders included "<UserID>".
        # id_terms = ["<UserID>", "<ItemIDList>", "<TargetItemID>", "<DCNFeature>"]
        id_terms = ["<ItemIDList>", "<TargetItemID>", "<DCNFeature>"]
        for prompt in self.prompt_list:
            for id_term in id_terms:
                if id_term in prompt:
                    return True

        if self.llm_model is not None and hasattr(self.llm_model, "peft_config"):
            for n, p in self.llm_model.named_parameters():
                if "lora_" in n and p.requires_grad:
                    return True

        return False

    def set_answer_type(self,mode):
        if mode == 'v1':
            self.pos_ans = ["former"]
            self.neg_ans = ["latter"]
        elif mode == 'v2':
            self.pos_ans = ['Yes']
            self.neg_ans = ['No']
            pos_ans_id = self.llm_tokenizer(self.pos_ans[0],add_special_tokens=False).input_ids[0]
            neg_ans_id = self.llm_tokenizer(self.neg_ans[0],add_special_tokens=False).input_ids[0]
            log_step("answer token ids: pos:{}, neg ids:{}".format(pos_ans_id, neg_ans_id))
            
        else:
            raise NotImplementedError("not implement this types of answers")

    def print_prompt(self):
        log_step('Prompt Pos Example \n{} {} or {}'.format(self._sample_prompt(),self.pos_ans[0],self.neg_ans[0]))
    
    def rec_to_cpu(self):
        self.rec_encoder.to("cpu")
        self.rec_encoder.float()
    
    def get_placeholder_order(self, prompt: str, placeholders=PLACEHOLDERS_FOR_EMBED):
        positions = []
        for ph in placeholders:
            pos = prompt.find(ph)
            if pos >= 0:
                positions.append((pos, ph))
        positions.sort(key=lambda x: x[0])
        return [ph for _, ph in positions]

    def encode_rec_features_to_llm_v2(self, batch_data, feature_order=None, instruction_list=None):
        """
        Encodes recommendation features (History, Target) into LLM embedding space.
        
        Args:
            batch_data (dict): Dictionary containing:
                - 'UserID': (B,)
                - 'TargetItemID': (B,)
                - 'InteractedItemIDs_pad': (B, L)
            feature_order (list): Order of features, e.g., ["<ItemIDList>", "<TargetItemID>"]
            
        Returns:
            rec_embeds (dict):
                - 'User_emb': None while TEMP_DISABLED_USER_CF is active
                - 'TargetItem_emb': (B, 1, H) - Individual target item representation
                - 'InteractedItems_embs': (B, L, H) - Historical items (includes padding)
                - 'merged_embs': (N, H) - Flattened & filtered valid tokens for LLM input
            rec_atts: None (Placeholder for future attention masks)
        """
        if self.rec_encoder is None:
            return None, None

        self._log_trainable_module_stats()

        device = batch_data["UserID"].device
        B = batch_data["UserID"].shape[0]
        Q = self.proj_token_num
        H = self.llm_model.config.hidden_size

        if instruction_list is None:
            instruction_list = batch_data.get(
                "instruction",
                self._build_qformer_instructions(B),
            )
        if isinstance(instruction_list, str):
            ins_list = [instruction_list] * B
        else:
            ins_list = list(instruction_list)
        if len(ins_list) != B:
            raise ValueError(f"Expected {B} instructions, got {len(ins_list)}")

        with self.maybe_autocast():
            # Stage-2 uses the in-tree rec encoder API: direct embedding lookup from ids.
            # TEMP_DISABLED_USER_CF: old path injected a user CF token into the LLM prompt.
            # user_cf = self.rec_encoder.user_encoder(batch_data["UserID"])          # [B,d_cf]
            user_q = None
            user_llm = None
            target_cf = self.rec_encoder.item_encoder(batch_data["TargetItemID"])  # [B,d_cf]

            # 2) QFormer outputs (instruction-conditioned)
            # user_q = self.qformer(user_cf, ins_list)        # [B,Q,d_model]
            target_q = self.qformer(target_cf, ins_list)      # [B,Q,d_model]

            # 3) Project to LLM hidden per token
            # user_llm = self.llm_proj(user_q)               # [B,Q,H]
            target_llm = self.llm_proj(target_q)           # [B,Q,H]

            if self.ablate_soft_tokens:
                target_llm = torch.zeros_like(target_llm)

            interacted_llm_flat = None
            merged_flat = None

            has_interacted = "InteractedItemIDs_pad" in batch_data
            need_merge = (
                has_interacted
                and feature_order is not None
                and "<ItemIDList>" in feature_order
                and "<TargetItemID>" in feature_order
            )

            if need_merge:
                ids = batch_data["InteractedItemIDs_pad"]  # [B,L]
                L = ids.shape[1]

                inter_cf = self.rec_encoder.item_encoder(ids)                              # [B,L,d_cf]
                inter_cf_flat = inter_cf.reshape(B * L, -1)                               # [B*L,d_cf]
                inter_ins_list = [ins for ins in ins_list for _ in range(L)]               # len B*L

                inter_q_flat = self.qformer(inter_cf_flat, inter_ins_list)                 # [B*L,Q,d_model]
                inter_llm_flat2 = self.llm_proj(inter_q_flat)                         # [B*L,Q,H]
                if self.ablate_soft_tokens:
                    inter_llm_flat2 = torch.zeros_like(inter_llm_flat2)
                inter_llm = inter_llm_flat2.reshape(B, L, Q, H)                       # [B,L,Q,H]
                interacted_llm_flat = inter_llm.reshape(B, L * Q, H)                  # [B,L*Q,H]

                # mask expand theo Q
                item_mask = (ids != self.rec_encoder.padding_index).long()                # [B,L]
                item_mask_q = item_mask.unsqueeze(-1).repeat(1, 1, Q).reshape(B, L * Q)   # [B,L*Q]
                ones_q = torch.ones((B, Q), device=device, dtype=item_mask.dtype)         # [B,Q]

                ph2emb = {
                    # TEMP_DISABLED_USER_CF: old merge map included "<UserID>": user_llm.
                    # "<UserID>": user_llm,                 # [B,Q,H]
                    "<ItemIDList>": interacted_llm_flat,  # [B,L*Q,H]
                    "<TargetItemID>": target_llm          # [B,Q,H]
                }
                ph2mask = {
                    # "<UserID>": ones_q,
                    "<ItemIDList>": item_mask_q,
                    "<TargetItemID>": ones_q
                }

                merged_embeds = torch.cat([ph2emb[ph] for ph in feature_order], dim=1)    # [B, L*Q + Q, H]
                full_mask = torch.cat([ph2mask[ph] for ph in feature_order], dim=1)       # [B, L*Q + Q]

                idx = torch.nonzero(full_mask, as_tuple=False)                            # [N,2]
                merged_flat = merged_embeds[idx[:, 0], idx[:, 1]]                         # [N,H]

            rec_embeds = {
                "User_emb": user_llm,                 # None while TEMP_DISABLED_USER_CF is active
                "TargetItem_emb": target_llm,         # [B,Q,H]
                "InteractedItems_embs": interacted_llm_flat,  # [B,L*Q,H] or None
                "merged_embs": merged_flat,             # [N,H] or None
            }
            self._log_information_flow(user_q, target_q, user_llm, target_llm, merged_flat)

        return rec_embeds, None

    def wrap_prompt_with_soft_tokens_v2(self, rec_embeds, rec_atts, batch_data, prompt_template):
        if not prompt_template:
             return None, None
        
        prompt_ori = prompt_template
        batch_size = batch_data['UserID'].shape[0]
        bos = self.llm_tokenizer.bos_token if self.llm_tokenizer.bos_token else ""

        unk_token = self._soft_token_str
        unk_seq = " ".join([unk_token] * self.proj_token_num)
        
        prompt_template = bos + prompt_template 
        # TEMP_DISABLED_USER_CF: old prompt path replaced <UserID> with soft tokens.
        # prompt_template = prompt_template.replace("<UserID>", unk_seq)
        prompt_template = prompt_template.replace("<UserID>", "")
        prompt_template = prompt_template.replace("<TargetItemID>", unk_seq)
        # prompt_template = prompt_template.replace("<DCNFeature>", unk_seq)

        prompt_list = []
        for k in range(batch_size):
            current_prompt = prompt_template
            
            if 'InteractedItemIDs_pad' in batch_data:
                valid_items = (batch_data['InteractedItemIDs_pad'][k] != self.rec_encoder.padding_index).sum().item()
                item_list_placeholder = " ".join([unk_seq] * valid_items)
                current_prompt = current_prompt.replace('<ItemIDList>', item_list_placeholder)

            if "<ItemTitleList>" in current_prompt and 'InteractedItemTitles' in batch_data:
                 current_prompt = current_prompt.replace("<ItemTitleList>", str(batch_data['InteractedItemTitles'][k]))
            
            if "<TargetItemTitle>" in current_prompt and 'TargetItemTitle' in batch_data:
                 current_prompt = current_prompt.replace("<TargetItemTitle>", str(batch_data['TargetItemTitle'][k]))

            prompt_list.append(current_prompt)
        
        if not self.has_print_prompt:
            preview_parts = []
            if "<ItemIDList>" in prompt_ori and 'InteractedItemIDs_pad' in batch_data:
                history_ids = batch_data['InteractedItemIDs_pad'][0].detach().cpu().tolist()
                history_ids = [int(i) for i in history_ids if int(i) != self.rec_encoder.padding_index]
                preview_parts.append(
                    f"[ItemIDList ids={history_ids} soft_tokens={len(history_ids) * self.proj_token_num}]",
                )
            if "<TargetItemID>" in prompt_ori and 'TargetItemID' in batch_data:
                target_id = int(batch_data['TargetItemID'][0].detach().cpu().item())
                preview_parts.append(
                    f"[TargetItemID id={target_id} soft_tokens={self.proj_token_num}]",
                )
            log_step("prompt injection preview:", " | ".join(preview_parts))
            self.has_print_prompt = True

        self.llm_tokenizer.padding_side = "left"
        prompts_tokens = self.llm_tokenizer(
            prompt_list,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(batch_data['UserID'].device)

        unk_token_id = self._soft_token_id
        
        embed_layer = self.llm_model.get_input_embeddings()
        inputs_embeds = embed_layer(prompts_tokens.input_ids)

        replaced_idx = torch.nonzero(prompts_tokens.input_ids == unk_token_id)

        has_history_placeholder = "<ItemIDList>" in prompt_ori
        has_target_placeholder = "<TargetItemID>" in prompt_ori

        if has_history_placeholder and has_target_placeholder and rec_embeds.get('merged_embs') is not None:
            inputs_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = rec_embeds['merged_embs'].to(inputs_embeds)

        elif has_target_placeholder:
            # TEMP_DISABLED_USER_CF: old target-only branch concatenated user and target tokens.
            # emb_to_inject = torch.cat([rec_embeds['User_emb'], rec_embeds['TargetItem_emb']], dim=1)
            emb_to_inject = rec_embeds['TargetItem_emb']
            emb_to_inject = emb_to_inject.reshape(-1, emb_to_inject.shape[-1])
            inputs_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = emb_to_inject.to(inputs_embeds.dtype)

        elif "<DCNFeature>" in prompt_ori:
            raise NotImplementedError("<DCNFeature> is not implemented in this version")

        if not self._has_logged_prompt_injection_stats:
            valid_history_items = 0
            if 'InteractedItemIDs_pad' in batch_data:
                valid_history_items = int(
                    (batch_data['InteractedItemIDs_pad'][0] != self.rec_encoder.padding_index).sum().item()
                )

            target_soft_tokens = self.proj_token_num if "<TargetItemID>" in prompt_ori else 0
            history_soft_tokens = valid_history_items * self.proj_token_num if "<ItemIDList>" in prompt_ori else 0
            total_soft_tokens = target_soft_tokens + history_soft_tokens
            sample_unk_slots = int((prompts_tokens.input_ids[0] == unk_token_id).sum().item())

            log_step(
                "Prompt injection stats",
                (
                    f"valid_history_items={valid_history_items}, "
                    f"history_soft_tokens={history_soft_tokens}, "
                    f"target_soft_tokens={target_soft_tokens}, "
                    f"sample_soft_tokens={total_soft_tokens}, "
                    f"sample_unk_slots={sample_unk_slots}, "
                    f"batch_unk_slots={replaced_idx.shape[0]}"
                ),
            )
            self._has_logged_prompt_injection_stats = True

        return inputs_embeds, prompts_tokens.attention_mask

    def assemble_llm_sequences(self, input_embeds, input_atts, label_embeds, label_atts):
        full_embeds = torch.cat([input_embeds, label_embeds], dim=1)
        full_atts = torch.cat([input_atts, label_atts], dim=1)
        return full_embeds, full_atts

    def prepare_llm_targets(self, input_atts, label_tokens):
        batch_size, input_len = input_atts.shape
        device = input_atts.device
        
        empty_targets = torch.full((batch_size, input_len), -100, device=device)
        
        label_targets = label_tokens.input_ids.masked_fill(
            label_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100
        )
        
        return torch.cat([empty_targets, label_targets], dim=1)

    def execute_llm_forward(self, embeds, atts, targets):
        with self.maybe_autocast():
            return self.llm_model(
                inputs_embeds=embeds,
                attention_mask=atts,
                return_dict=True,
            )

    def calculate_recommendation_loss(self, outputs, label_tokens, batch_data, ans_map):
        pos_id = self.llm_tokenizer(ans_map[1], add_special_tokens=False).input_ids[0]
        neg_id = self.llm_tokenizer(ans_map[0], add_special_tokens=False).input_ids[0]
        label_seq_len = label_tokens.input_ids.shape[-1]
        
        prediction_logits = outputs.logits[:, -(label_seq_len + 1), :]
        binary_logits = torch.stack(
            [prediction_logits[:, neg_id], prediction_logits[:, pos_id]],
            dim=1,
        )
        labels = batch_data['label'].long()
        
        loss = nn.functional.cross_entropy(binary_logits, labels)
        
        return loss

    def recommendation_scores(self, outputs, label_tokens, ans_map):
        pos_id = self.llm_tokenizer(ans_map[1], add_special_tokens=False).input_ids[0]
        neg_id = self.llm_tokenizer(ans_map[0], add_special_tokens=False).input_ids[0]
        label_seq_len = label_tokens.input_ids.shape[-1]

        prediction_logits = outputs.logits[:, -(label_seq_len + 1), :]
        binary_logits = torch.stack(
            [prediction_logits[:, neg_id], prediction_logits[:, pos_id]],
            dim=1,
        )
        return torch.softmax(binary_logits, dim=1)[:, 1]

    def build_llm_outputs_from_labels(self, batch_data):
        device = batch_data['UserID'].device
        ans_map = {1: self.pos_ans[0], 0: self.neg_ans[0]}
        text_labels = [ans_map[int(label)] for label in batch_data["label"]]

        self.llm_tokenizer.padding_side = "right"
        label_tokens = self.llm_tokenizer(
            text_labels,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        embed_layer = self.llm_model.get_input_embeddings()
        label_embeds = embed_layer(label_tokens.input_ids)

        return label_embeds, label_tokens, ans_map

    def _build_qformer_instructions(self, batch_size: int) -> list:
        """Build short item-text instructions for the Q-Former.

        Matches the distribution the Q-Former was trained on in stage 1: a
        fresh sample per row during training, a deterministic fixed string
        during eval/inference so the same input maps to the same embedding.
        """
        if self.training:
            return random.choices(self.QFORMER_ITEM_INSTRUCTIONS, k=batch_size)
        return [self.QFORMER_ITEM_INSTRUCTIONS[0]] * batch_size

    def build_llm_inputs_from_prompt_v2(self, prompt_template, batch_data):
        feature_order = self.get_placeholder_order(prompt_template) if prompt_template else None
        batch_size = batch_data["UserID"].shape[0]

        if not feature_order:
            self._log_trainable_module_stats()
            rec_embeds = {
                "User_emb": None,
                "TargetItem_emb": None,
                "InteractedItems_embs": None,
                "merged_embs": None,
            }
            rec_atts = None
        else:
            instruction_list = self._build_qformer_instructions(batch_size)
            rec_embeds, rec_atts = self.encode_rec_features_to_llm_v2(
                batch_data,
                feature_order=feature_order,
                instruction_list=instruction_list,
            )

        llm_embeds, llm_atts = self.wrap_prompt_with_soft_tokens_v2(rec_embeds, rec_atts, batch_data, prompt_template)
        return llm_embeds, llm_atts

    def generate_for_samples(self, samples, return_all=False):
        prompt = self.prompt_list[0]
        input_embeds, input_atts = self.build_llm_inputs_from_prompt_v2(prompt, samples)
        label_embeds, label_tokens, ans_map = self.build_llm_outputs_from_labels(samples)

        full_embeds, full_atts = self.assemble_llm_sequences(
            input_embeds, input_atts, label_embeds, label_tokens.attention_mask
        )

        targets = self.prepare_llm_targets(input_atts, label_tokens)

        outputs = self.execute_llm_forward(full_embeds, full_atts, targets)
        loss = self.calculate_recommendation_loss(outputs, label_tokens, samples, ans_map)

        logits = self.recommendation_scores(outputs, label_tokens, ans_map)

        self._maybe_log_predictions(samples, logits, ans_map)

        if return_all:
            return outputs, logits

        return {"loss": loss, "logits": logits}

    def _maybe_log_predictions(self, samples, prob_yes, ans_map, samples_per_batch=3):
        """Emit a compact per-sample log of (UserID, TargetItemID, label,
        predicted Yes/No, probability). Bounded by
        ``_max_eval_pred_log_batches`` to avoid log spam; the counter resets
        at the start of every eval pass (see ``eval``)."""

        if self._eval_pred_log_count >= self._max_eval_pred_log_batches:
            return

        user_ids = samples["UserID"].detach().cpu().tolist()
        item_ids = samples["TargetItemID"].detach().cpu().tolist()
        labels = samples["label"].detach().cpu().tolist()
        probs = prob_yes.detach().float().cpu().tolist()

        n = min(len(user_ids), samples_per_batch)
        rows = []
        for i in range(n):
            pred = ans_map[1] if probs[i] >= 0.5 else ans_map[0]
            gt = ans_map[int(labels[i])]
            outcome = "OK" if pred == gt else "WRONG"
            rows.append(
                f"user={user_ids[i]} item={item_ids[i]} "
                f"gt={gt} prob_yes={probs[i]:.3f} pred={pred} {outcome}"
            )

        log_step(
            f"[stage3 eval batch #{self._eval_pred_log_count}]",
            " | ".join(rows),
        )
        self._eval_pred_log_count += 1

    def eval(self):
        """Reset the prediction-log counter so each eval pass starts logging
        from sample 0 again. ``train(True)`` does not reset, so logs stay
        scoped to eval calls."""
        self._eval_pred_log_count = 0
        return super().eval()

    def forward_v2(self, batch_data):
        prompt = self._sample_prompt()
        input_embeds, input_atts = self.build_llm_inputs_from_prompt_v2(prompt, batch_data)
        label_embeds, label_tokens, ans_map = self.build_llm_outputs_from_labels(batch_data)

        full_embeds, full_atts = self.assemble_llm_sequences(
            input_embeds, input_atts, label_embeds, label_tokens.attention_mask
        )
        
        targets = self.prepare_llm_targets(input_atts, label_tokens)
        
        outputs = self.execute_llm_forward(full_embeds, full_atts, targets)
        loss = self.calculate_recommendation_loss(outputs, label_tokens, batch_data, ans_map)

        return {"loss": loss}

    def forward(self, samples):
        if self.run_mode_ == 'v2':
            return self.forward_v2(samples)
        else:
            raise NotImplementedError("Only forward_v2 is implemented in this version")

    @classmethod
    def from_config(cls, cfg):
        rec_model = cfg.get('rec_model',"MF")
        freeze_rec = cfg.get("freeze_rec",True)
        rec_config = cfg.get("rec_config")
        qformer_config = cfg.get("qformer_config") or {}
        llm_model = cfg.get("llm_model")
        proj_token_num = cfg.get("proj_token_num")
        freeze_proj = cfg.get("freeze_proj")
        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 1024)
        end_sym = cfg.get("end_sym", '\n')
        num_queries = qformer_config.get("num_queries", 8)
        num_heads = qformer_config.get("num_heads", 8)
        num_layers = qformer_config.get("num_layers", 2)
        qformer_d_model = qformer_config.get("qformer_d_model", 768)
        qformer_output_dim = qformer_config.get("qformer_output_dim")
        pretrained_qformer = qformer_config.get("qformer_ckpt")
        qformer_text_model_name = qformer_config.get("qformer_text_model_name", "bert-base-uncased")
        max_instruction_length = qformer_config.get("max_instruction_length", 48)
        pretrained_llm_proj = qformer_config.get("llm_proj_ckpt")
        ablate_soft_tokens = cfg.get("ablate_soft_tokens", False)

        lora_cfg = cfg.get("lora_config") or {}
        use_lora = bool(lora_cfg.get("use_lora", False))
        lora_r = int(lora_cfg.get("r", 8))
        lora_alpha = int(lora_cfg.get("alpha", 16))
        lora_target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
        lora_dropout = float(lora_cfg.get("dropout", 0.05))
        tuning_step = cfg.get("tuning_step", None)

        model = cls(
            rec_model=rec_model,
            rec_config=rec_config,
            pretrained_rec=rec_config['pretrained_path'],
            pretrained_qformer=pretrained_qformer,
            pretrained_llm_proj=pretrained_llm_proj,
            freeze_rec=freeze_rec,
            llm_model=llm_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            proj_token_num=proj_token_num,
            num_queries=num_queries,
            num_heads=num_heads,
            num_layers=num_layers,
            qformer_d_model=qformer_d_model,
            qformer_output_dim=qformer_output_dim,
            qformer_text_model_name=qformer_text_model_name,
            max_instruction_length=max_instruction_length,
            freeze_proj=freeze_proj,
            ablate_soft_tokens=ablate_soft_tokens,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            tuning_step=tuning_step,
        )

        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            log_step("Load QRecLLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            log_step("loading message, msg.... {}".format(msg))
            if os.path.exists(rec_config['pretrained_path']) and freeze_rec:
                model.rec_encoder.load_state_dict(torch.load(rec_config['pretrained_path'], map_location="cpu"))

        ans_type = cfg.get('ans_type')
        model.set_answer_type(mode=ans_type)
        model.print_prompt()
        return model
