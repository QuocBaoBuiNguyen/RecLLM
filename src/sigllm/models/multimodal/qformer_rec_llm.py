
import logging
import random
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import os

from sigllm.common.logging_utils import NotebookLogger
from sigllm.common.registry import registry
from sigllm.models.multimodal.base.rec_base_model import Rec2Base
# from sigllm.models.q_former.q_former import QFormer
from sigllm.models.q_former.hf_qformer_adapter import HFQFormerAdapter
from sigllm.models.q_former.text_encoder import TextEncoder

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
    
    PLACEHOLDERS_FOR_EMBED = ["<UserID>", "<ItemIDList>", "<TargetItemID>"]
    SOFT_TOKEN_PLACEHOLDERS = {
        "<UserID>": "<SOFT_USER_EMB>",
        "<ItemIDList>": "<SOFT_HISTORY_EMB>",
        "<TargetItemID>": "<SOFT_TARGET_EMB>",
    }

    def __init__(
        self,
        rec_model="MF",
        rec_config=None,
        pretrained_rec=None,
        pretrained_qformer=None,
        freeze_rec=True,
        rec_precision='fp16',
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        proj_token_num=1, # the number of tokens that the user/item embedding projected to
        proj_drop=0,
        num_queries=8,
        num_heads=8,
        num_layers=2,
        lora_config=None,
        proj_mid=5,
        freeze_lora=False,
        freeze_proj=False
    ):
        super().__init__()

        self.low_resource = low_resource
        self.proj_token_num = proj_token_num
        self.use_lora = False
        self._has_logged_trainable_stats = False
        self._flow_log_steps = 0
        self._max_flow_log_steps = 3
        self._has_logged_prompt_injection_stats = False

        log_step("Running MiniGPT4Rec_v2 initialization")

        self.rec_model_type = rec_model
        
        # Initialize components
        self._init_rec_model(rec_model, rec_config, rec_precision, pretrained_rec, freeze_rec)
        self._init_llm_model(llama_model, low_resource, device_8bit)
        self.text_encoder, d_model = self._init_text_encoder(freeze_text_encoder=True)
        self._init_qformer(d_cf=rec_config.embedding_size, d_model=d_model, num_queries=num_queries, num_heads=num_heads, num_layers=num_layers, pretrained_qformer=pretrained_qformer, freeze_qformer=False)
        self._init_projection(proj_mid, proj_token_num, freeze_proj)
        self._init_prompts(prompt_path, prompt_template, max_txt_len, end_sym)

    def _init_text_encoder(self, freeze_text_encoder: bool):
        """
        Initializes the TextEncoder.
        """
        text_encoder = TextEncoder(model_name="bert-base-uncased")
        
        # Freeze if necessary
        if freeze_text_encoder:
            for p in text_encoder.parameters():
                p.requires_grad = False
            text_encoder.eval()
            text_encoder.train = disabled_train.__get__(text_encoder, TextEncoder)

        return text_encoder, text_encoder.model.config.hidden_size

    def _init_rec_model(self, rec_model, rec_config, rec_precision, pretrained_rec, freeze_rec):
        log_step("Loading Rec_model")
        self.rec_encoder = self.init_rec_encoder(rec_model, rec_config, rec_precision)
        
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

    def _init_llm_model(self, llama_model, low_resource, device_8bit):
        log_step(f"Loading LLAMA: {llama_model}")
        model_path = llama_model if llama_model else "./content/ckpt/llm/base"
        
        self.llama_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.llama_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )

        added_token_count = self.llama_tokenizer.add_special_tokens(
            {"additional_special_tokens": list(self.SOFT_TOKEN_PLACEHOLDERS.values())}
        )
        if added_token_count > 0:
            self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
            log_step("Added soft-token placeholders", f"count={added_token_count}")
        
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        log_step("Loading LLAMA Done")

    # def _init_lora(self, lora_config, freeze_lora):
    #     self.use_lora = False
    #     if lora_config is not None and lora_config.use_lora:
    #         log_step("Setting Lora")
    #         self.use_lora = True
    #         peft_config = LoraConfig(
    #             r=lora_config.r,
    #             lora_alpha=lora_config.alpha,
    #             target_modules=lora_config.target_modules,
    #             lora_dropout=lora_config.dropout,
    #             bias="none",
    #             task_type="CAUSAL_LM"
    #         ) 
    #         self.llama_model_lora = get_peft_model(self.llama_model, peft_config)
    #         log_step("Setting Lora Done")
        
    #     if freeze_lora and hasattr(self, 'llama_model_lora'):
    #         log_step("Freeze Lora...")
    #         for name, param in self.llama_model_lora.named_parameters():
    #             param.requires_grad = False

    def _init_qformer(self, d_cf, d_model, num_queries, num_heads, num_layers,
                    pretrained_qformer: str, freeze_qformer: bool):
        log_step("Loading QFormer")

        # 1) init qformer kiến trúc giống stage1
        # self.qformer = QFormer(
        #     d_cf=d_cf,
        #     d_model=d_model,
        #     num_queries=num_queries,
        #     num_heads=num_heads,
        #     num_layers=num_layers
        # ).to(self.device)
        self.qformer = HFQFormerAdapter(
            d_cf=d_cf,
            d_model=d_model,
            num_queries=num_queries,
            num_heads=num_heads,
            num_layers=num_layers
        ).to(self.device)

        # 2) load checkpoint stage1
        if pretrained_qformer and pretrained_qformer != "not_have":
            ckpt = torch.load(pretrained_qformer, map_location="cpu")

            # nếu bạn save thẳng state_dict: ckpt là dict param
            state_dict = ckpt

            # nếu ckpt có prefix "qformer." (trường hợp save full model)
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
            log_step("Train QFormer in stage2")

        log_step("Loading QFormer Done")
        return self.qformer

    def _init_projection(self, proj_mid, proj_token_num, freeze_proj):
        """
        Stage 2 projection: map Q-Former output tokens -> LLM hidden tokens.
        Input  : qformer_out [B, Q, d_q]
        Output : llm_tokens  [B, Q, H]
        """
        log_step("Loading Projection (QFormer -> LLM)")

        if self.qformer is None:
            raise ValueError("qformer is None. Please init/load Q-Former before init projection.")
        if not hasattr(self.qformer, "proj_cf") or not isinstance(self.qformer.proj_cf, nn.Linear):
            raise ValueError("qformer.proj_cf (nn.Linear) is required to infer d_q.")
        if not hasattr(self.qformer, "q"):
            raise ValueError("qformer.q (learned query tokens) is required to infer num_queries.")
        if self.llama_model is None:
            raise ValueError("llama_model is None. Please init LLM backbone before init projection.")

        d_q = self.qformer.proj_cf.out_features
        # Q = int(self.qformer.q.shape[0]) // Qformer self-implemented 
        Q = int(self.qformer.q.shape[-2])
        H = int(self.llama_model.config.hidden_size)

        # luôn sync theo Q-Former để tránh lệch số <unk> khi inject
        self.proj_token_num = Q
        if proj_token_num is not None and int(proj_token_num) != Q:
            log_step("WARNING",
                    f"proj_token_num({proj_token_num}) != qformer.num_queries({Q}). "
                    f"Using Q={Q} to keep injection consistent.")

        mid = int(proj_mid) if proj_mid is not None else 4

        # per-token projection: [B,Q,d_q] -> [B,Q,H]
        self.llama_proj = nn.Sequential(
            nn.LayerNorm(d_q),
            nn.Linear(d_q, d_q * mid),
            nn.GELU(),
            nn.Linear(d_q * mid, H),
        )

        if freeze_proj:
            for p in self.llama_proj.parameters():
                p.requires_grad = False
            self.llama_proj.eval()
            self.llama_proj.train = disabled_train
            log_step("Freeze llama_proj")

        log_step("Loading Projection Done",
                f"d_q={d_q}, H={H}, Q={self.proj_token_num}, mid={mid}")

    def _log_trainable_module_stats(self):
        if self._has_logged_trainable_stats:
            return

        stats = [
            f"rec_encoder={count_trainable_parameters(self.rec_encoder) if self.rec_encoder is not None else 0}",
            f"qformer={count_trainable_parameters(self.qformer) if self.qformer is not None else 0}",
            f"llama_proj={count_trainable_parameters(self.llama_proj) if hasattr(self, 'llama_proj') else 0}",
            f"llama_model={count_trainable_parameters(self.llama_model) if self.llama_model is not None else 0}",
        ]
        log_step("Trainable parameter counts", ", ".join(stats))
        self._has_logged_trainable_stats = True

    def _log_information_flow(self, user_q, target_q, user_llama, target_llama, merged_flat):
        if self._flow_log_steps >= self._max_flow_log_steps:
            return

        log_step(
            "Information flow",
            " | ".join(
                [
                    tensor_stat_string("user_q", user_q),
                    tensor_stat_string("target_q", target_q),
                    tensor_stat_string("user_llama", user_llama),
                    tensor_stat_string("target_llama", target_llama),
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
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            log_step(f"Load {len(self.prompt_list)} training prompts")
            log_step(f"Prompt List: \n{self.prompt_list}")
            self.has_pri_decode = False
            self.prompt_list_p = None
        else:
            self.prompt_list = []
            self.prompt_list_p = None

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
        if self.use_lora:
            return True

        id_terms = ["<UserID>", "<ItemIDList>", "<TargetItemID>", "<DCNFeature>"]
        for prompt in self.prompt_list:
            for id_term in id_terms:
                if id_term in prompt:
                    return True

        return False

    def set_answer_type(self,mode):
        if mode == 'v1':
            self.pos_ans = ["former"]
            self.neg_ans = ["latter"]
        elif mode == 'v2':
            self.pos_ans = ['Yes']
            self.neg_ans = ['No']
            pos_ans_id = self.llama_tokenizer(self.pos_ans[0],add_special_tokens=False).input_ids[0]
            neg_ans_id = self.llama_tokenizer(self.neg_ans[0],add_special_tokens=False).input_ids[0]
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

    def encode_rec_features_to_llm_v2(self, batch_data, feature_order=None):
        """
        Encodes recommendation features (User, History, Target) into LLM embedding space.
        
        Args:
            batch_data (dict): Dictionary containing:
                - 'UserID': (B,)
                - 'TargetItemID': (B,)
                - 'InteractedItemIDs_pad': (B, L)
            feature_order (list): Order of features, e.g., ["<UserID>", "<ItemIDList>", "<TargetItemID>"]
            
        Returns:
            rec_embeds (dict):
                - 'User_emb': (B, 1, H) - Individual user representation
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
        H = self.llama_model.config.hidden_size

        # 0) instruction tokens (stage2 MVP: fixed instruction)
        # (Nếu batch_data có instruction thì dùng batch_data["instruction"])
        ins_list = batch_data.get(
            "instruction",
            ["Dựa trên lịch sử tương tác, dự đoán người dùng có thích bộ phim này không. Yes/No."] * B
        )
        ins_tok_emb, _ = self.text_encoder(ins_list, device, max_len=48)  # h:[B,L,d_model], pooled:[B,d_model]
        # NOTE: assume TextEncoder returns (h, pooled) like stage1

        with self.maybe_autocast():
            # Stage-2 uses the in-tree rec encoder API: direct embedding lookup from ids.
            user_cf = self.rec_encoder.user_encoder(batch_data["UserID"])          # [B,d_cf]
            target_cf = self.rec_encoder.item_encoder(batch_data["TargetItemID"])  # [B,d_cf]

            # 2) QFormer outputs (instruction-conditioned)
            user_q = self.qformer(user_cf, ins_tok_emb)        # [B,Q,d_model]
            target_q = self.qformer(target_cf, ins_tok_emb)    # [B,Q,d_model]

            # 3) Project to LLM hidden per token
            user_llama = self.llama_proj(user_q)               # [B,Q,H]
            target_llama = self.llama_proj(target_q)           # [B,Q,H]

            interacted_llama_flat = None
            merged_flat = None

            has_interacted = "InteractedItemIDs_pad" in batch_data
            need_merge = has_interacted and feature_order is not None and len(feature_order) == 3

            if need_merge:
                ids = batch_data["InteractedItemIDs_pad"]  # [B,L]
                L = ids.shape[1]

                inter_cf = self.rec_encoder.item_encoder(ids)                              # [B,L,d_cf]
                inter_cf_flat = inter_cf.reshape(B * L, -1)                               # [B*L,d_cf]
                ins_rep = ins_tok_emb.repeat_interleave(L, dim=0)                         # [B*L,L_ins,d_model]

                inter_q_flat = self.qformer(inter_cf_flat, ins_rep)                       # [B*L,Q,d_model]
                inter_llama_flat2 = self.llama_proj(inter_q_flat)                         # [B*L,Q,H]
                inter_llama = inter_llama_flat2.reshape(B, L, Q, H)                       # [B,L,Q,H]
                interacted_llama_flat = inter_llama.reshape(B, L * Q, H)                  # [B,L*Q,H]

                # mask expand theo Q
                item_mask = (ids != self.rec_encoder.padding_index).long()                # [B,L]
                item_mask_q = item_mask.unsqueeze(-1).repeat(1, 1, Q).reshape(B, L * Q)   # [B,L*Q]
                ones_q = torch.ones((B, Q), device=device, dtype=item_mask.dtype)         # [B,Q]

                ph2emb = {
                    "<UserID>": user_llama,                 # [B,Q,H]
                    "<ItemIDList>": interacted_llama_flat,  # [B,L*Q,H]
                    "<TargetItemID>": target_llama          # [B,Q,H]
                }
                ph2mask = {
                    "<UserID>": ones_q,
                    "<ItemIDList>": item_mask_q,
                    "<TargetItemID>": ones_q
                }

                merged_embeds = torch.cat([ph2emb[ph] for ph in feature_order], dim=1)    # [B, Q + L*Q + Q, H]
                full_mask = torch.cat([ph2mask[ph] for ph in feature_order], dim=1)      # [B, Q + L*Q + Q]

                idx = torch.nonzero(full_mask, as_tuple=False)                            # [N,2]
                merged_flat = merged_embeds[idx[:, 0], idx[:, 1]]                         # [N,H]

            rec_embeds = {
                "User_emb": user_llama,                 # [B,Q,H]
                "TargetItem_emb": target_llama,         # [B,Q,H]
                "InteractedItems_embs": interacted_llama_flat,  # [B,L*Q,H] or None
                "merged_embs": merged_flat,             # [N,H] or None
            }
            self._log_information_flow(user_q, target_q, user_llama, target_llama, merged_flat)

        return rec_embeds, None

    def wrap_prompt_with_soft_tokens_v2(self, rec_embeds, rec_atts, batch_data, prompt_template):
        if not prompt_template:
             return None, None
        
        prompt_ori = prompt_template
        batch_size = batch_data['UserID'].shape[0]
        bos = self.llama_tokenizer.bos_token if self.llama_tokenizer.bos_token else "<s>"
        
        user_soft_token = self.SOFT_TOKEN_PLACEHOLDERS["<UserID>"]
        history_soft_token = self.SOFT_TOKEN_PLACEHOLDERS["<ItemIDList>"]
        target_soft_token = self.SOFT_TOKEN_PLACEHOLDERS["<TargetItemID>"]
        user_seq = " ".join([user_soft_token] * self.proj_token_num)
        target_seq = " ".join([target_soft_token] * self.proj_token_num)
        
        prompt_template = bos + prompt_template 
        prompt_template = prompt_template.replace("<UserID>", user_seq)
        prompt_template = prompt_template.replace("<TargetItemID>", target_seq)
        # prompt_template = prompt_template.replace("<DCNFeature>", user_seq)

        prompt_list = []
        for k in range(batch_size):
            current_prompt = prompt_template
            
            if 'InteractedItemIDs_pad' in batch_data:
                valid_items = (batch_data['InteractedItemIDs_pad'][k] != self.rec_encoder.padding_index).sum().item()
                item_list_placeholder = " ".join([history_soft_token] * (valid_items * self.proj_token_num))
                current_prompt = current_prompt.replace('<ItemIDList>', item_list_placeholder)

            if "<ItemTitleList>" in current_prompt and 'InteractedItemTitles' in batch_data:
                 current_prompt = current_prompt.replace("<ItemTitleList>", str(batch_data['InteractedItemTitles'][k]))
            
            if "<TargetItemTitle>" in current_prompt and 'TargetItemTitle' in batch_data:
                 current_prompt = current_prompt.replace("<TargetItemTitle>", str(batch_data['TargetItemTitle'][k]))

            prompt_list.append(current_prompt)
        
        if not self.has_print_prompt:
            log_step("prompt example:", prompt_list[0])
            self.has_print_prompt = True

        self.llama_tokenizer.padding_side = "left"
        prompts_tokens = self.llama_tokenizer(
            prompt_list,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(batch_data['UserID'].device)

        embed_layer = self.llama_model.get_input_embeddings()
        inputs_embeds = embed_layer(prompts_tokens.input_ids)

        replace_token_ids = torch.tensor(
            [
                self.llama_tokenizer.convert_tokens_to_ids(user_soft_token),
                self.llama_tokenizer.convert_tokens_to_ids(history_soft_token),
                self.llama_tokenizer.convert_tokens_to_ids(target_soft_token),
            ],
            device=prompts_tokens.input_ids.device,
        )
        replaced_idx = torch.nonzero(torch.isin(prompts_tokens.input_ids, replace_token_ids))
        
        if "<UserID>" in prompt_ori and "<TargetItemID>" in prompt_ori and "<ItemIDList>" in prompt_ori:
            if rec_embeds["merged_embs"] is None:
                raise ValueError("merged_embs is None while prompt requires <ItemIDList> soft-token injection.")
            if replaced_idx.shape[0] != rec_embeds["merged_embs"].shape[0]:
                raise ValueError(
                    "Soft-token placeholder count does not match merged embedding count: "
                    f"replaced_positions={replaced_idx.shape[0]}, merged_embs={rec_embeds['merged_embs'].shape[0]}"
                )
            inputs_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = rec_embeds['merged_embs'].to(inputs_embeds)

        elif "<UserID>" in prompt_ori and "<TargetItemID>" in prompt_ori and "<ItemIDList>" not in prompt_ori:
            emb_to_inject = torch.cat([rec_embeds['User_emb'], rec_embeds['TargetItem_emb']], dim=1)
            emb_to_inject = emb_to_inject.reshape(-1, emb_to_inject.shape[-1])
            if replaced_idx.shape[0] != emb_to_inject.shape[0]:
                raise ValueError(
                    "Soft-token placeholder count does not match injected embedding count: "
                    f"replaced_positions={replaced_idx.shape[0]}, emb_to_inject={emb_to_inject.shape[0]}"
                )
            inputs_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = emb_to_inject.to(inputs_embeds.dtype)

        elif "<DCNFeature>" in prompt_ori:
            raise NotImplementedError("<DCNFeature> is not implemented in this version")

        if not self._has_logged_prompt_injection_stats:
            valid_history_items = 0
            if 'InteractedItemIDs_pad' in batch_data:
                valid_history_items = int(
                    (batch_data['InteractedItemIDs_pad'][0] != self.rec_encoder.padding_index).sum().item()
                )

            user_soft_tokens = self.proj_token_num if "<UserID>" in prompt_ori else 0
            target_soft_tokens = self.proj_token_num if "<TargetItemID>" in prompt_ori else 0
            history_soft_tokens = valid_history_items * self.proj_token_num if "<ItemIDList>" in prompt_ori else 0
            total_soft_tokens = user_soft_tokens + target_soft_tokens + history_soft_tokens

            log_step(
                "Prompt injection stats",
                (
                    f"valid_history_items={valid_history_items}, "
                    f"user_soft_tokens={user_soft_tokens}, "
                    f"history_soft_tokens={history_soft_tokens}, "
                    f"target_soft_tokens={target_soft_tokens}, "
                    f"total_soft_tokens={total_soft_tokens}, "
                    f"replaced_positions={replaced_idx.shape[0]}"
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
            label_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        
        return torch.cat([empty_targets, label_targets], dim=1)

    def execute_llm_forward(self, embeds, atts, targets):
        with self.maybe_autocast():
            model = self.llama_model_lora if self.use_lora else self.llama_model
            return model(
                inputs_embeds=embeds,
                attention_mask=atts,
                labels=targets,
                return_dict=True
            )

    def calculate_recommendation_loss(self, outputs, label_tokens, batch_data, ans_map):
        pos_id = self.llama_tokenizer(ans_map[1], add_special_tokens=False).input_ids[0]
        label_seq_len = label_tokens.input_ids.shape[-1]
        
        prediction_logits = outputs.logits[:, -(label_seq_len + 1), :]
        target_logits = prediction_logits[:, pos_id]
        
        loss = nn.functional.binary_cross_entropy_with_logits(
            target_logits, 
            batch_data['label'].float()
        )
        
        return loss

    def build_llm_outputs_from_labels(self, batch_data):
        device = batch_data['UserID'].device
        ans_map = {1: self.pos_ans[0], 0: self.neg_ans[0]}
        text_labels = [ans_map[int(label)] for label in batch_data["label"]]

        self.llama_tokenizer.padding_side = "right"
        label_tokens = self.llama_tokenizer(
            text_labels,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        embed_layer = self.llama_model.get_input_embeddings()
        label_embeds = embed_layer(label_tokens.input_ids)

        return label_embeds, label_tokens, ans_map

    def build_llm_inputs_from_prompt_v2(self, prompt_template, batch_data):
        feature_order = self.get_placeholder_order(prompt_template) if prompt_template else None
        rec_embeds, rec_atts = self.encode_rec_features_to_llm_v2(batch_data, feature_order=feature_order)
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

        pos_id = self.llama_tokenizer(ans_map[1], add_special_tokens=False).input_ids[0]
        label_seq_len = label_tokens.input_ids.shape[-1]
        logits = outputs.logits[:, -(label_seq_len + 1), :][:, pos_id]
        logits = torch.sigmoid(logits)

        if return_all:
            return outputs, logits

        return {"loss": loss, "logits": logits}

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
        embedding_size = cfg.get("rec_emb_size")
        freeze_rec = cfg.get("freeze_rec",True)
        rec_precision = cfg.get("rec_precision", 'fp16')
        rec_config = cfg.get("rec_config")
        qformer_config = cfg.get("qformer_config")
        lora_config = cfg.get("lora_config")
        llama_model = cfg.get("llama_model")
        proj_token_num = cfg.get("proj_token_num")
        proj_drop = cfg.get("proj_drop")
        proj_mid = cfg.get("proj_mid_times")
        freeze_proj = cfg.get("freeze_proj")
        freeze_lora = cfg.get("freeze_lora")
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)
        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        num_queries = qformer_config.get("num_queries", 8) if qformer_config is not None else 8
        num_heads = qformer_config.get("num_heads", 8) if qformer_config is not None else 8
        num_layers = qformer_config.get("num_layers", 2) if qformer_config is not None else 2
        pretrained_qformer = qformer_config.get("qformer_ckpt") if qformer_config is not None else None

        model = cls(
            rec_model=rec_model,
            rec_config=rec_config,
            pretrained_rec=rec_config['pretrained_path'],
            pretrained_qformer=pretrained_qformer,
            freeze_rec=freeze_rec,
            rec_precision=rec_precision,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            proj_token_num=proj_token_num,
            proj_drop=proj_drop,
            num_queries=num_queries,
            num_heads=num_heads,
            num_layers=num_layers,
            lora_config=lora_config,
            proj_mid=proj_mid,
            freeze_lora=freeze_lora,
            freeze_proj=freeze_proj
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
