# USER-LLM — Paper Notes

**Title:** USER-LLM: Efficient LLM Contextualization with User Embeddings
**Authors:** Lin Ning, Luyang Liu, Jiaxing Wu, Neo Wu et al. (Google DeepMind)
**Venue:** arXiv:2402.13598 (Feb 2024, AAAI)
**Source:** `docs/USER-LLM.pdf`

---

## 1. Core idea

Inspired by image-LLM models (Flamingo, BLIP-2), treat **user timeline as a distinct modality** → embed directly into LLM via **cross-attention** at multiple layers (not just input).

**Two key advantages:**
1. **78x inference speedup** vs text-prompt method (no long context)
2. **+16.33% accuracy** on tasks requiring deep user understanding (long sequences)

---

## 2. Two paradigms compared

### Paradigm 1: Text Prompt (traditional LLM-Rec)
```
"User's history: watched Avengers, Iron Man, Thor..."
→ Tokenize → LLM
```
Cost: very long context window (potentially thousands of tokens)

### Paradigm 2: USER-LLM (this work)
```
User history IDs → Autoregressive User Encoder → user embeddings
                                                     ↓ Perceiver compression
                                                K query tokens
                                                     ↓ Cross-attention at each layer
                                                Frozen LLM
```
Cost: K tokens (e.g., K=16) instead of N×T tokens

---

## 3. Architecture

### 3.1 Autoregressive User Encoder (Stage 1 pretraining)

```
User timeline: [item_1, item_2, ..., item_L]
       ↓ Feature-specific embeddings (name, rating, category, ...)
       ↓ Concatenated and projected
       ↓ Transformer decoder (autoregressive)
       ↓
E_su ∈ R^(L × d)
```

Pretrained with **next-item prediction** on user behavior sequences (self-supervised).

### 3.2 LLM Contextualization with Cross-Attention

**Flamingo-style** integration — insert cross-attention layers between LLM's self-attention layers:

```
O_i = LayerNorm(O_i)
O_i = CrossAttn(O_i, E_su, E_su) + O_i      ← cross-attention to user encoder output
I_{i+1} = FeedForward(O_i)
```

Where:
- Q (query) = LLM hidden states
- K, V (key, value) = User encoder output `E_su` (compressed by Perceiver)

### 3.3 Gated Cross-Attention (Flamingo gating)

```
O_i = tanh(α_i) · CrossAttn(O_i, E_su, E_su) + O_i
```

Where `α_i` is **trainable, zero-initialized scalar** per layer.

→ At init: cross-attention contribution = 0 → safe start. Gradient grows α_i only if CF signal helps task.

### 3.4 Perceiver compression (Section 3.2)

User encoder output `E_su` has L tokens (one per history item). For long histories (L = 100+), too many K/V tokens for cross-attention.

**Perceiver:** Trainable latent query (e.g., 16 queries) → cross-attend `E_su` → output 16 compressed query tokens.

```python
# Perceiver compression
compressed_user = perceiver_query[16].cross_attend(E_su)  # [16, d]
# Now LLM cross-attention K, V = compressed_user (16 tokens)
```

→ Effectively does what Q-Former does (compress sequence → K tokens).

---

## 4. Training framework — 2 stages

### Stage 1: User Encoder Pretraining
- Train autoregressive transformer on user interaction sequences
- Self-supervised: next-item prediction
- Multiple datasets: MovieLens, Amazon Review, Google Local Review

### Stage 2: Encoder-LLM Fine-tuning
- Three configurations tested:
  - **LLM frozen + encoder frozen + cross-attn trainable** (most efficient)
  - **LLM frozen + encoder trainable + cross-attn trainable**
  - **Everything trainable** (most parameters)

→ Frozen LLM with only cross-attention layers trainable is the efficient sweet spot.

---

## 5. Key Results

### Performance gains

| Dataset | Task | USER-LLM vs Text-prompt |
|---|---|---|
| MovieLens20M | Long sequence rec | up to **+16.33%** |
| Amazon Review | Various | significant gains on long contexts |
| Google Local Review | Personalization | gains |

### Inference efficiency

| Setup | Speedup |
|---|---|
| Text-prompt method (full history as text) | 1x baseline |
| USER-LLM (user embeddings + cross-attn) | **up to 78.1x** |

### Long-context advantage

USER-LLM particularly excels when:
- Long user histories (100+ items)
- Subtle behavioral shifts requiring deep understanding
- Tasks where text translation is verbose

---

## 6. Comparison with related approaches

| Approach | Integration | LLM | Trainable |
|---|---|---|---|
| Text prompt | Input text | Any | LoRA or full |
| USER-LLM | Cross-attention at multiple layers | Frozen | Cross-attn + α gates (~M params) |
| BLIP-2 / Q-Former | Soft tokens at INPUT only | Frozen | Q-Former + projection (~76M) |
| Flamingo | Cross-attention at multiple layers (image) | Frozen | Cross-attn + gates |

→ USER-LLM = **Flamingo-style for user behavior** instead of images.

---

## 7. Architecture differences vs SigLLM

| Aspect | USER-LLM | SigLLM |
|---|---|---|
| **Source encoder** | Autoregressive transformer (pretrained) | MF (static factorization) |
| **Source pretraining** | Self-supervised next-item | BPR-style matrix factorization |
| **Integration point** | Cross-attention at MULTIPLE LLM layers | Soft tokens at INPUT only |
| **Compression** | Perceiver (trainable queries) | Q-Former (similar concept) |
| **Gating** | Tanh(α) zero-init scalar | None |
| **Training stages** | Encoder pretrain → LLM finetune | MF → Q-Former Stage 1 → 2 → 3 |

---

## 8. Ideas applicable to SigLLM

### 🥇 Idea 1 — Gated cross-attention at LLM intermediate layers

**Hypothesis:** Soft tokens at LLM input only → CF signal may be "forgotten" by deeper layers. Add cross-attention at layers 8, 16, 24 (every 8 layers) with gated control.

**Implementation:**
```python
class QwenWithUserGatedXAttn(QwenModel):
    def __init__(self):
        self.xattn_layers = nn.ModuleList([
            CrossAttention(d_model=3584) for _ in [8, 16, 24]
        ])
        self.alphas = nn.Parameter(torch.zeros(3))  # zero-init gates
    
    def layer_forward(self, hidden, layer_idx, soft_tokens):
        if layer_idx in [8, 16, 24]:
            xattn_out = self.xattn_layers[i](hidden, soft_tokens, soft_tokens)
            hidden = hidden + torch.tanh(self.alphas[i]) * xattn_out
        # ... rest of layer
```

**Trade-offs:**
- Pros: Deeper integration, gated safe start
- Cons: **Major refactor of HuggingFace Qwen2 forward**, conflicts with LoRA, breaks standard model compatibility
- Effort: 1-2 weeks
- Probability: 25-35%

**Verdict:** High complexity for SigLLM. Not recommended unless very ambitious.

### 🥈 Idea 2 — Autoregressive user encoder (replace MF)

**Hypothesis:** MF static embeddings miss temporal patterns. Replace with autoregressive transformer on user history.

**Adaptation:**
```python
class UserSequenceEncoder(nn.Module):
    def __init__(self):
        self.item_embeddings = nn.Embedding(n_items, d)
        self.transformer = TransformerDecoder(num_layers=4)
        # Pretrain on next-item prediction
    
    def encode_user(self, history_ids):
        x = self.item_embeddings(history_ids)
        x = self.transformer(x)
        return x  # [L, d] — sequence of context-aware embeddings
```

→ Feed this output to Q-Former instead of MF user embedding.

**Effort:** 1-2 weeks (new source model + pretraining).
**Probability:** 20-30%.

### 🥉 Idea 3 — Perceiver compression (already done by Q-Former)

USER-LLM uses Perceiver to compress N user tokens → K queries.

**SigLLM Q-Former already does this** — encode_cf compresses single CF vector (M=1) to 8 query tokens.

→ Could combine: M-token source (multi-token CF projection) + K-query compression via Q-Former.

### Idea 4 — Zero-init scalar gating

Apply gating pattern to existing soft token integration:
```python
self.alpha = nn.Parameter(torch.zeros(1))
soft_tokens_contribution = torch.tanh(self.alpha) * soft_tokens
```

→ Safe init, learn to use soft tokens only if CTR rewards it.

**Effort:** 5 lines code.
**Probability:** 10-15%.

---

## 9. Key takeaways for SigLLM thesis

### What USER-LLM validates for SigLLM design
1. **Q-Former-like compression is valid** — Perceiver-style query compression mirrors Q-Former approach
2. **Cross-modal-style alignment** of user behavior with LLM works
3. **Frozen LLM + lightweight adapter** is the right efficiency direction

### What USER-LLM challenges for SigLLM
1. **Single input-layer injection may be limiting** — USER-LLM gets +16% by cross-attending at multiple layers
2. **Static MF may underperform** sequential encoders for temporal user understanding
3. **78x inference speedup** highlights cost of text-translation approaches (SigLLM avoids this with soft tokens, but cross-attention would be even better)

### Most actionable for SigLLM
- **Idea 1 (gated cross-attention)** is highest theoretical value but highest engineering cost
- **Idea 4 (zero-init scalar gating)** is cheapest and could be added to current SigLLM with minimal effort

---

## 10. Citation key

```bibtex
@inproceedings{ning2024userllm,
  title={USER-LLM: Efficient LLM Contextualization with User Embeddings},
  author={Ning, Lin and Liu, Luyang and Wu, Jiaxing and Wu, Neo and others},
  booktitle={AAAI},
  year={2024}
}
```
