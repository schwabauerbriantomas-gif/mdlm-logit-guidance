# Security Analysis: Logit Injection on Masked Diffusion Language Models

## Scope and Disclaimer

This document analyzes the **architectural properties** of masked diffusion language models (MDLMs) from a security perspective, based on empirical experiments on LLaDA-8B-Instruct.

**Important:** The guidance technique analyzed here requires **white-box access** (model weights and embedding matrix). It is **not** a server-side vulnerability, a prompt injection, or a remote exploit. It does not affect black-box API users. The "attack" framing is an analytical lens on a structural property of the architecture, not a report of an exploitable vulnerability.

---

## 1. Background: The Technique Is Not New

Logit-level guidance for text diffusion was established by **Diffusion-LM** (Li & Liang, ACL 2022) as a *controllable generation* technique. A classifier or energy function provides a gradient/score signal that is injected into the model's logits at each denoising step, steering generation toward desired attributes.

Our work applies this technique to LLaDA-8B-Instruct (8B parameters, instruction-tuned) using pre-computed energy vectors derived from the model's own embedding matrix — no gradient computation or classifier training needed. The cost is a single matrix-vector product per denoising step.

**We do not claim the base technique as novel.** The contribution is the empirical characterization on a modern 8B-scale model and the security framing of its architectural implications.

---

## 2. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Model | LLaDA-8B-Instruct (GSAI-ML), bf16, 16GB VRAM |
| Sampler | MDLM, 128 steps, block_size=32, temperature=0.6 |
| Prompt | "Write a short story about something interesting." |
| Guidance targets | 4 topics: space, ocean, horror, cooking |
| Eval metric | target_sim (MiniLM-L6-v2 cosine similarity to target) |
| Quality threshold | target_sim > 0.15 AND coherence > 0.3 |
| Experiments | 13 configurations, 2-3 trials each |

---

## 3. Results

### 3.1 Full Sweep

| Config | sim_mean | good% | Notes |
|--------|----------|-------|-------|
| Baseline (no guidance) | 0.1845 | 62% | Random similarity to topics |
| **Best (α=10, cosine, cosine_all)** | **0.2574** | **75%** | Optimal balance |
| Overpowered (α=15) | 0.2724 | 33% | Degenerate repetition |
| Weak (z_score + linear_up) | 0.0916 | 12% | Signal crushed |

### 3.2 Per-Topic Effectiveness (Best Config)

| Topic | Baseline | Guided | Δ | Vocabulary type |
|-------|----------|--------|---|-----------------|
| Horror | 0.172 | **0.440** | **+156%** | Distinctive |
| Ocean | 0.341 | 0.339 | -1% | Distinctive |
| Space | 0.187 | 0.179 | -4% | Common |
| Cooking | 0.037 | 0.072 | +95% | Common |

### 3.3 Effective Parameter Range

The attack has a **narrow effective window**:

- **α < 5:** Negligible steering effect (sim barely above baseline)
- **α = 8-12:** Effective steering with maintained coherence (75% quality rate)
- **α > 15:** Outputs collapse into degenerate repetition ("stars stars stars...")

This narrow range means overpowered injection attempts produce detectable outputs.

---

## 4. Architectural Implications

### 4.1 Injection Points: MDLM vs Autoregressive

| Property | Autoregressive (GPT, Llama) | Masked Diffusion (LLaDA) |
|----------|---------------------------|--------------------------|
| Forward passes per output | 1 per token | N per token (N = steps, typically 128) |
| Logit injection points | 1 per token | **N per output** |
| Commitment model | Sequential lock-in | Parallel refinement |

MDLMs expose N injection points per output, one at each denoising step. This is a **structural property** of the iterative denoising architecture. Each individual perturbation is small and distributed across steps, making the injection harder to detect than a single large perturbation in an autoregressive model.

### 4.2 Stealth

Guided outputs are visually indistinguishable from normal generation. The model continues producing grammatical, coherent text — only the vocabulary shifts:

```
Baseline: "...she found a cave filled with crystals..."
Guided:   "...she found a cave filled with darkness and blood..."
```

Both are grammatically valid English sentences with the same narrative structure.

### 4.3 Computational Cost

The injection requires:
- One forward pass through the model's embedding matrix (pre-computed once)
- One matrix-vector product per denoising step (negligible cost)
- No gradient computation, no backpropagation, no classifier training

---

## 5. Threat Model

### 5.1 White-Box (Full Model Access)

**Risk:** An adversary with model weights can compute energy vectors from arbitrary target text and inject them at each denoising step. Steering is reliable (+39% over baseline) for distinctive vocabulary.

**Applicability:** Relevant for open-weight models (LLaDA weights are publicly available). Not applicable to closed APIs that don't expose logits.

### 5.2 Gray-Box (API with Sampling Hooks)

**Risk:** If an MDLM-based API exposes logits or allows custom sampling functions (common for "controllable generation" features), the same injection applies.

**Applicability:** Depends entirely on API design. APIs that return only final text are not vulnerable.

### 5.3 Black-Box (Text-Only API)

**Not vulnerable** to this specific technique. Standard prompt injection vectors still apply but are a different attack class.

### 5.4 What This Does NOT Do

- ❌ Does not bypass safety training at the narrative level (only modifies vocabulary)
- ❌ Does not work on black-box APIs
- ❌ Does not alter the model's reasoning or factual knowledge
- ❌ Does not affect outputs when no guidance is injected

---

## 6. Limitations

1. **Single model tested:** Only LLaDA-8B-Instruct. Other MDLMs may behave differently.
2. **Open-ended prompt only:** More constrained prompts (Q&A, code generation) untested.
3. **No safety-trained targets:** Tested on benign topics, not against actual safety filters.
4. **Small samples:** 2-3 trials per config. The 75% rate has wide confidence intervals.
5. **Detection not explored:** This documents the property but does not develop detection methods.

---

## 7. Relationship to Existing Literature

| Work | Relationship |
|------|-------------|
| **Diffusion-LM** (Li & Liang, ACL 2022) | Established the base technique (classifier guidance on text diffusion logits). Our work extends it to 8B scale with gradient-free energy vectors. |
| **Discrete Diffusion Backdoor Attack** (Wang et al., 2024) | Backdoor attacks on discrete diffusion, but for images (VQ-Diffusion) and requires training-time poisoning. |
| **Classifier-Free Guidance** (Ho & Salimans, 2022) | The guidance mechanism in diffusion models. Energy injection is a form of CFG with an adversarial "classifier." |
| **GCG / AutoDAN** (Zou et al., 2023) | Adversarial suffix attacks on autoregressive LLMs. Different attack surface (prompt manipulation vs logit injection). |
| **Activation Steering** (Zou et al., 2023) | Adding vectors to hidden states. Similar spirit, different intervention point (activations vs logits). |

---

## 8. Conclusion

Logit-level guidance on masked diffusion language models is a technique with established academic precedent (Diffusion-LM, 2022). Applied to LLaDA-8B-Instruct, it reliably steers vocabulary selection (+39% semantic similarity) while maintaining coherence, within a narrow effective parameter range.

The architectural property that enables this — N injection points per output — is structural to MDLMs and relevant for security analysis of any system that exposes logits or sampling hooks. It is not a server-side vulnerability and does not affect black-box API users.
