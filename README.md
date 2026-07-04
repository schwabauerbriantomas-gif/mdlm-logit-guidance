# Logit-Level Guidance on Masked Diffusion Language Models

Empirical characterization of energy-guided logit injection on **LLaDA-8B-Instruct**, an 8B-parameter masked diffusion language model.

## What This Is

An experiment applying logit-level guidance — a technique established by [Diffusion-LM](https://arxiv.org/abs/2205.14217) (Li & Liang, ACL 2022) — to a modern instruction-tuned masked diffusion model at 8B scale. The goal is to document how this technique behaves on a model of this size and capability, including its effectiveness, limitations, and architectural implications.

## Key Result

13 experiments were run using an autoresearch methodology (hypothesis → experiment → measure → keep/revert). The best configuration achieved:

- **+39% semantic steering** over baseline (target cosine similarity 0.26 vs 0.18)
- **75% quality rate** (outputs with target_sim > 0.15 AND coherence > 0.3)
- **Zero degenerate outputs** at the optimal parameter range

The optimal configuration: `logit_additive + cosine_all scoring + cosine alpha schedule + α=10`.

## How It Works

```
Standard MDLM denoising:
  for step in range(N):
      logits = model.forward(masked_input)
      probs  = softmax(logits / temperature)
      unmask(sample(probs, mask_positions))

Guided denoising:
  for step in range(N):
      logits = model.forward(masked_input)
      logits += alpha_schedule(step) * energy_scores   # injection
      probs  = softmax(logits / temperature)
      unmask(sample(probs, mask_positions))
```

Energy scores are computed once from target text via the model's own embedding matrix — a single matrix-vector product. No gradient computation, no classifier training, no fine-tuning.

## What Works and What Doesn't

| Topic | target_sim | Coherence | Why |
|-------|-----------|-----------|-----|
| Horror | **0.44** | 0.52 | Distinctive vocabulary competes weakly with model priors |
| Ocean | **0.34** | 0.67 | Distinctive vocabulary, moderate competition |
| Space | 0.18 | 0.51 | Common vocabulary, model priors dominate |
| Cooking | 0.07 | 0.57 | Common vocabulary, priors too strong to overcome |

**Fundamental limitation:** Logit injection steers vocabulary selection but not narrative planning. All guided outputs retain the model's default story template regardless of target.

## Findings

1. **α=10 is the sweet spot.** Below α=5: negligible effect. Above α=15: degenerate repetition collapse.
2. **Cosine alpha schedule outperforms constant and linear.** Guidance ramps 0→α_max→0 across denoising steps.
3. **Logit-space injection outperforms probability-space blending.** Additive logit modification preserves the model's distribution shape.
4. **Architectural property:** MDLMs expose N injection points per output (one per denoising step, typically 128) vs 1 per token in autoregressive models. This is structural, not a bug. It requires white-box access (model weights + embedding matrix).

## Reproduce

**Hardware:** GPU with ≥16GB VRAM (tested on RTX 3090)

```bash
git clone https://github.com/schwabauerbriantomas-gif/mdlm-logit-guidance.git
cd mdlm-logit-guidance
pip install dllm torch sentence-transformers

python src/guidance_experiment.py \
  --label "reproduce" \
  --strategy logit_additive \
  --alpha 10.0 \
  --schedule cosine \
  --norm abs_max \
  --score_method cosine_all \
  --trials 3
```

Each experiment takes ~7 minutes (2 min model load + 5 min generation).

## Repository Structure

```
├── src/
│   └── guidance_experiment.py    # Experiment driver (13 configurations supported)
├── data/
│   ├── e00_baseline.jsonl        # Raw results from all 13 experiments
│   ├── e01_zscore_linearup_a5.jsonl
│   ├── ...
│   └── e13_cosall_a10_blended.jsonl
├── notebooks/
│   └── analysis.ipynb            # Visualizations and statistical analysis
├── docs/
│   └── SECURITY_ANALYSIS.md      # Threat model and security framing
└── README.md
```

## Data Format

Each `.jsonl` file contains one JSON object per line:

```json
{
  "experiment": "horror",
  "trial": 0,
  "alpha": 10.0,
  "strategy": "logit_additive",
  "schedule": "cosine",
  "norm": "abs_max",
  "response": "Once upon a time...",
  "target_sim": 0.4529,
  "coherence": 0.4226,
  "diversity": 0.7913,
  "non_rep": 0.8621,
  "gen_time": 9.8
}
```

## Acknowledgments

- The guidance technique builds on **Diffusion-LM** (Li & Liang, ACL 2022, [arXiv:2205.14217](https://arxiv.org/abs/2205.14217))
- The model is **LLaDA-8B-Instruct** by Nie et al. ([arXiv:2502.09992](https://arxiv.org/abs/2502.09992))
- The experimental methodology follows **Karpathy's autoresearch** approach

## License

MIT
