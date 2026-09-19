# EEG Foundation Model Benchmark (Paper 1)

This independent research repository plans and tracks a benchmark of self-supervised EEG foundation models across clinical downstream tasks, using the Temple University EEG Corpus family. It provides evaluation utilities for transparent review and extension.

## Series position

This is **Paper 1** of the clinical EEG foundation-model series (4 papers). It is the foundation of the series; Papers 2–4 build on its pretrained encoders and evaluation harness.

## Research plan

| Planned work | Expected outcome |
|---|---|
| TUEG access + preprocessing pipeline (license: free for research) | Versioned preprocessed corpus. |
| Adapt/pretrain SSL encoder (masked modeling / contrastive; fine-tune BrainBERT/LaBraM-class checkpoints) | Versioned encoder checkpoints. |
| Benchmark across downstream tasks: TUSZ seizure detection, TUAB abnormality, Sleep-EDF staging, CHB-MIT validation | Cross-task leaderboard vs. task-specific baselines. |
| Release evaluation harness | Reused by Papers 2–4. |

**Current status:** planning stage; TUEG license form to be submitted; no experiments have been run.

## What is included

| Path | Contents |
|---|---|
| `src/` | Preprocessing, SSL training, and evaluation utilities. |
| `tests/` | Synthetic signal tests. |
| `docs/` | Research status, methods scope, and contribution guidance. |

## Use and validation

```bash
python -m pytest -q
```

## Keywords

EEG, foundation models, self-supervised learning, seizure detection, clinical neurophysiology, computational neuroscience, reproducible research.

## Documentation

- [Introduction for new readers](docs/INTRODUCTION.md)
