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

**Current status:** implementation stage; the SSL framework (channel-masked reconstruction pretraining, linear-probe/fine-tune evaluation, benchmark runner) is implemented and validated on synthetic EEG; TUEG license form to be submitted; no corpus experiments have been run.

## What is included

| Path | Contents |
|---|---|
| `src/eegfm/data.py` | EEG window dataset interface; channel-set standardization (canonical 10-20 montage harmonization). |
| `src/eegfm/simulate.py` | Synthetic EEG generator with planted class-discriminative cross-channel phase-coupling patterns. |
| `src/eegfm/models.py` | Compact transformer encoder (channel×time patch embedding, positional encoding), linear probe head, channel-masked reconstruction wrapper. |
| `src/eegfm/pretrain.py` | MAE-style channel-masked reconstruction SSL pretraining loop. |
| `src/eegfm/finetune.py` | Linear-probe (frozen encoder) and fine-tune (unfrozen) evaluation. |
| `src/eegfm/benchmark.py` | Benchmark runner: SSL-pretrained probe vs. from-scratch supervised baseline; AUROC/AUPRC results table. |
| `tests/` | Synthetic-signal tests: forward shapes, pretraining loss reduction, pretrained probe beats from-scratch baseline. |
| `docs/` | Research status, methods scope, data access (TUH DUA) guidance, and contribution guidance. |

## Use and validation

```bash
pip install -e ".[dev]"
python -m pytest -q
```

All tests run on CPU in under a minute using synthetic signals; no corpus access is required. See [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md) for TUH EEG Corpus access (free for research, signed DUA with Temple University required).

## Keywords

EEG, foundation models, self-supervised learning, seizure detection, clinical neurophysiology, computational neuroscience, reproducible research.

## Documentation

- [Introduction for new readers](docs/INTRODUCTION.md)
- [TUH EEG Corpus data access](docs/DATA_ACCESS.md)
- [Contributing](CONTRIBUTING.md)
