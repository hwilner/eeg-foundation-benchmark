# Contributing

Thank you for your interest in contributing to the EEG Foundation Model
Benchmark (Paper 1 of the clinical EEG foundation-model series).

## Getting started

```bash
git clone https://github.com/hwilner/eeg-foundation-benchmark.git
cd eeg-foundation-benchmark
pip install -e ".[dev]"
python -m pytest -q
```

All tests run on CPU in under a minute using small synthetic signals — no
corpus access is required for development.

## Development conventions

- Source lives in `src/eegfm/` (src layout); tests live in `tests/`.
- Keep everything data-free: new functionality must be testable with the
  synthetic EEG generator (`eegfm.simulate.generate_synthetic_eeg`).
- `python -m pytest -q` must pass before opening a pull request.
- Keep changes focused; follow the existing module boundaries
  (data / models / pretrain / finetune / benchmark).

## Task workflow

Work is tracked as GitHub issues with task cards (see the issue templates).
Pick an open issue, assign yourself, and reference the issue in your PR.

Note that TUAB/TUSZ/TUEV/TUEG data access requires a signed data use
agreement with Temple University — see `docs/DATA_ACCESS.md`. Do not
commit any corpus data or derived artifacts.

## Scientific review

Analysis and benchmarking changes (leaderboards, metrics, evaluation
harness behavior) require scientific-owner review before merge, per the
backlog gates.
