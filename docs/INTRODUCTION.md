# Introduction — Paper 1: A Cross-Task Benchmark for Clinical EEG Foundation Models

**Series note:** This is **Paper 1 of 4** in the clinical EEG foundation-model series. It is the first paper and does not build on any former paper. Papers 2 (seizure forecasting), 3 (clinical risk detection), and 4 (fairness/generalizability audit) all build on this paper's encoder and evaluation harness.

## Background

The Temple University EEG Corpus (TUEG, ~60–70k clinical EEGs from >15k patients, free academic license) enables self-supervised pretraining at a scale previously impossible for academic labs. EEG foundation models (EEGFormer, CLEF, BrainBERT, LaBraM) emerged in 2024–26, but most are window-level and single-task. A cross-task, session-scale clinical benchmark is open. Unlike saturated MRI brain-age work, EEG modeling is a young niche.

## Research questions

1. Do SSL representations pretrained on TUEG transfer across seizure detection, abnormality classification, and sleep staging?
2. Where do foundation models beat task-specific models, and where do they not?
3. How much does session-scale (vs. window-level) context matter?

## Data

| Resource | Scale | Access |
|---|---|---|
| TUEG / TUSZ / TUAB | ~60–70k EEGs | Free license (form + rsync) |
| CHB-MIT, Sleep-EDF, NSRR | small/open | PhysioNet |

## Methods

Masked modeling / contrastive SSL, linear probing + fine-tuning, fixed cross-task evaluation harness, task-specific baselines.

## Expected contributions

- An open cross-task clinical-EEG benchmark and encoder reused by Papers 2–4.

## Scope and boundary

Planning, software, and synthetic tests live here. Clinical EEG data are handled under the TUEG license; no diagnostic claims are made.
