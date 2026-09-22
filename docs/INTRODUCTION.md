# Introduction — Paper 1: A Cross-Task Benchmark for Clinical EEG Foundation Models

**Series note:** This is **Paper 1 of 4** in the clinical EEG foundation-model series. It is the first paper and does not build on any former paper. Papers 2 (seizure forecasting), 3 (clinical risk detection), and 4 (fairness/generalizability audit) all build on this paper's encoder and evaluation harness.

**Concept figure:** the pipeline of this paper — raw clinical EEG (TUEG) → harmonized preprocessing → self-supervised transformer encoder (masked channel reconstruction + contrastive objectives) → TUAB/TUSZ/TUEV downstream tasks, all scored under one unified benchmark harness. See [figures/concept_figure.md](figures/concept_figure.md) for a faithful Mermaid rendering (the FigForge/NeurIPS-style PNG could not be committed with the current text-only tooling; the Mermaid version depicts the identical pipeline).

## Background

Electroencephalography (EEG) is the workhorse of clinical neurophysiology. Routinely acquired at the bedside, in epilepsy monitoring units, and in outpatient clinics, it captures electrical brain activity with millisecond resolution at a fraction of the cost of any neuroimaging modality. Yet its interpretation remains a profound bottleneck: reading an EEG requires years of fellowship training, inter-rater agreement among experts is modest, and the global shortage of clinical neurophysiologists means that many recordings are interpreted late or not at all. Automating even part of this workflow — triage, abnormality flagging, seizure detection, event marking — would have immediate clinical value.

Deep learning offered an early answer to this bottleneck. Convolutional networks applied to raw EEG achieved accuracy competitive with classical feature-engineering pipelines [2], and compact architectures such as EEGNet showed that useful EEG decoders can be trained with few parameters [13]. Systematic reviews of this literature, however, document recurring pathologies: small single-site datasets, heterogeneous preprocessing, leakage-prone splits, and limited external validation [3, 4]. Most published EEG models are trained from scratch on a few dozen to a few hundred subjects and consequently fail to transfer.

In natural language processing, this exact stagnation was broken by the pretraining paradigm: models such as BERT are first trained on massive unlabeled corpora with self-supervised objectives (masked-token prediction), then adapted to downstream tasks with modest labeled data [5]. Masked modeling and contrastive learning have since transformed computer vision and speech. EEG, despite being a signal domain with abundant unlabeled data, has lagged behind. The obstacles are real: recordings vary in channel count, montage, and sampling rate; clinical EEG is contaminated by artifacts and physiological confounds; and labels are noisy. The situation changed when large clinical corpora became available — above all the Temple University Hospital EEG Corpus (TUEG), which aggregates more than 60,000 clinical EEG recordings from over 15,000 patients collected in routine hospital care [1]. TUEG and its annotated subcorpora provide both the scale needed for pretraining and curated evaluation benchmarks: the TUH Abnormal EEG Corpus (TUAB) for normal/abnormal classification [12], the TUH EEG Seizure Corpus (TUSZ) for seizure detection [11], and the TUH EEG Events Corpus (TUEV) for fine-grained event classification [1].

A first wave of EEG foundation models has begun to exploit this scale. BENDr adapted the wav2vec contrastive framework to tens of thousands of TUEG recordings [9]. BIOT proposed a unified "biosignal sentence" tokenization enabling cross-dataset learning [6]. LaBraM introduced vector-quantized masked modeling over a very large, heterogeneous EEG collection [8], EEGPT combined masked reconstruction with spatio-temporal alignment and demonstrated broad downstream transfer [7], and CBraMod further refined criss-cross attention over EEG patches [14]. These models are promising, but they are typically evaluated on heterogeneous, often window-level, small benchmarks, with inconsistent preprocessing and splits. What the field lacks — and what this paper provides — is a reproducible, cross-task, session-scale benchmark on standardized TUH evaluation sets, comparing pretrained foundation encoders against strong supervised baselines under a single, MLSPred-Bench-style harness with fixed splits, metrics, and statistical testing.

## Prior work and gap

Task-specific supervised models remain the state of the art on several TUH benchmarks. On TUAB, deep convolutional networks trained from scratch reach roughly 85–87% balanced accuracy for pathology decoding [10]; on TUSZ, specialized architectures dominate seizure detection leaderboards [11]. Foundation models [6–9, 14] report competitive or superior results, but each paper uses different windows, montages, splits, and baselines, making cross-paper comparison unreliable. Three gaps persist: (1) no shared evaluation harness spans TUAB, TUSZ, and TUEV under identical conditions; (2) head-to-head comparisons of pretraining objectives (masked modeling vs. contrastive) on clinical tasks are rare; and (3) the benefit of pretraining at session scale — long recordings with realistic label proportions — is largely untested.

## Research questions

1. Do self-supervised representations pretrained on TUEG transfer across seizure detection (TUSZ), abnormality classification (TUAB), and event classification (TUEV) better than training from scratch?
2. Which pretraining objective — masked reconstruction or contrastive instance discrimination — yields the most transferable clinical EEG representations, and at what data and compute scale?
3. Where do foundation encoders beat task-specific supervised baselines, and where do they not?
4. How much does session-scale context, versus window-level classification, matter for each task?

## Data

| Resource | Content | Scale | Access |
|---|---|---|---|
| TUEG [1] | Unlabeled clinical EEG for pretraining | ~60,000+ recordings, >15,000 patients | Free academic license (form + rsync) |
| TUAB [10, 12] | Normal/abnormal labels | ~3,000 recordings, official train/eval split | Free academic license |
| TUSZ [11] | Expert seizure annotations | ~5,700 files, >3,000 seizures | Free academic license |
| TUEV [1] | Six event types (SPSW, GPED, PLED, EYEM, ARTF, BCKG) | ~500 recordings | Free academic license |
| External controls (CHB-MIT, Sleep-EDF) | Sanity checks / cross-dataset transfer | Small, open | PhysioNet |

## Methods

We pretrain a transformer encoder on unlabeled TUEG recordings using both masked-patch reconstruction and contrastive objectives, following the design space established by BENDr, BIOT, LaBraM, and EEGPT [6–9] but re-implemented under one codebase with shared tokenization (common average reference, resampling, 10–20 montage harmonization). Downstream evaluation follows an MLSPred-Bench-style protocol: fixed official splits, subject-disjoint folds, patient-level leakage checks, and a frozen evaluation harness. Adaptation strategies include linear probing, partial fine-tuning, and full fine-tuning. Baselines comprise EEGNet [13], deep and shallow ConvNets [2], the TUAB pathology-decoding reference pipeline [10], and published foundation-model results where strictly comparable. Primary metrics are balanced accuracy and AUROC (TUAB), event- and overlap-based sensitivity with false-alarm rate (TUSZ), and macro-F1 (TUEV), all reported with bootstrap confidence intervals. Ablations vary pretraining corpus size, objective, and session- versus window-level context.

## Expected contributions

- A unified, open, reproducible benchmark for clinical EEG foundation models across TUAB, TUSZ, and TUEV, with fixed splits, baselines, and statistical comparison procedures.
- An encoder and evaluation harness reused throughout the series: Paper 2 adds seizure-forecasting heads on top, Paper 3 builds clinical risk detection on the same representations, and Paper 4 audits the resulting models for fairness and generalizability.
- Empirical evidence on when self-supervised pretraining on TUEG helps, which objective transfers best, and how session-scale context changes performance.

## Scope and boundary

This repository contains planning documents, software, and synthetic-data tests. Clinical EEG data are handled exclusively under the TUEG license and are never redistributed. All results are research-grade; no diagnostic or clinical-deployment claims are made.

## References

[1] Obeid I, Picone J. The Temple University Hospital EEG Data Corpus. *Frontiers in Neuroscience* 10:196, 2016. doi:10.3389/fnins.2016.00196.

[2] Schirrmeister RT, Springenberg JT, Fiederer LDJ, Glasstetter M, Eggensperger K, Tangermann M, Hutter F, Burgard W, Ball T. Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping* 38(11):5391–5420, 2017. doi:10.1002/hbm.23730.

[3] Craik A, He Y, Contreras-Vidal JL. Deep learning for electroencephalogram (EEG) classification tasks: a review. *Journal of Neural Engineering* 16(3):031001, 2019. doi:10.1088/1741-2552/ab0ab5.

[4] Roy Y, Banville H, Albuquerque I, Gramfort A, Falk TH, Faubert J. Deep learning-based electroencephalography analysis: a systematic review. *Journal of Neural Engineering* 16(5):051001, 2019. doi:10.1088/1741-2552/ab260c.

[5] Devlin J, Chang MW, Lee K, Toutanova K. BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT*, 2019. arXiv:1810.04805. doi:10.18653/v1/N19-1423.

[6] Yang C, Westover MB, Sun J. BIOT: Biosignal transformer for cross-data learning in the wild. *Advances in Neural Information Processing Systems* 36:78240–78260, 2023. arXiv:2305.10351.

[7] Wang G, Liu W, He Y, Xu C, Ma L, Li H. EEGPT: Pretrained transformer for universal and reliable representation of EEG signals. *Advances in Neural Information Processing Systems* 37:39249–39280, 2024.

[8] Jiang WB, Zhao LM, Lu BL. Large brain model for learning generic representations with tremendous EEG data in BCI (LaBraM). *International Conference on Learning Representations (ICLR)*, 2024. arXiv:2405.18765.

[9] Kostas D, Aroca-Ouellette S, Rudzicz F. BENDr: Using transformers and a contrastive self-supervised learning task to learn from massive amounts of EEG data. *Frontiers in Human Neuroscience* 15:653659, 2021. doi:10.3389/fnhum.2021.653659.

[10] Gemein LAW, Schirrmeister RT, Chrabąszcz P, Wilson D, Boedecker J, Schulze-Bonhage A, Hutter F, Ball T. Machine-learning-based diagnostics of EEG pathology. *NeuroImage* 220:117021, 2020. doi:10.1016/j.neuroimage.2020.117021.

[11] Shah V, von Weltin E, Lopez S, McHugh JR, Veloso L, Golmohammadi M, Obeid I, Picone J. The Temple University Hospital Seizure Detection Corpus. *Frontiers in Neuroinformatics* 12:83, 2018. doi:10.3389/fninf.2018.00083.

[12] López de Diego S. Automated interpretation of abnormal adult electroencephalograms. MS Thesis, Temple University, 2017.

[13] Lawhern VJ, Solon AJ, Waytowich NR, Gordon SM, Hung CP, Lance BJ. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering* 15(5):056013, 2018. doi:10.1088/1741-2552/aace8c.

[14] Wang J, Zhao S, Luo Z, Zhou Y, Jiang H, Li S, Li T, Pan G. CBraMod: A criss-cross brain foundation model for EEG decoding. *International Conference on Learning Representations (ICLR)*, 2025. arXiv:2412.07236.
