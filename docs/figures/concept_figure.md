# Concept Figure — Paper 1: A Cross-Task Benchmark for Clinical EEG Foundation Models

*(Binary figure placeholder: the generated 1536x1024 PNG concept figure could not be
committed because the available GitHub tooling only supports text content. This file
provides a faithful Mermaid rendering of the same pipeline. FigForge/NeurIPS-style
layout: raw EEG -> preprocessing -> self-supervised transformer encoder ->
TUAB/TUSZ/TUEV downstream tasks, all over a unified benchmark harness.)*

```mermaid
flowchart LR
    subgraph INPUT["Raw EEG (TUEG, 60k+ recordings)"]
        H[Head with scalp electrodes]
        W1[Channel 1: wiggly voltage trace]
        W2[Channel 2: wiggly voltage trace]
        W3[Channel M: wiggly voltage trace]
    end
    INPUT --> PRE["Preprocessing<br/>resample to 128 Hz<br/>common average reference<br/>10-20 montage harmonization"]
    PRE --> ENC
    subgraph ENC["Transformer encoder — self-supervised pretraining (no labels)"]
        P[Patchify: channel x time tiles]
        MASK[Hide ~40% of channels<br/>with mask token]
        TR[Transformer encoder<br/>attention across patches]
        REC[Small decoder reconstructs<br/>masked channels]
        CON[Contrastive objective:<br/>two views of same recording<br/>pulled together]
        P --> MASK --> TR --> REC
        TR --> CON
    end
    ENC --> T1["TUAB<br/>abnormal vs normal<br/>(balanced accuracy, AUROC)"]
    ENC --> T2["TUSZ<br/>seizure detection<br/>(sensitivity, false alarms/hr)"]
    ENC --> T3["TUEV<br/>six-way event classification<br/>(macro-F1)"]
    T1 --> BH["Unified benchmark harness<br/>fixed official splits, subject-disjoint folds<br/>supervised baselines (EEGNet, ConvNets)<br/>bootstrap confidence intervals"]
    T2 --> BH
    T3 --> BH
```
