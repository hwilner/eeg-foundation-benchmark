"""Benchmark runner: SSL-pretrained encoder vs from-scratch baseline.

Pipeline on a labeled window dataset:
  1. Channel-masked reconstruction pretraining on unlabeled windows
     (labels unused). Pass `X_unlabeled` to mimic the realistic setting
     of abundant unlabeled EEG; defaults to the training split.
  2. Linear-probe evaluation of the pretrained encoder (frozen).
  3. From-scratch supervised baseline: identical architecture, randomly
     initialized, trained end-to-end on the labeled split only.
Reports AUROC/AUPRC in a results table (pandas DataFrame).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import EEGWindowDataset
from .finetune import evaluate_linear_probe
from .models import EEGTransformerEncoder
from .pretrain import pretrain_mae, train_test_split


def _metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
    }


def run_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    X_unlabeled: np.ndarray | None = None,
    pretrain_epochs: int = 60,
    probe_epochs: int = 30,
    test_frac: float = 0.6,
    seed: int = 0,
    **model_kwargs,
) -> pd.DataFrame:
    """Compare pretrained-then-probed vs from-scratch on the same split.

    Returns a DataFrame with columns [method, auroc, auprc].
    """
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_frac=test_frac, seed=seed)
    if X_unlabeled is None:
        X_unlabeled = X_tr
    n_channels, n_times = X.shape[1], X.shape[2]
    model_kwargs.setdefault("n_channels", n_channels)
    model_kwargs.setdefault("n_times", n_times)

    # 1. Pretrained encoder + linear probe (frozen encoder)
    pretrained = EEGTransformerEncoder(**model_kwargs)
    pretrain_mae(
        pretrained, EEGWindowDataset(X_unlabeled), epochs=pretrain_epochs, seed=seed
    )
    p_pre = evaluate_linear_probe(
        pretrained, X_tr, y_tr, X_te, y_te, freeze_encoder=True,
        epochs=probe_epochs, seed=seed,
    )

    # 2. From-scratch baseline: same architecture, randomly initialized,
    #    trained end-to-end with the linear head on labeled data only.
    scratch = EEGTransformerEncoder(**model_kwargs)
    p_scr = evaluate_linear_probe(
        scratch, X_tr, y_tr, X_te, y_te, freeze_encoder=False,
        epochs=probe_epochs, seed=seed,
    )

    rows = [
        {"method": "ssl-pretrained + linear-probe", **_metrics(y_te, p_pre)},
        {"method": "from-scratch supervised", **_metrics(y_te, p_scr)},
    ]
    return pd.DataFrame(rows)
