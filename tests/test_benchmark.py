"""Test benchmark."""

from sklearn.metrics import roc_auc_score

from eegfm.benchmark import run_benchmark
from eegfm.data import EEGWindowDataset
from eegfm.finetune import evaluate_linear_probe
from eegfm.models import EEGTransformerEncoder
from eegfm.pretrain import pretrain_mae, train_test_split
from eegfm.simulate import generate_synthetic_eeg

MODEL_KW = dict(n_channels=8, n_times=128, patch_len=16, d_model=32, n_heads=4, n_layers=1)


def test_pretrained_probe_beats_from_scratch():
    # Realistic SSL setting: abundant unlabeled windows, limited labels.
    # The label lives in cross-channel phase coupling (see simulate.py):
    # per-channel power is uninformative, so a from-scratch supervised
    # model cannot find the pattern at this scale, while channel-masked
    # reconstruction pretraining must learn it.
    """Test pretrained probe beats from scratch."""
    X_unlabeled, _ = generate_synthetic_eeg(n_samples=256, n_channels=8, n_times=128, seed=99)
    X, y = generate_synthetic_eeg(n_samples=360, n_channels=8, n_times=128, seed=1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_frac=0.6, seed=1)

    pre = EEGTransformerEncoder(**MODEL_KW)
    pretrain_mae(pre, EEGWindowDataset(X_unlabeled), epochs=60, batch_size=32, lr=5e-3, seed=0)
    p_pre = evaluate_linear_probe(pre, X_tr, y_tr, X_te, y_te, freeze_encoder=True, epochs=60, seed=0)

    scr = EEGTransformerEncoder(**MODEL_KW)
    p_scr = evaluate_linear_probe(scr, X_tr, y_tr, X_te, y_te, freeze_encoder=False, epochs=30, seed=0)

    auroc_pre = roc_auc_score(y_te, p_pre)
    auroc_scr = roc_auc_score(y_te, p_scr)
    assert auroc_pre > 0.8  # probe actually reads out the planted pattern
    assert auroc_pre > auroc_scr


def test_run_benchmark_table():
    """Test run benchmark table."""
    X_unlabeled, _ = generate_synthetic_eeg(n_samples=160, n_channels=8, n_times=128, seed=7)
    X, y = generate_synthetic_eeg(n_samples=300, n_channels=8, n_times=128, seed=1)
    df = run_benchmark(
        X, y, X_unlabeled=X_unlabeled, pretrain_epochs=40, probe_epochs=40, seed=1,
        **MODEL_KW,
    )
    assert list(df.columns) == ["method", "auroc", "auprc"]
    assert len(df) == 2
    assert df["auroc"].between(0, 1).all()
    pre = df.loc[df["method"] == "ssl-pretrained + linear-probe", "auroc"].iloc[0]
    scr = df.loc[df["method"] == "from-scratch supervised", "auroc"].iloc[0]
    assert pre > scr
