"""
features.py  (DE — Data Engineering)
-------------------------------------
Turns raw per-keystroke rows into:
  1. A per-keystroke feature sequence for the LSTM (X), one row per key,
     including that key's deviation from the user's personal baseline.
  2. Per-user baselines (mean/std of resting inter-key timing), used both
     to build training features and to score brand-new live sessions.

This module is imported by both train.py (offline) and infer_live.py
(online), so training and live inference always compute features the
same way.
"""
import numpy as np
import pandas as pd

SEQ_LEN = 40  # keystrokes per session (matches generator)
FEATURES = ["inter_key_ms", "dwell_ms", "is_backspace", "is_error", "baseline_dev"]


def compute_user_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline = each user's typical rhythm, estimated from their calm
    (label == 0) sessions only — exactly like a real system would seed a
    baseline from a user's early, presumably-normal usage."""
    calm = df[df["label"] == 0]
    baseline = calm.groupby("user_id")["inter_key_ms"].agg(["mean", "std"]).reset_index()
    baseline.columns = ["user_id", "baseline_mean_ms", "baseline_std_ms"]
    baseline["baseline_std_ms"] = baseline["baseline_std_ms"].clip(lower=5.0)
    return baseline


def build_sequences(df: pd.DataFrame, baselines: pd.DataFrame):
    """Returns X (n_sessions, SEQ_LEN, n_features), y (n_sessions,), and
    session_ids in matching order."""
    merged = df.merge(baselines, on="user_id", how="left")
    # fall back to population baseline for any user we've never seen before
    pop_mean = baselines["baseline_mean_ms"].mean()
    pop_std = baselines["baseline_std_ms"].mean()
    merged["baseline_mean_ms"] = merged["baseline_mean_ms"].fillna(pop_mean)
    merged["baseline_std_ms"] = merged["baseline_std_ms"].fillna(pop_std)

    merged["baseline_dev"] = (
        (merged["inter_key_ms"] - merged["baseline_mean_ms"]) / merged["baseline_std_ms"]
    )

    X, y, sids = [], [], []
    for sid, g in merged.sort_values("key_index").groupby("session_id"):
        g = g.sort_values("key_index")
        if len(g) < SEQ_LEN:
            continue
        X.append(g[FEATURES].values[:SEQ_LEN])
        y.append(g["label"].iloc[0])
        sids.append(sid)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), sids


def fit_scaler(X: np.ndarray):
    """Simple per-feature standardization fit on the training set only."""
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0).clip(min=1e-6)
    return mean, std


def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / std
