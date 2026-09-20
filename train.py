"""
train.py
---------
Runs the full offline pipeline: load raw data -> DE feature engineering
-> DSP-protected storage -> DL training -> save model + scaler + baselines
for use by infer_live.py and the dashboard.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from features import compute_user_baselines, build_sequences, fit_scaler, apply_scaler, FEATURES, SEQ_LEN
from model import build_lstm
import dsp


def main():
    df = pd.read_csv("data/raw_keystrokes.csv")

    # ---- DE: baselines + sequences ----
    baselines = compute_user_baselines(df)
    X, y, session_ids = build_sequences(df, baselines)
    print(f"Built {X.shape[0]} sessions, sequence length {X.shape[1]}, {X.shape[2]} features/step")

    X_train, X_test, y_train, y_test, sid_train, sid_test = train_test_split(
        X, y, session_ids, test_size=0.2, random_state=42, stratify=y
    )

    mean, std = fit_scaler(X_train)
    X_train_s = apply_scaler(X_train, mean, std)
    X_test_s = apply_scaler(X_test, mean, std)

    # ---- DSP: encrypt the engineered feature store + pseudonymize users ----
    os.makedirs("models", exist_ok=True)
    baselines_pseudo = baselines.copy()
    baselines_pseudo["user_id"] = baselines_pseudo["user_id"].apply(dsp.pseudonymize_id)
    key = dsp.get_or_create_key()
    payload = baselines_pseudo.to_json(orient="records").encode()
    with open("models/baselines.enc", "wb") as f:
        f.write(dsp.encrypt_bytes(payload, key))
    # keep a plain copy too, only because infer_live.py needs to map a live
    # (non-hashed) user_id back to its baseline in this demo
    baselines.to_csv("models/baselines_plain.csv", index=False)
    print("Baselines pseudonymized + encrypted at models/baselines.enc")

    # ---- DL: train the LSTM ----
    model = build_lstm(SEQ_LEN, len(FEATURES))
    history = model.fit(
        X_train_s, y_train,
        validation_split=0.15,
        epochs=25,
        batch_size=16,
        verbose=2,
    )

    y_pred_prob = model.predict(X_test_s, verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.3f}   F1: {f1:.3f}")
    print("Confusion matrix [ [TN FP] [FN TP] ]:\n", cm)

    model.save("models/lstm_model.keras")
    np.savez("models/scaler.npz", mean=mean, std=std)
    with open("models/metrics.json", "w") as f:
        json.dump({"accuracy": acc, "f1": f1, "confusion_matrix": cm.tolist()}, f, indent=2)

    print("\nSaved: models/lstm_model.keras, models/scaler.npz, models/metrics.json")


if __name__ == "__main__":
    main()
