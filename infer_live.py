"""
infer_live.py
---------------
Simulates "a new user types, we score them live":
  1. Generate one fresh session (or accept one passed in) for a user
  2. Run it through the same DE feature pipeline used in training
  3. Load the trained LSTM and produce a behavioral-state + strain score
  4. Append the result to a session log the dashboard reads

Run directly for a demo: `python3 src/infer_live.py`
"""
import sys, os, json, datetime
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from tensorflow import keras

from features import compute_user_baselines, FEATURES, SEQ_LEN
from generate_data import simulate_session
import dsp

LOG_PATH = "data/session_log.csv"


def make_live_session(user_id: str, label_hint: int, rng: np.random.Generator):
    """Stand-in for real live typing capture: generates one session with
    the same statistical process used for training data, for demo
    purposes. Swap this out for real captured keystroke timings in
    production — everything downstream (features -> model -> dashboard)
    stays identical."""
    baselines = pd.read_csv("models/baselines_plain.csv")
    row = baselines[baselines.user_id == user_id]
    if row.empty:
        user_speed, user_var = 180.0, 45.0
    else:
        user_speed, user_var = row.iloc[0]["baseline_mean_ms"], row.iloc[0]["baseline_std_ms"]

    ikt, dwell, bksp, err = simulate_session(user_speed, user_var, label_hint)
    return pd.DataFrame({
        "user_id": user_id,
        "session_id": f"{user_id}_live_{rng.integers(1_000_000)}",
        "label": label_hint,
        "key_index": range(SEQ_LEN),
        "inter_key_ms": ikt,
        "dwell_ms": dwell,
        "is_backspace": bksp,
        "is_error": err,
    })


def score_session(session_df: pd.DataFrame) -> dict:
    baselines = pd.read_csv("models/baselines_plain.csv")
    b_row = baselines[baselines.user_id == session_df.user_id.iloc[0]]
    if b_row.empty:
        b_mean, b_std = baselines.baseline_mean_ms.mean(), baselines.baseline_std_ms.mean()
    else:
        b_mean, b_std = b_row.iloc[0][["baseline_mean_ms", "baseline_std_ms"]]

    session_df = session_df.copy()
    session_df["baseline_dev"] = (session_df["inter_key_ms"] - b_mean) / b_std
    X = session_df.sort_values("key_index")[FEATURES].values[:SEQ_LEN][None, ...].astype(np.float32)

    scaler = np.load("models/scaler.npz")
    X = (X - scaler["mean"]) / scaler["std"]

    model = keras.models.load_model("models/lstm_model.keras")
    prob = float(model.predict(X, verbose=0).ravel()[0])
    strain_score = round(prob * 100, 1)
    state = "Strained" if prob >= 0.5 else "Calm / Baseline"

    avg_dev = float(session_df["baseline_dev"].mean())
    return {
        "user_id": session_df.user_id.iloc[0],
        "user_id_pseudonymized": dsp.pseudonymize_id(session_df.user_id.iloc[0]),
        "session_id": session_df.session_id.iloc[0],
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "avg_typing_speed_ms": round(float(session_df["inter_key_ms"].mean()), 1),
        "backspace_rate": round(float(session_df["is_backspace"].mean()), 3),
        "baseline_deviation": round(avg_dev, 2),
        "strain_probability": round(prob, 4),
        "strain_score": strain_score,
        "predicted_state": state,
    }


def log_result(result: dict):
    os.makedirs("data", exist_ok=True)
    df_new = pd.DataFrame([result])
    if os.path.exists(LOG_PATH):
        df_new.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df_new.to_csv(LOG_PATH, index=False)


def plain_language_report(result: dict) -> str:
    if result["strain_score"] < 35:
        body = "Typing rhythm looks close to this user's normal baseline. No notable signs of strain."
    elif result["strain_score"] < 65:
        body = "Some deviation from baseline — slightly slower pace and a few more corrections than usual."
    else:
        body = "Typing shows a clear deviation from baseline: slower pace, longer pauses, and a higher correction rate."
    return (
        f"Session for {result['user_id']} — strain score {result['strain_score']}/100 "
        f"({result['predicted_state']}).\n{body}\n"
        "This is a behavioral estimate based on typing patterns, not a medical diagnosis."
    )


if __name__ == "__main__":
    rng = np.random.default_rng()
    user_id = sys.argv[1] if len(sys.argv) > 1 else "user_003"
    label_hint = int(sys.argv[2]) if len(sys.argv) > 2 else int(rng.integers(0, 2))

    session = make_live_session(user_id, label_hint, rng)
    result = score_session(session)
    log_result(result)

    print(json.dumps(result, indent=2))
    print("\n" + plain_language_report(result))
