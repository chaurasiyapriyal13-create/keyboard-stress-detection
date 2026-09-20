"""
dashboard.py  (DV — Data Visualization)
------------------------------------------
Run with:  streamlit run src/dashboard.py

Shows: typing speed trend, baseline deviation, error patterns, model
results over time, and a plain-language user report — reading from the
same session log that infer_live.py writes to.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import streamlit as st
import pandas as pd
import numpy as np

from infer_live import make_live_session, score_session, log_result, plain_language_report, LOG_PATH

st.set_page_config(page_title="Mental-State Detection Dashboard", layout="wide")

st.title("Typing Behavior — Mental-State Dashboard")
st.caption("A behavioral estimate based on keystroke dynamics — not a medical diagnosis.")

baselines = pd.read_csv("models/baselines_plain.csv")
users = baselines["user_id"].tolist()

with st.sidebar:
    st.header("Simulate a session")
    st.caption(
        "Stands in for a real live-typing capture — generates one session "
        "with the chosen ground-truth state and runs it through the full "
        "DE → DL pipeline, exactly like a real captured session would be."
    )
    user_id = st.selectbox("User", users, index=3)
    state_choice = st.radio("Simulated state", ["Calm / Baseline", "Strained"])
    label_hint = 0 if state_choice == "Calm / Baseline" else 1
    if st.button("Run session", type="primary"):
        rng = np.random.default_rng()
        session = make_live_session(user_id, label_hint, rng)
        result = score_session(session)
        log_result(result)
        st.session_state["last_result"] = result

if not os.path.exists(LOG_PATH):
    st.info("No sessions yet — use the sidebar to simulate one.")
    st.stop()

log = pd.read_csv(LOG_PATH)
log["timestamp"] = pd.to_datetime(log["timestamp"])

if "last_result" in st.session_state:
    r = st.session_state["last_result"]
    st.success(plain_language_report(r))

user_log = log[log.user_id == user_id].sort_values("timestamp")

col1, col2, col3 = st.columns(3)
col1.metric("Sessions logged", len(user_log))
col2.metric("Latest strain score", f"{user_log['strain_score'].iloc[-1]:.1f}" if len(user_log) else "—")
col3.metric("Latest state", user_log["predicted_state"].iloc[-1] if len(user_log) else "—")

st.subheader("Typing speed trend")
st.line_chart(user_log.set_index("timestamp")[["avg_typing_speed_ms"]])

c1, c2 = st.columns(2)
with c1:
    st.subheader("Baseline deviation")
    st.bar_chart(user_log.set_index("timestamp")[["baseline_deviation"]])
with c2:
    st.subheader("Backspace / error rate")
    st.bar_chart(user_log.set_index("timestamp")[["backspace_rate"]])

st.subheader("Model results over time")
st.dataframe(
    user_log[["timestamp", "session_id", "strain_score", "predicted_state", "baseline_deviation"]]
    .sort_values("timestamp", ascending=False),
    use_container_width=True,
)

st.subheader("All users — latest strain score")
latest_per_user = log.sort_values("timestamp").groupby("user_id").tail(1)
st.bar_chart(latest_per_user.set_index("user_id")[["strain_score"]])
