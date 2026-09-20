# Mental-State Detection via Typing Behavior — Working Demo

A full DE → DSP → DL → DV pipeline that estimates behavioral strain from
keystroke timing. Built with a **synthetic dataset** so it runs end-to-end
right now, with every module written so real captured keystroke data can
drop in later without changing the pipeline shape.

**This produces a behavioral estimate, not a medical diagnosis.**

## What's actually real here vs. simulated

| Part | Status |
|---|---|
| Feature engineering, baselines | Real code, runs on real timing data |
| Privacy controls (hash, encrypt, DP noise, minimization) | Real, actually applied to the data on disk |
| LSTM model | Real TensorFlow/Keras model, really trained, really evaluated |
| Dashboard | Real Streamlit app reading real logged results |
| **Training data** | **Synthetic** — simulated keystroke timing for 40 users, generated with the statistical signature (slower speed, more pauses/backspaces) that keystroke-dynamics research associates with cognitive load. Swap in real captured data later without touching the rest of the pipeline. |
| **Live typing capture** | Simulated per-session for the demo (`infer_live.py` generates a fresh session in the same way the training data was made). Wiring up a real in-browser keystroke capture is the next step — see "Next steps" below. |

## Setup

```bash
pip install tensorflow-cpu streamlit scikit-learn pandas numpy cryptography
```

## Run it

```bash
# 1. Generate the synthetic dataset
python3 src/generate_data.py

# 2. Run the DE -> DSP -> DL pipeline (trains and saves the LSTM)
python3 src/train.py

# 3. Try a live "session" from the command line
python3 src/infer_live.py user_003 1     # 1 = simulate a strained session
python3 src/infer_live.py user_003 0     # 0 = simulate a calm session

# 4. Launch the dashboard
streamlit run src/dashboard.py
```

Open the URL Streamlit prints (usually http://localhost:8501). Use the
sidebar to simulate sessions and watch the trend charts, baseline
deviation, and strain score update.

## Project structure

```
src/
  generate_data.py   # synthetic keystroke dataset (DE input)
  features.py        # DE: feature engineering + personal baselines
  dsp.py              # DSP: pseudonymization, encryption, DP noise, minimization
  model.py            # DL: LSTM architecture
  train.py            # runs DE -> DSP -> DL end-to-end, saves model
  infer_live.py       # scores a new/live session with the trained model
  dashboard.py        # DV: Streamlit dashboard
data/
  raw_keystrokes.csv  # generated synthetic data
  session_log.csv     # log of scored live sessions (dashboard reads this)
models/
  lstm_model.keras    # trained model
  scaler.npz          # feature scaler
  baselines.enc       # encrypted, pseudonymized per-user baselines
  baselines_plain.csv # plain copy (demo only — see note below)
  metrics.json        # test accuracy / F1 / confusion matrix
```

## Honest limitations to mention if asked

- The dataset is synthetic. Results (99% test accuracy) reflect how
  separable the *simulated* classes are, not real-world performance —
  say this plainly rather than presenting it as a validated accuracy.
- `baselines_plain.csv` exists only so this demo can look up a user's
  baseline by their real ID without wiring up a full key-management
  service. In production, only the encrypted, pseudonymized store
  (`baselines.enc`) should exist.
- Differential privacy noise (`dsp.add_differential_privacy_noise`) is
  implemented but only meant for aggregate/reporting numbers, not
  individual live scores — using it on a single user's score would
  make the estimate useless.

## Next steps (if you have more time later)

1. Replace `make_live_session()` in `infer_live.py` with a real
   in-browser keystroke capture (a JS `keydown`/`keyup` listener posting
   timestamps to a small API) feeding the same `features.py` functions.
2. Replace the synthetic dataset with real sessions + a short self-report
   survey (e.g. PANAS) once you can collect consented data.
3. Swap `baselines_plain.csv` for a proper encrypted key-value store.
