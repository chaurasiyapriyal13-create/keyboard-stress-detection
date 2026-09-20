"""
generate_data.py
-----------------
Simulates keystroke-timing sessions for many users under two behavioral
states ("baseline" = normal typing, "strained" = cognitive/emotional load).

Why synthetic data: a real labeled keystroke + mental-state dataset takes
weeks of data collection with consent/IRB. For a working demo of the full
DE -> DSP -> DL -> DV pipeline, we simulate keystroke timing with the same
statistical signature research has found for cognitive load:
  - slower typing speed
  - longer, more variable inter-key intervals
  - more/longer pauses
  - higher backspace / error rate

Each row = one keystroke event within one session.
Columns: user_id, session_id, label, key_index, inter_key_ms, dwell_ms,
         is_backspace, is_error
"""
import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)

N_USERS = 40
SESSIONS_PER_USER = 14
KEYSTROKES_PER_SESSION = 40


def simulate_session(user_baseline_speed, user_baseline_var, label):
    """label: 0 = baseline/calm, 1 = strained"""
    n = KEYSTROKES_PER_SESSION

    if label == 0:
        speed_mult = RNG.normal(1.0, 0.05)
        var_mult = 1.0
        pause_rate = 0.04
        backspace_rate = 0.03
    else:
        speed_mult = RNG.normal(1.35, 0.12)   # slower -> larger inter-key gaps
        var_mult = RNG.uniform(1.6, 2.3)      # more erratic rhythm
        pause_rate = 0.14
        backspace_rate = 0.11

    base_ikt = user_baseline_speed * speed_mult
    base_var = user_baseline_var * var_mult

    inter_key = RNG.normal(base_ikt, base_var, n).clip(40, None)

    # inject hesitation pauses
    pause_mask = RNG.random(n) < pause_rate
    inter_key[pause_mask] += RNG.uniform(400, 1400, pause_mask.sum())

    dwell = RNG.normal(base_ikt * 0.55, base_var * 0.4, n).clip(20, None)

    is_backspace = (RNG.random(n) < backspace_rate).astype(int)
    is_error = ((RNG.random(n) < backspace_rate * 0.8) & (is_backspace == 0)).astype(int)

    return inter_key, dwell, is_backspace, is_error


def main():
    rows = []
    for uid in range(N_USERS):
        # each user has their own natural typing rhythm (this is what the
        # "personal baseline" in the DE stage will later have to learn)
        user_speed = RNG.normal(180, 30)      # ms between keys, baseline
        user_var = RNG.normal(45, 8)

        # a handful of strained sessions per user, rest baseline
        n_strained = RNG.integers(3, 6)
        labels = [0] * (SESSIONS_PER_USER - n_strained) + [1] * n_strained
        RNG.shuffle(labels)

        for sid, label in enumerate(labels):
            ikt, dwell, bksp, err = simulate_session(user_speed, user_var, label)
            for k in range(KEYSTROKES_PER_SESSION):
                rows.append((
                    f"user_{uid:03d}", f"user_{uid:03d}_s{sid:02d}", label, k,
                    round(float(ikt[k]), 2), round(float(dwell[k]), 2),
                    int(bksp[k]), int(err[k]),
                ))

    df = pd.DataFrame(rows, columns=[
        "user_id", "session_id", "label", "key_index",
        "inter_key_ms", "dwell_ms", "is_backspace", "is_error",
    ])
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/raw_keystrokes.csv", index=False)
    print(f"Generated {len(df)} keystroke rows across "
          f"{df['session_id'].nunique()} sessions, {df['user_id'].nunique()} users.")
    print(df.groupby("label")["session_id"].nunique())


if __name__ == "__main__":
    main()
