"""
model.py  (DL — Deep Learning)
--------------------------------
A small LSTM classifier: reads a session as a sequence of per-keystroke
feature vectors and outputs a probability of "strained" state, which we
also report as a continuous 0-100 strain score.
"""
from tensorflow import keras
from tensorflow.keras import layers


def build_lstm(seq_len: int, n_features: int) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=(seq_len, n_features)),
        layers.Masking(mask_value=0.0),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(16),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
