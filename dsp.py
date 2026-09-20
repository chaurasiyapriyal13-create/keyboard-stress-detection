"""
dsp.py  (DSP — Data Security & Privacy)
-----------------------------------------
Minimal, real implementations of the four privacy controls named in the
project spec. These aren't decorative — pseudonymize_id() and the
Fernet encrypt/decrypt calls actually run on the feature store before
it touches disk.
"""
import hashlib
import os
import numpy as np
from cryptography.fernet import Fernet

SECRET_SALT = os.environ.get("MINDTYPE_SALT", "change-me-in-production")


def pseudonymize_id(user_id: str) -> str:
    """One-way hash: the real user_id is never stored alongside typing
    features. Not reversible without the salt + original id."""
    h = hashlib.sha256((SECRET_SALT + user_id).encode()).hexdigest()
    return f"u_{h[:16]}"


def get_or_create_key(path="models/dsp.key") -> bytes:
    if os.path.exists(path):
        return open(path, "rb").read()
    key = Fernet.generate_key()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(key)
    return key


def encrypt_bytes(data: bytes, key: bytes) -> bytes:
    return Fernet(key).encrypt(data)


def decrypt_bytes(token: bytes, key: bytes) -> bytes:
    return Fernet(key).decrypt(token)


def add_differential_privacy_noise(value: float, epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """Laplace-mechanism noise for aggregate stats (e.g. a cohort-level
    average strain score) so no single user's contribution is identifiable
    in a published aggregate. Not applied to per-user live inference —
    only to aggregate/reporting numbers, where it belongs."""
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return value + noise


def minimize_fields(record: dict) -> dict:
    """Data minimization: strip anything beyond what the model actually
    needs. Notably, raw key *content* is never present here at all —
    only timing metadata ever reaches this layer."""
    allowed = {"user_id", "session_id", "label", "key_index",
               "inter_key_ms", "dwell_ms", "is_backspace", "is_error"}
    return {k: v for k, v in record.items() if k in allowed}
