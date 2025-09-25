from __future__ import annotations
import numpy as np

def to_scalar_float(x):
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return None
        return float(arr.reshape(-1)[-1])
    except Exception:
        try:
            return float(x)
        except Exception:
            return None
