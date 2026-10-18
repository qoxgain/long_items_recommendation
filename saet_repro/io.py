"""Small, strict artifact helpers shared by experiment runners."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping

import numpy as np


def json_safe(value):
    """Convert numpy values and undefined numbers to portable JSON values.

    Undefined ranking metrics are written as JSON null, never as a measured zero
    or as the nonstandard JSON tokens NaN/Infinity.
    """
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    """Atomically write readable, deterministic strict JSON; return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize before touching the destination so bad inputs cannot corrupt it.
    payload = json.dumps(json_safe(value), indent=2, sort_keys=True,
                         ensure_ascii=False, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def sha256(path, chunk_size=1024 * 1024):
    """Return the SHA-256 hex digest of the bytes in a local file."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
