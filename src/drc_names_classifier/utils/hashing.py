from __future__ import annotations

import hashlib

import numpy as np
import polars as pl


def stable_text_buckets(
    values: pl.Series,
    *,
    namespace: bytes,
    bucket_count: int,
) -> np.ndarray:
    """Map strings to reproducible buckets independently of dataframe versions."""

    if len(namespace) > 16:
        raise ValueError("BLAKE2 namespace must contain at most 16 bytes")
    if bucket_count <= 0:
        raise ValueError("bucket_count must be greater than zero")

    def bucket(value: str) -> int:
        digest = hashlib.blake2b(
            value.encode("utf-8"),
            digest_size=8,
            person=namespace,
        ).digest()
        return int.from_bytes(digest, "little") % bucket_count

    return np.fromiter(
        (bucket(value) for value in values.fill_null("").cast(pl.String)),
        dtype=np.uint64,
        count=len(values),
    )
