"""Small reusable utilities shared across experiment modules."""

from ners.utils.io import read_json, write_json
from ners.utils.hashing import stable_text_buckets
from ners.utils.runtime import configure_tensorflow
from ners.utils.text import (
    NameView,
    coerce_name_view,
    full_name_array,
    full_name_series,
    name_view_expression,
    name_view_series,
    native_group_array,
    normalize_name_expression,
)

__all__ = [
    "configure_tensorflow",
    "coerce_name_view",
    "full_name_array",
    "full_name_series",
    "name_view_expression",
    "name_view_series",
    "native_group_array",
    "normalize_name_expression",
    "NameView",
    "read_json",
    "stable_text_buckets",
    "write_json",
]
