from __future__ import annotations

import logging
from typing import Any


def configure_tensorflow(
    model_params: dict[str, Any],
    *,
    random_seed: int,
) -> Any:
    """Apply the shared TensorFlow seed, GPU-memory, and precision policy."""

    import tensorflow as tf

    tf.keras.utils.set_random_seed(random_seed)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        if model_params.get("use_gpu", False):
            logging.warning("GPU requested but no TensorFlow GPU device is available")
        return tf

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            logging.debug("TensorFlow GPU memory policy was already initialized")

    if model_params.get("mixed_precision", False):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logging.info("Enabled TensorFlow mixed precision")
    return tf
