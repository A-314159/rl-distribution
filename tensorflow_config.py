import os, subprocess, sys, textwrap

# ------------------------------------------------------------------------
# Configure tensorflow
# ------------------------------------------------------------------------

# Instructions:
# Call configure once in the __main__ file before importing the rest of the project
# Then use @tf_compile as decorator for functions where you want to apply it
#
# ------------------------------------------------------------------------
# Example: in __main__
#
# from project_config import configure
#  if __name__ == "__main__":
#    configure(mode="eager")
#    import train           # rest of your project
#    train.run()
#
# ------------------------------------------------------------------------
# Example: in any other file:
#
# from project_config import tf_compile
# @tf_ compile
# def train_step(model, x, y):
#    ...
# project_config.py


_MODE = "graph"
_REDUCE_RETRACING = True
_JIT_DEFAULT = False
_PRECISION = "float32"
_DATASET_DETERMINISTIC = True


# ---------- Public API ----------

def configure(*,
              mode: str = "graph",
              use_onednn: bool = True,
              use_gpu: bool = True,
              reduce_retracing: bool = True,
              jit_compile: bool = False,
              log_level: str = "2",
              precision: str = "float32",  # "float32" | "float64" | "mixed16"
              num_threads_CPU: int = None,
              deterministic_ops: bool = False,
              deterministic_shuffling: bool = True,
              seed: int = None,
              show_diagnostic: bool = False):
    """
    Global TensorFlow runtime configuration.
    For reproducibility:
        a) with CPU only: set a seed is enough
        b) with GPU: set a seed, set deterministic_ops=True,
                     set deterministic_shuffling=True (if using the function make_dataset below)
    For debugging: set mode='eager'
    :param mode: 'eager' (use for debugging), 'graph' (fast), 'graph_xla' (faster in some cases)
    :param use_onednn: True to apply intel OneDNN on CPU (no effect on GPU)
    :param use_gpu:
    :param reduce_retracing: True (faster)
    :param jit_compile: True (faster)
    :param log_level:
    :param precision: "float32" | "float64" | "mixed16"
    :param num_threads_CPU: integer to limit the number of threads  (for better speed) if running on CPU
    :param deterministic_ops: True (slower) for reproducibility (debug, publications) to prevent non-deterministic GPU kernels (e.g. atomic adds)
    :param deterministic_shuffling: False is better (True if reproducibility is needed, or while debuggin)
    :param seed: int (-1 for no seed) (Set the seed for numpy, tensorflow and random)
    :param show_diagnostic: True to show results of configuration
    :return:
    """

    global _REDUCE_RETRACING, _JIT_DEFAULT, _PRECISION, _DATASET_DETERMINISTIC
    _REDUCE_RETRACING = reduce_retracing
    _JIT_DEFAULT = bool(jit_compile) or (mode == "graph_xla")
    _PRECISION = precision
    _DATASET_DETERMINISTIC = deterministic_shuffling

    if "tensorflow" in sys.modules:
        raise RuntimeError("configure() must run before importing TensorFlow")

    # ---- Env flags (must be set before TF import) ----
    if use_onednn:
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
    else:
        os.environ.pop("TF_ENABLE_ONEDNN_OPTS", None)

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", log_level)

    if deterministic_ops:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    else:
        os.environ.pop("TF_DETERMINISTIC_OPS", None)

    # ---- Import TF AFTER env flags ----
    import tensorflow as tf

    # GPU visibility
    _use_gpu_internal(use_gpu)

    # Execution mode
    set_tf_mode(mode)

    # Precision
    if precision == "float64":
        tf.keras.backend.set_floatx("float64")
    elif precision == "float32":
        tf.keras.backend.set_floatx("float32")
    elif precision == "mixed16":
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
    else:
        raise ValueError("precision must be 'float32', 'float64', or 'mixed16'")

    # CPU threading
    if num_threads_CPU is not None:
        tf.config.threading.set_intra_op_parallelism_threads(num_threads_CPU)
        tf.config.threading.set_inter_op_parallelism_threads(num_threads_CPU)

    # Seeds
    if seed != -1:
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    # Dataset determinism (only applied if you call dataset.with_options())
    options = tf.data.Options()
    options.experimental_deterministic = deterministic_shuffling

    if show_diagnostic:
        diagnostic()


def set_tf_mode(mode: str):
    import tensorflow as tf
    global _MODE
    if mode not in {"eager", "graph", "graph_xla"}:
        raise ValueError("mode must be 'eager', 'graph', or 'graph_xla'")
    _MODE = mode
    tf.config.run_functions_eagerly(mode == "eager")


def tf_compile(fn=None, *, reduce_retracing=None, jit=None):
    """
        Decorator: uses per-function overrides if provided, else global defaults.
        - reduce_retracing: True/False or None (use global)
        - jit: True/False or None (use global)
    """

    def _wrap(f):
        import tensorflow as tf  # deferred import
        if _MODE == "eager":
            return f
        rr = _REDUCE_RETRACING if (reduce_retracing is None) else bool(reduce_retracing)
        jc = _JIT_DEFAULT if (jit is None) else bool(jit)
        return tf.function(f, reduce_retracing=rr, jit_compile=jc)

    return _wrap(fn) if fn is not None else _wrap


# ---------- Helpers ----------

def _use_gpu_internal(use_gpu: bool):
    """
    Configure visible GPU(s). Call only once, before any tensors/models are created.
    """
    import tensorflow as tf

    if use_gpu is False:
        # Hide all GPUs → force CPU
        tf.config.set_visible_devices([], "GPU")
        print("Forcing CPU only (no GPUs visible).")
        return

    # Determine which GPU index to use
    gpu_index = 0 if use_gpu is True else int(use_gpu)

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU found; running on CPU.")
        return

    if gpu_index < 0 or gpu_index >= len(gpus):
        print(f"Requested GPU index {gpu_index} not available; using CPU.")
        tf.config.set_visible_devices([], "GPU")
        return

    try:
        tf.config.set_visible_devices(gpus[gpu_index], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
        print(f"Using GPU: {gpus[gpu_index]}")
    except RuntimeError as e:
        # Happens if called after GPUs already initialized
        print("GPU config error (likely called too late):", e)


# ------------------------------------------------------------------------
# Check configuration
# ------------------------------------------------------------------------

def _run_cmd(cmd: str) -> str:
    try:
        out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
        return out.strip()
    except subprocess.CalledProcessError as e:
        return f"[error] {e.output.strip()}"


def diagnostic():
    import tensorflow as tf
    print(textwrap.dedent(f"""
    Python exe: {sys.executable}
    Sys version: {sys.version.split()[0]}
    Pip exe: {_run_cmd('python -m pip --version')}
    Pip show TF: {_run_cmd('python -m pip show tensorflow')}
    TF_ENABLE_ONEDNN_OPTS: {os.environ.get('TF_ENABLE_ONEDNN_OPTS')}
    """))

    print("TF imported:", tf.__version__)
    print("CPU devices:", tf.config.list_physical_devices("CPU"))
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("GPUs visible:", tf.config.list_physical_devices("GPU"))


# -------------------------------------------------------------------------
# Example of usage of make_dataset
#
# from project_config import configure, make_dataset
#
# if __name__ == "__main__":
#    configure(mode="graph", deterministic_shuffling=False, show_diag=True)
#
#    import numpy as np
#    x = np.random.randn(1000, 4).astype("float32")
#    y = np.sin(x[:, 0:1]).astype("float32")

#    ds = make_dataset(x, y, batch_size=64, shuffle=True, cache=True)
#    for xb, yb in ds.take(1):
#        print("Batch:", xb.shape, yb.shape)
# -------------------------------------------------------------------------

def make_dataset(data, labels=None, batch_size=32, shuffle=True, cache=False, prefetch=True):
    """
    Convenience helper to build a tf.data.Dataset with global options applied.

    Args:
        data: array-like or tensor (features)
        labels: array-like or tensor (targets), or None
        batch_size: int
        shuffle: bool (whether to shuffle before batching)
        cache: bool (cache dataset in memory)
        prefetch: bool (whether to prefetch for overlap with compute)

    Returns:
        A tf.data.Dataset ready for training.
    """
    import tensorflow as tf
    if labels is not None:
        ds = tf.data.Dataset.from_tensor_slices((data, labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(data)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(data), reshuffle_each_iteration=True)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size)
    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    # Apply global deterministic option
    options = tf.data.Options()
    options.experimental_deterministic = _DATASET_DETERMINISTIC
    ds = ds.with_options(options)

    return ds
