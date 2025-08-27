# gnlm_demo.py
import tensorflow as tf
import numpy as np

# =========================================================
# 0) Global execution mode: "eager" | "graph" | "graph_xla"
# =========================================================
MODE = "graph"   # change this for debug vs speed

if MODE == "eager":
    tf.config.run_functions_eagerly(True)
    _DECORATOR = lambda f: f
elif MODE == "graph":
    tf.config.run_functions_eagerly(False)
    def _DECORATOR(f):
        return tf.function(f, reduce_retracing=True)
elif MODE == "graph_xla":
    tf.config.run_functions_eagerly(False)
    def _DECORATOR(f):
        return tf.function(f, reduce_retracing=True, jit_compile=True)
else:
    raise ValueError("MODE must be 'eager', 'graph', or 'graph_xla'")

# =========================================================
# 1) Model (your (4,32,32,1) MLP) & utilities
# =========================================================
def make_mlp():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(1)
    ])

def pack_vars(vars_list):
    return tf.concat([tf.reshape(v, [-1]) for v in vars_list], axis=0)

def unpack_like(vec, vars_like):
    parts, offset = [], 0
    for v in vars_like:
        size = tf.size(v)
        part = tf.reshape(vec[offset: offset + size], tf.shape(v))
        parts.append(part)
        offset += size
    return parts

def apply_delta(vars_list, delta_flat):
    for v, d in zip(vars_list, unpack_like(delta_flat, vars_list)):
        v.assign_add(d)

# Your residuals function (batched). Return a *flat* tensor.
def residuals(model, x, y):
    r = model(x) - y                      # [B, out]
    return tf.reshape(r, [-1])            # [B*out]

# =========================================================
# 2) GN/LM Optimizer class
#    - Works like a TF optimizer, but `minimize(...)` expects
#      a residuals_fn that returns the flattened residual vector.
# =========================================================
class GNLMOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, lam=1e-2, cg_tol=1e-6, cg_iters=50, name="GNLM", **kwargs):
        super().__init__(name, **kwargs)
        self.lam = tf.constant(lam, dtype=tf.float32)
        self.cg_tol = tf.constant(cg_tol, dtype=tf.float32)
        self.cg_iters = tf.constant(cg_iters, dtype=tf.int32)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(lam=float(self.lam.numpy()),
                        cg_tol=float(self.cg_tol.numpy()),
                        cg_iters=int(self.cg_iters.numpy())))
        return cfg

    # ---- CG solver (matrix-free): (A) v = J^T J v + lam v
    @_DECORATOR
    def _cg(self, matvec, b):
        x = tf.zeros_like(b)
        r = b - matvec(x)
        p = tf.identity(r)
        rs_old = tf.tensordot(r, r, 1)
        for _ in tf.range(self.cg_iters):
            Ap = matvec(p)
            alpha = rs_old / (tf.tensordot(p, Ap, 1) + 1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = tf.tensordot(r, r, 1)
            if tf.sqrt(rs_new) < self.cg_tol:
                break
            beta = rs_new / (rs_old + 1e-30)
            p = r + beta * p
            rs_old = rs_new
        return x

    # ---- One GN/LM step wrapped as "minimize"
    # residuals_fn() must return the *flattened residual vector* r (shape [B*out])
    @_DECORATOR
    def minimize(self, residuals_fn, var_list):
        # g = J^T r (gradient of mean 0.5||r||^2)
        with tf.GradientTape() as tape:
            r = residuals_fn()
            loss = 0.5 * tf.reduce_mean(tf.square(r))
        g_list = tape.gradient(loss, var_list)
        g_flat = pack_vars([tf.zeros_like(v) if g is None else g for v, g in zip(var_list, g_list)])
        b = -g_flat

        # matvec: v -> J^T J v + lam v  (no explicit Jacobian)
        def matvec(v_flat):
            v_list = unpack_like(v_flat, var_list)
            # Forward JVP: u = J v
            with tf.autodiff.ForwardAccumulator(primals=var_list, tangents=v_list) as acc:
                r = residuals_fn()
            u = acc.jvp(r)
            # Reverse VJP: J^T u
            with tf.GradientTape() as tape2:
                r2 = residuals_fn()
                inner = tf.reduce_sum(r2 * tf.stop_gradient(u)) / tf.cast(tf.size(r2), tf.float32)
            jtju_list = tape2.gradient(inner, var_list)
            jtju_flat = pack_vars([tf.zeros_like(v) if g is None else g for v, g in zip(var_list, jtju_list)])
            return jtju_flat + self.lam * v_flat

        # Solve for delta with CG and apply
        delta = self._cg(matvec, b)
        apply_delta(var_list, delta)
        return loss, tf.norm(g_flat), tf.norm(delta)

# =========================================================
# 3) Standard training step (Adam/SGD/etc.)
# =========================================================
@_DECORATOR
def standard_step(model, x, y, opt):
    with tf.GradientTape() as tape:
        r = residuals(model, x, y)
        loss = 0.5 * tf.reduce_mean(tf.square(r))
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# =========================================================
# 4) Demo usage
# =========================================================
if __name__ == "__main__":
    tf.keras.backend.set_floatx('float32')

    # Synthetic regression: y = sin(x0)
    N = 4096
    x_data = np.random.randn(N, 4).astype(np.float32)
    y_data = np.sin(x_data[:, :1]).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(128).repeat()
    it = iter(ds)

    model = make_mlp()

    # ---- Choose optimizer: Adam (standard) OR GN/LM
    USE_GNLM = True
    if USE_GNLM:
        opt = GNLMOptimizer(lam=1e-2, cg_tol=1e-6, cg_iters=50)
    else:
        opt = tf.keras.optimizers.Adam(1e-3)

    # ---- Training loop
    for step in range(1, 51):
        xb, yb = next(it)

        if isinstance(opt, GNLMOptimizer):
            # residuals_fn closes over current batch & model
            residuals_fn = lambda: residuals(model, xb, yb)
            loss, gnorm, dnorm = opt.minimize(residuals_fn, model.trainable_variables)
            if step % 10 == 0:
                print(f"[GN/LM] step {step:02d}  loss={loss.numpy():.6f}  ||g||={gnorm.numpy():.3e}  ||Δ||={dnorm.numpy():.3e}")
        else:
            loss = standard_step(model, xb, yb, opt)
            if step % 10 == 0:
                print(f"[Adam ] step {step:02d}  loss={loss.numpy():.6f}")

    # Optional: Adam warmup then GN/LM polish (simple pattern)
    # - Train some steps with Adam, then reassign `opt = GNLMOptimizer(...)`
    # - Keep the same loop; only the instance type check changes behavior.
