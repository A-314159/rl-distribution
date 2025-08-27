#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F-critic training with:
A) TF-native data & caches (on-device) to avoid CPU<->GPU churn
B) sqrt(T-t) feature (parents & children) => 4-dim inputs
C) Child y' out-of-range override: F_child := 0/1
D) x-sampling bounds h(k), l(k) with (a,b,c)
E-F) x_child out-of-band override using precomputed PV at child

Keeps: mixed precision, hint blending, precompute, fused forward, logging, save/load, MC chart
"""

import os, sys, time, json, math, argparse, csv, threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------
# Precision
# -------------------------
def set_precision(policy: str):
    if policy == "float64":
        tf.keras.backend.set_floatx("float64")
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    elif policy == "mixed_float16":
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
    else:
        tf.keras.backend.set_floatx("float32")

def is_mixed() -> bool:
    try:
        from tensorflow.keras import mixed_precision
        return mixed_precision.global_policy().name == "mixed_float16"
    except Exception:
        return False

# -------------------------
# Black–Scholes, dtype-safe (compute in float32)
# -------------------------
SQRT2 = np.sqrt(2.0)
def std_norm_cdf(z): return 0.5*(1.0 + tf.math.erf(z / SQRT2))
def _cast_all(dtype, *xs): return [tf.cast(tf.convert_to_tensor(x), dtype=dtype) for x in xs]

@tf.function
def bs_call_price(S, K, sigma, tau):
    compute_dtype = tf.float32
    out_dtype = tf.keras.backend.floatx()
    S, K, sigma, tau = _cast_all(compute_dtype, S, K, sigma, tau)
    eps = tf.constant(1e-12, dtype=compute_dtype)
    sqrt_tau = tf.sqrt(tf.maximum(tau, eps))
    d1 = (tf.math.log(tf.maximum(S, eps)/K) + 0.5*sigma*sigma*tau) / (sigma*sqrt_tau + eps)
    d2 = d1 - sigma*sqrt_tau
    price = S*std_norm_cdf(d1) - K*std_norm_cdf(d2)
    return tf.cast(price, out_dtype)

@tf.function
def bs_delta(S, K, sigma, tau):
    compute_dtype = tf.float32
    out_dtype = tf.keras.backend.floatx()
    S, K, sigma, tau = _cast_all(compute_dtype, S, K, sigma, tau)
    eps = tf.constant(1e-12, dtype=compute_dtype)
    sqrt_tau = tf.sqrt(tf.maximum(tau, eps))
    d1 = (tf.math.log(tf.maximum(S, eps)/K) + 0.5*sigma*sigma*tau) / (sigma*sqrt_tau + eps)
    return tf.cast(std_norm_cdf(d1), out_dtype)

# -------------------------
# Universe
# -------------------------
@dataclass
class UniverseBS:
    sigma: float
    T: float
    P: int
    K: float = 1.0
    @property
    def h(self): return self.T / self.P

# -------------------------
# Train & sampler cfg
# -------------------------
@dataclass
class SamplerConfig:
    N: int = 60000
    x0: float = 0.0
    a: float = 0.6   # meta a
    b: float = 2.0   # meta b
    c: float = 0.0   # meta c
    r0: float = 0.02
    r1: float = 0.002
    seed: int = 42

@dataclass
class TrainConfig:
    precision: str = "float32"
    optimizer: str = "adam"
    lr: float = 1e-3
    batch_size: int = 1024
    max_epochs: int = 100
    max_time_sec: int = 600
    loss_tol_sqrt: float = 1e-4
    hidden: Tuple[int,...] = (32,32)
    activation: str = "tanh"
    children_method: str = "binomial"  # (binomial only here for TF-native, GH can be added similarly)
    n_children: int = 2
    lambda_hint_init: float = 1.0
    lambda_hint_final: float = 0.0
    lambda_hint_anneal_epochs: int = 10
    model_dir: str = "saved_F_model_v2"
    log_csv: str = "training_log.csv"
    chart_pdf: str = "comparison.pdf"
    eval_pairs: List[Tuple[int, float]] = None
    mc_paths: int = 20000
    precompute_children: bool = True
    fused_forward: bool = True

# -------------------------
# Actor (BS delta), TF-return
# -------------------------
class ActorBSDelta:
    def __init__(self, K): self.K = K
    def __call__(self, t_idx_tf: tf.Tensor, x_tf: tf.Tensor, u: UniverseBS) -> tf.Tensor:
        tau = tf.cast(u.T, x_tf.dtype) - tf.cast(t_idx_tf, x_tf.dtype) * tf.cast(u.h, x_tf.dtype)
        S = tf.exp(x_tf)
        return -bs_delta(S, tf.cast(u.K, x_tf.dtype), tf.cast(u.sigma, x_tf.dtype), tau)

# -------------------------
# Children generator (TF-native, binomial)
# -------------------------
@tf.function
def children_binomial(t_idx, x, u_sigma, u_h):
    """Return x_children (N,2) and probs (N,2)."""
    mu = -0.5 * (u_sigma**2) * u_h
    d = u_sigma * tf.sqrt(u_h)
    x_up   = x + (mu + d)
    x_down = x + (mu - d)
    x_children = tf.stack([x_up, x_down], axis=1)
    probs = tf.ones_like(x_children) * 0.5
    return x_children, probs

# -------------------------
# Build model (4 inputs: t, x, y, sqrt(T-t))
# -------------------------
def build_F_model(input_dim=4, hidden=(32,32), activation="tanh", dtype=None):
    if dtype is None: dtype = tf.keras.backend.floatx()
    inputs = keras.Input(shape=(input_dim,), dtype=dtype, name="features")
    norm = layers.Normalization(axis=-1, name="norm")
    x = norm(inputs)
    for i,h in enumerate(hidden):
        x = layers.Dense(h, activation=activation, name=f"dense_{i}")(x)
    out = layers.Dense(1, activation="sigmoid", name="cdf")(x)
    return keras.Model(inputs, out, name="F_critic")

def hint_cdf(y, mu, sigma):
    z = (y - mu) / (sigma + 1e-12)
    return 0.5 * (1.0 + tf.math.erf(z / np.sqrt(2.0)))

# -------------------------
# TF-native sampling with bounds h(k), l(k)
# -------------------------
def tf_sample_dataset(cfg: SamplerConfig, u: UniverseBS):
    """Returns a dict of TF tensors on the current device."""
    N = cfg.N
    rng = tf.random.Generator.from_seed(cfg.seed)

    t_idx = rng.uniform(shape=(N,), minval=0, maxval=u.P, dtype=tf.int32)
    k = tf.cast(t_idx, tf.float32)
    tau = tf.cast(u.T, tf.float32) - k * tf.cast(u.h, tf.float32)
    # bounds for x
    center = tf.math.log(tf.cast(u.K, tf.float32)) + 0.5*(u.sigma**2)*tau
    band   = cfg.b * u.sigma * tf.sqrt(tau + cfg.c)
    dlin   = u.sigma * tf.sqrt(u.h) * k
    hi = tf.maximum(cfg.x0 + cfg.a + dlin, center + band)
    lo = tf.minimum(cfg.x0 - cfg.a - dlin, center - band)
    # uniform sample in [lo,hi]
    u01 = rng.uniform(shape=(N,), minval=0., maxval=1., dtype=tf.float32)
    x = lo + u01 * (hi - lo)

    # y sampling around mu = -C_BS(t,x)
    S = tf.exp(tf.cast(x, tf.keras.backend.floatx()))
    tau_dtype = tf.cast(tau, tf.keras.backend.floatx())
    mu = -bs_call_price(S, tf.cast(u.K, tf.keras.backend.floatx()), tf.cast(u.sigma, tf.keras.backend.floatx()), tau_dtype)

    # schedule for y_half
    y_half = cfg.r1 + (1.0 - tau / u.T) * (cfg.r0 - cfg.r1)
    y_half = tf.cast(y_half, tf.keras.backend.floatx())

    u02 = tf.cast(rng.uniform(shape=(N,), minval=0., maxval=1., dtype=tf.float32), tf.keras.backend.floatx())
    y = (mu - y_half) + u02 * (2.0 * y_half)

    # sqrt_tau feature
    sqrt_tau = tf.sqrt(tf.cast(tau, tf.keras.backend.floatx()))

    # Cast to policy dtype (keep t_idx int32)
    def castp(z):
        if is_mixed(): return tf.cast(z, tf.float16)
        return tf.cast(z, tf.keras.backend.floatx())

    data = dict(
        t_idx=t_idx,
        x=castp(x),
        y=castp(y),
        mu=castp(mu),
        y_half=castp(y_half),
        sqrt_tau=castp(sqrt_tau),
    )
    return data

# -------------------------
# Precompute children, masks, PV (TF-native)
# -------------------------
def tf_precompute_children_cache(data, u: UniverseBS, cfg: SamplerConfig):
    """Adds child tensors: x_children, probs, t_children, terminal, dS, payoff, lprime,
       tau_child, sqrt_tau_child, PV_child, y_range_child (lo/up), masks for overrides.
    """
    t_idx = data["t_idx"]; x = data["x"]

    # children (binomial)
    x_children, probs = children_binomial(t_idx, x, tf.cast(u.sigma, x.dtype), tf.cast(u.h, x.dtype))
    t_children = tf.expand_dims(t_idx + 1, axis=-1)
    terminal = tf.cast(tf.equal(t_children, u.P), tf.int32)

    S = tf.exp(x)[:, None]
    S_child = tf.exp(x_children)
    dS = S_child - S

    # actor q (TF)
    actor = ActorBSDelta(u.K)
    q = actor(t_idx, x, u)  # (N,)

    payoff = tf.where(tf.equal(terminal, 1), tf.maximum(S_child - tf.cast(u.K, S_child.dtype), 0.0), 0.0)
    lprime = - q[:, None] * dS - payoff

    # child tau, sqrt_tau
    tau_child = tf.cast(u.T, x.dtype) - tf.cast(t_children, x.dtype) * tf.cast(u.h, x.dtype)
    sqrt_tau_child = tf.sqrt(tf.maximum(tau_child, tf.cast(0.0, x.dtype)))

    # child PV and child y-range (for C, D)
    mu_child = -bs_call_price(S_child, tf.cast(u.K, x_children.dtype), tf.cast(u.sigma, x_children.dtype), tf.cast(tau_child, x_children.dtype))
    # schedule for y_half at child
    y_half_child = cfg.r1 + (1.0 - (tf.cast(t_children, tf.float32) * u.h) / u.T) * (cfg.r0 - cfg.r1)
    y_half_child = tf.cast(y_half_child, x_children.dtype)

    y_lo_child = mu_child - y_half_child
    y_up_child = mu_child + y_half_child

    # x_child band masks for (E-F):
    center_child = tf.math.log(tf.cast(u.K, x_children.dtype)) + 0.5*(u.sigma**2)*tf.cast(tau_child, x_children.dtype)
    band_child   = cfg.b * u.sigma * tf.sqrt(tf.cast(tau_child, tf.float32) + cfg.c)
    band_child   = tf.cast(band_child, x_children.dtype)
    x_band_lo = center_child - band_child
    x_band_hi = center_child + band_child
    mask_x_out = tf.logical_or(x_children < x_band_lo, x_children > x_band_hi)  # (N,2) bool

    # store
    cache = dict(
        x_children=x_children, probs=probs, t_children=t_children, terminal=terminal,
        dS=dS, payoff=payoff, lprime=lprime, tau_child=tau_child, sqrt_tau_child=sqrt_tau_child,
        PV_child=-mu_child,  # PV = C, and mu_child = -C
        y_lo_child=y_lo_child, y_up_child=y_up_child,
        x_band_lo=x_band_lo, x_band_hi=x_band_hi, mask_x_out=mask_x_out
    )
    data.update(cache)
    return data

# -------------------------
# Hotkeys
# -------------------------
class HotKeys:
    def __init__(self):
        self.q_stop=False; self.show_chart=False
        threading.Thread(target=self._listen, daemon=True).start()
    def _listen(self):
        try:
            for line in sys.stdin:
                s=line.strip().lower()
                if s=="q": self.q_stop=True
                elif s=="l": self.show_chart=True
        except Exception:
            pass

# -------------------------
# Model, optimizer, lambda
# -------------------------
def build_model_and_adapt(data, dtype=None, hidden=(32,32), activation="tanh"):
    model = build_F_model(4, hidden, activation, dtype or tf.keras.backend.floatx())
    # Adapt normalization on a TF tensor (parents)
    feats_null = tf.stack([tf.cast(data["t_idx"], tf.keras.backend.floatx()),
                           tf.cast(data["x"], tf.keras.backend.floatx()),
                           tf.cast(data["y"], tf.keras.backend.floatx()),
                           tf.cast(data["sqrt_tau"], tf.keras.backend.floatx())], axis=1)
    model.get_layer("norm").adapt(feats_null)
    return model

def make_optimizer(name, lr):
    name=name.lower()
    if name=="adam": return keras.optimizers.Adam(lr)
    if name=="sgd": return keras.optimizers.SGD(lr, momentum=0.9, nesterov=True)
    if name=="rmsprop": return keras.optimizers.RMSprop(lr)
    raise ValueError(name)

def current_lambda(ep, cfg: TrainConfig):
    if cfg.lambda_hint_anneal_epochs <= 0: return cfg.lambda_hint_final
    if ep >= cfg.lambda_hint_anneal_epochs: return cfg.lambda_hint_final
    frac = ep/float(cfg.lambda_hint_anneal_epochs)
    return cfg.lambda_hint_init*(1.0-frac) + cfg.lambda_hint_final*frac

# -------------------------
# Training loop (TF tensors, precompute, overrides)
# -------------------------
def train_F(u: UniverseBS, samp: SamplerConfig, cfg: TrainConfig, reload_model=False):
    set_precision(cfg.precision)
    out_dir = Path(cfg.model_dir); out_dir.mkdir(parents=True, exist_ok=True)

    data = tf_sample_dataset(samp, u)
    if cfg.precompute_children:
        data = tf_precompute_children_cache(data, u, samp)

    # Save hyperparams
    with open(out_dir/"hyperparams.json","w") as f:
        json.dump({"universe":u.__dict__, "sampler":asdict(samp), "train":asdict(cfg)}, f, indent=2)

    # Build / load model
    if reload_model and (out_dir/"model.keras").exists():
        model = keras.models.load_model(out_dir/"model.keras")
    else:
        model = build_model_and_adapt(data, dtype=tf.keras.backend.floatx(), hidden=cfg.hidden, activation=cfg.activation)

    opt = make_optimizer(cfg.optimizer, cfg.lr)

    # CSV logging
    csv_path = out_dir / cfg.log_csv
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","elapsed_sec","rmse_loss","lambda_hint"])

    hot = HotKeys()
    t0 = time.time()
    N = int(data["t_idx"].shape[0])
    bsz = cfg.batch_size

    # convenience handles
    k_dtype = model.input.dtype

    @tf.function
    def forward_batch(idx_batch, lam):
        # Gather parents
        t_b = tf.gather(data["t_idx"], idx_batch)
        x_b = tf.gather(data["x"], idx_batch)
        y_b = tf.gather(data["y"], idx_batch)
        mu_b = tf.gather(data["mu"], idx_batch)
        yhalf_b = tf.gather(data["y_half"], idx_batch)
        sqrt_tau_b = tf.gather(data["sqrt_tau"], idx_batch)

        parent_feats = tf.stack([tf.cast(t_b, k_dtype),
                                 tf.cast(x_b, k_dtype),
                                 tf.cast(y_b, k_dtype),
                                 tf.cast(sqrt_tau_b, k_dtype)], axis=1)

        # Children (cached)
        tchild_b = tf.gather(data["t_children"], idx_batch)        # (B,1)
        xchild_b = tf.gather(data["x_children"], idx_batch)        # (B,2)
        lprime_b = tf.gather(data["lprime"], idx_batch)            # (B,2)
        probs_b  = tf.gather(data["probs"], idx_batch)             # (B,2)
        terminal_b = tf.gather(data["terminal"], idx_batch)        # (B,1)

        ychild_b = tf.cast(y_b[:,None], xchild_b.dtype) - lprime_b
        sqrt_tau_child_b = tf.gather(data["sqrt_tau_child"], idx_batch)  # (B,2)

        # child ranges and PV/masks
        y_lo_child_b = tf.gather(data["y_lo_child"], idx_batch)
        y_up_child_b = tf.gather(data["y_up_child"], idx_batch)
        PV_child_b   = tf.gather(data["PV_child"], idx_batch)
        mask_x_out_b = tf.gather(data["mask_x_out"], idx_batch)    # bool

        # Build child feats: [t', x', y', sqrt_tau']
        child_feats = tf.stack([
            tf.cast(tf.squeeze(tchild_b, axis=-1), k_dtype),
            tf.cast(xchild_b, k_dtype),
            tf.cast(ychild_b, k_dtype),
            tf.cast(sqrt_tau_child_b, k_dtype)
        ], axis=-1)  # (B,2,4)
        # Flatten
        child_feats_flat = tf.reshape(child_feats, (-1, 4))
        parent_in = parent_feats
        child_in  = child_feats_flat

        # Fused forward
        fused_in = tf.concat([parent_in, child_in], axis=0)
        fused_out = tf.squeeze(model(fused_in, training=True), axis=-1)
        F_parent = fused_out[:tf.shape(parent_in)[0]]
        F_child_raw = fused_out[tf.shape(parent_in)[0]:]
        F_child_raw = tf.reshape(F_child_raw, tf.shape(xchild_b)[:2])  # (B,2)

        # ----- Overrides -----
        # C) y' outside child sampling range -> 0/1
        mask_lo = tf.cast(ychild_b <= y_lo_child_b, tf.bool)
        mask_hi = tf.cast(ychild_b >= y_up_child_b, tf.bool)
        override_y = tf.where(mask_hi, tf.ones_like(F_child_raw), tf.zeros_like(F_child_raw))
        mask_y_any = tf.logical_or(mask_lo, mask_hi)
        F_child_after_y = tf.where(mask_y_any, override_y, F_child_raw)

        # E-F) x_child out of band -> 1 if y' > -PV, else 0
        cond_gt = tf.cast(ychild_b > (-PV_child_b), tf.float32)
        override_x = tf.cast(cond_gt, F_child_after_y.dtype)
        F_child_final = tf.where(mask_x_out_b, override_x, F_child_after_y)

        # Targets
        indicator = tf.cast(ychild_b > 0.0, F_child_final.dtype)
        probs_tf = tf.cast(probs_b, F_child_final.dtype)
        term_mask = tf.cast(tf.squeeze(terminal_b, axis=-1), tf.bool)

        Y_term = tf.reduce_sum(probs_tf * indicator, axis=1)
        Y_nonterm = tf.reduce_sum(probs_tf * F_child_final, axis=1)
        Y = tf.where(term_mask, Y_term, Y_nonterm)

        # Hint
        F_hint = hint_cdf(tf.cast(y_b, F_parent.dtype),
                          tf.cast(mu_b, F_parent.dtype),
                          tf.cast(yhalf_b, F_parent.dtype))

        loss_bell = tf.reduce_mean(tf.square(F_parent - tf.stop_gradient(Y)))
        loss_hint = tf.reduce_mean(tf.square(F_parent - tf.stop_gradient(F_hint)))
        loss = (1.0 - lam)*loss_bell + lam*loss_hint
        return loss

    epoch=0
    best = 1e9
    while epoch < cfg.max_epochs:
        if hot.q_stop: print("[Stop] User requested stop (q)."); break
        if time.time()-t0 > cfg.max_time_sec: print(f"[Stop] Max time {cfg.max_time_sec}s reached."); break

        lam = current_lambda(epoch, cfg)
        perm = tf.random.shuffle(tf.range(N))
        losses=[]
        for start in range(0, N, bsz):
            idx = perm[start: min(start+bsz, N)]
            with tf.GradientTape() as tape:
                loss = forward_batch(idx, tf.cast(lam, tf.keras.backend.floatx()))
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            losses.append(float(loss.numpy()))

        rmse = math.sqrt(sum(losses)/len(losses))
        best = min(best, rmse)
        elapsed = time.time()-t0
        print(f"Epoch {epoch:04d}  rmse={rmse:.6e}  lambda={lam:.3f}  elapsed={elapsed:.1f}s")
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{elapsed:.3f}", f"{rmse:.8e}", f"{lam:.6f}"])
        if epoch % 5 == 0: model.save(out_dir/"model.keras")
        if rmse < cfg.loss_tol_sqrt:
            print(f"[Stop] Converged: rmse {rmse:.3e} < tol {cfg.loss_tol_sqrt}")
            break

        epoch += 1

    model.save(out_dir/"model.keras")
    if cfg.eval_pairs:
        try:
            make_and_save_chart(model, u, cfg, out_dir / cfg.chart_pdf)
            print("[Chart] Saved final chart to:", out_dir / cfg.chart_pdf)
        except Exception as e:
            print("[Chart] Failed:", e)

    return model

# -------------------------
# MC chart (unchanged logic)
# -------------------------
def mc_cdf(u: UniverseBS, t_idx: int, x0: float, y_grid: np.ndarray, paths: int):
    h = u.h; steps = u.P - t_idx
    if steps <= 0: return (y_grid>0).astype(float)
    rng = np.random.default_rng(12345+t_idx)
    S0 = np.exp(x0)
    ys = np.zeros(paths)
    for p in range(paths):
        S = S0; ycum=0.0
        for k in range(steps):
            tau = u.T - (t_idx + k) * h
            q = - float(bs_delta(S, u.K, u.sigma, tau).numpy())
            z = rng.standard_normal()
            S_next = S * math.exp((-0.5*u.sigma**2)*h + u.sigma*math.sqrt(h)*z)
            payoff = max(S_next-u.K,0.0) if (t_idx+k+1)==u.P else 0.0
            ycum += -q*(S_next-S) - payoff
            S = S_next
        ys[p]=ycum
    ys.sort()
    return np.searchsorted(ys, y_grid, side="right")/float(paths)

def make_and_save_chart(model, u: UniverseBS, cfg: TrainConfig, out_pdf: Path):
    plt.figure(figsize=(7,5))
    for (t_idx, x_val) in (cfg.eval_pairs or []):
        tau = u.T - t_idx*u.h
        mu = - bs_call_price(np.exp(x_val), u.K, u.sigma, tau).numpy()
        y_half = 0.02
        y_grid = np.linspace(mu-2*y_half, mu+2*y_half, 201)
        sqrt_tau = np.sqrt(max(tau, 0.0))
        feats = np.stack([np.full_like(y_grid, t_idx),
                          np.full_like(y_grid, x_val),
                          y_grid,
                          np.full_like(y_grid, sqrt_tau)], axis=1)
        feats = feats.astype(np.float16 if is_mixed() else (np.float64 if tf.keras.backend.floatx()=="float64" else np.float32))
        preds = model(tf.convert_to_tensor(feats, dtype=model.input.dtype), training=False).numpy().squeeze(-1)
        mc = mc_cdf(u, t_idx, x_val, y_grid, cfg.mc_paths)
        plt.plot(y_grid, preds, label=f"Model t={t_idx} x={x_val:.2f}")
        plt.plot(y_grid, mc, "--", label=f"MC t={t_idx} x={x_val:.2f}")
    plt.xlabel("y"); plt.ylabel("CDF F(t,x,y)"); plt.title("F critic vs Monte Carlo")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(out_pdf); plt.close()

# -------------------------
# CLI
# -------------------------
def main():
    p = argparse.ArgumentParser(description="F-critic v2 with TF-native data, sqrt_tau feature, and child overrides.")
    p.add_argument("--precision", default="float32", choices=["float32","float64","mixed_float16"])
    p.add_argument("--optimizer", default="adam", choices=["adam","sgd","rmsprop"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--max_time_sec", type=int, default=600)
    p.add_argument("--loss_tol", type=float, default=1e-4)
    p.add_argument("--save_dir", default="saved_F_model_v2")
    p.add_argument("--reload", action="store_true")
    p.add_argument("--mc_paths", type=int, default=20000)
    p.add_argument("--eval_pairs", type=str, default="0:0.0,20:0.0,40:0.0")

    # Universe
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--T", type=float, default=0.5)
    p.add_argument("--P", type=int, default=60)
    p.add_argument("--K", type=float, default=1.0)

    # Sampler meta-params (D)
    p.add_argument("--N", type=int, default=60000)
    p.add_argument("--x0", type=float, default=0.0)
    p.add_argument("--a", type=float, default=0.6)
    p.add_argument("--b", type=float, default=2.0)
    p.add_argument("--c", type=float, default=0.0)
    p.add_argument("--r0", type=float, default=0.02)
    p.add_argument("--r1", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    u = UniverseBS(args.sigma, args.T, args.P, args.K)
    samp = SamplerConfig(N=args.N, x0=args.x0, a=args.a, b=args.b, c=args.c, r0=args.r0, r1=args.r1, seed=args.seed)
    eval_pairs=[]
    if args.eval_pairs:
        for tok in args.eval_pairs.split(","):
            t_s, x_s = tok.split(":"); eval_pairs.append((int(t_s), float(x_s)))
    cfg = TrainConfig(precision=args.precision, optimizer=args.optimizer, lr=args.lr,
                      batch_size=args.batch_size, max_epochs=args.max_epochs, max_time_sec=args.max_time_sec,
                      loss_tol_sqrt=args.loss_tol, model_dir=args.save_dir, log_csv="training_log.csv",
                      chart_pdf="comparison.pdf", eval_pairs=eval_pairs, mc_paths=args.mc_paths)
    train_F(u, samp, cfg, reload_model=args.reload)

if __name__ == "__main__":
    main()
