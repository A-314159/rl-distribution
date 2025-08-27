import tensorflow as tf
from .universe import UniverseBS
from .precision import policy_dtype
from .bs import bs_delta  # your dtype-safe helper

class Actor:
    def __call__(self, t_idx: tf.Tensor, x: tf.Tensor, u: UniverseBS) -> tf.Tensor:
        raise NotImplementedError

class ActorBSDelta(Actor):
    def __init__(self, K: float): self.K = K
    def __call__(self, t_idx, x, u: UniverseBS):
        tau = tf.cast(u.T, x.dtype) - tf.cast(t_idx, x.dtype) * tf.cast(u.h, x.dtype)
        S = tf.exp(x)
        return -bs_delta(S, tf.cast(self.K, policy_dtype()), tf.cast(u.sigma, policy_dtype()), tau)
