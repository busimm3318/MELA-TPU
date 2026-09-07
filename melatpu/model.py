"""JAX LM around the MELA-TPU core: pre-norm block (MELA + SwiGLU), tied embedding head,
optax AdamW training step. Parameters are nested dicts (pytrees)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from . import core

F32 = jnp.float32


def _lin(key, o, i):
    return jax.random.uniform(key, (o, i), F32, -1 / i ** 0.5, 1 / i ** 0.5)


def init_lm(key, vocab, cfg, layers):
    d = cfg["d"]
    h = int(round(8 * d / 3 / 64)) * 64
    keys = jax.random.split(key, layers + 1)
    blocks = []
    for i in range(layers):
        k = jax.random.split(keys[i], 4)
        blocks.append(dict(n1_g=jnp.ones((d,), F32), n1_b=jnp.zeros((d,), F32),
                           mix=core.init_params(k[0], cfg),
                           n2_g=jnp.ones((d,), F32), n2_b=jnp.zeros((d,), F32),
                           mlp_gate=_lin(k[1], h, d), mlp_up=_lin(k[2], h, d), mlp_down=_lin(k[3], d, h)))
    return dict(emb=jax.random.normal(keys[-1], (vocab, d), F32) * d ** -0.5,
                blocks=blocks, nf_g=jnp.ones((d,), F32), nf_b=jnp.zeros((d,), F32))


def layernorm(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps) * g + b


def lm_forward(P, cfg, tokens, us):
    """tokens [B,T] int; us: list over blocks of lists over events of [B,W,L+1,2] uniforms."""
    h = P["emb"][tokens]
    insts = []
    for bi, blk in enumerate(P["blocks"]):
        out, inst = core.layer_forward(blk["mix"], cfg, layernorm(h, blk["n1_g"], blk["n1_b"]), us[bi])
        h = h + out
        y = layernorm(h, blk["n2_g"], blk["n2_b"])
        h = h + (jax.nn.silu(y @ blk["mlp_gate"].T) * (y @ blk["mlp_up"].T)) @ blk["mlp_down"].T
        insts.append(inst)
    logits = layernorm(h, P["nf_g"], P["nf_b"]) @ P["emb"].T
    return logits, insts


def loss_fn(P, cfg, x, y, us):
    logits, insts = lm_forward(P, cfg, x, us)
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, y[..., None], axis=-1)[..., 0]
    return nll.mean(), insts


def make_train_step(cfg, lr=1e-3, clip=1.0):
    opt = optax.chain(optax.clip_by_global_norm(clip), optax.adamw(lr))

    @jax.jit
    def step(P, opt_state, x, y, us):
        (loss, insts), grads = jax.value_and_grad(loss_fn, has_aux=True)(P, cfg, x, y, us)
        upd, opt_state = opt.update(grads, opt_state, P)
        P = optax.apply_updates(P, upd)
        return P, opt_state, loss, insts

    return opt, step


def walk_uniforms(key, n_events, B, W, L):
    return jax.random.uniform(key, (n_events, B, W, L + 1, 2), F32)
