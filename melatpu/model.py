"""JAX LM around the MELA-TPU core: pre-norm block (MELA + SwiGLU), tied embedding head,
optax AdamW training step. Parameters are nested dicts (pytrees).

Training-path contract (MELA-TPU j0.5):
  * walk randomness is generated INSIDE the jitted step from one typed key
    (fold_in per step / block / event), so no host-side RNG, no per-step H2D copies,
    and every host of a multi-host run draws the same stream; the explicit `us`
    argument remains for equivalence tests;
  * every block is rematerialised (jax.checkpoint) so activation memory does not grow
    with depth; the event walk inside the block has its own remat (core.layer_forward);
  * make_train_step takes an optional device mesh for data-parallel sharding over the
    batch axis (jit SPMD; the per-step transport scale, a mean over (B,W), is reduced
    across devices by XLA) and donates the parameter/optimizer buffers.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from jax.sharding import NamedSharding, PartitionSpec as PS

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


def n_events(cfg):
    return len(range(cfg["k_event"], cfg["T"], cfg["k_event"]))


def walk_uniform_tree(key, layers, n_ev, B, W, L):
    """Walk uniforms for one step from one typed key: list over blocks of lists over
    events of [B,W,L+1,2] (the layout core.layer_forward consumes). Deterministic in
    (key, block, event); call inside jit."""
    return [[jax.random.uniform(jax.random.fold_in(jax.random.fold_in(key, bi), ei), (B, W, L + 1, 2), F32)
             for ei in range(n_ev)] for bi in range(layers)]


def _block(cfg, blk, h, u_list):
    out, inst = core.layer_forward(blk["mix"], cfg, layernorm(h, blk["n1_g"], blk["n1_b"]), u_list)
    h = h + out
    y = layernorm(h, blk["n2_g"], blk["n2_b"])
    h = h + (jax.nn.silu(y @ blk["mlp_gate"].T) * (y @ blk["mlp_up"].T)) @ blk["mlp_down"].T
    return h, inst


def lm_forward(P, cfg, tokens, us=None, key=None):
    """tokens [B,T] int. Either `us` (list over blocks of lists over events of [B,W,L+1,2]
    uniforms, for equivalence tests) or `key` (typed PRNG key; uniforms drawn in-graph)."""
    B, T = tokens.shape
    if us is None:
        us = walk_uniform_tree(key, len(P["blocks"]), n_events(cfg), B, cfg["n_walks"], cfg["walk_len"])
    h = P["emb"][tokens]
    insts = []
    block = jax.checkpoint(lambda blk, h, u: _block(cfg, blk, h, u)) if cfg.get("remat_blocks", True) \
        else (lambda blk, h, u: _block(cfg, blk, h, u))
    for bi, blk in enumerate(P["blocks"]):
        h, inst = block(blk, h, us[bi])
        insts.append(inst)
    logits = layernorm(h, P["nf_g"], P["nf_b"]) @ P["emb"].T
    return logits, insts


def loss_fn(P, cfg, x, y, us=None, key=None):
    logits, insts = lm_forward(P, cfg, x, us=us, key=key)
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, y[..., None], axis=-1)[..., 0]
    return nll.mean(), insts


def data_mesh():
    """One-axis mesh over every visible device (single host; multi-host after jax.distributed.initialize)."""
    try:
        return jax.make_mesh((jax.device_count(),), ("data",))
    except AttributeError:                                   # older jax
        import numpy as np
        return jax.sharding.Mesh(np.array(jax.devices()), ("data",))


def make_train_step(cfg, lr=1e-3, clip=1.0, mesh=None):
    """step(P, opt_state, x, y, key) -> (P, opt_state, loss, insts). `key` is a typed PRNG key
    for THIS step (caller: jax.random.fold_in(base_key, step_index)). With `mesh`, x/y are
    sharded over the batch axis and parameters/optimizer state are replicated; parameter and
    optimizer buffers are donated in both cases."""
    opt = optax.chain(optax.clip_by_global_norm(clip), optax.adamw(lr))

    def step(P, opt_state, x, y, key):
        (loss, insts), grads = jax.value_and_grad(loss_fn, has_aux=True)(P, cfg, x, y, key=key)
        upd, opt_state = opt.update(grads, opt_state, P)
        P = optax.apply_updates(P, upd)
        return P, opt_state, loss, insts

    if mesh is None:
        return opt, jax.jit(step, donate_argnums=(0, 1))
    rep, dat = NamedSharding(mesh, PS()), NamedSharding(mesh, PS("data"))
    return opt, jax.jit(step, in_shardings=(rep, rep, dat, dat, rep), out_shardings=(rep, rep, rep, rep),
                        donate_argnums=(0, 1))


def enable_compilation_cache(path):
    """Persistent XLA compilation cache (local dir or gs://bucket/dir): survives restarts and
    spot preemption; the key includes the device topology, so a v5e -> v6e move recompiles."""
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
