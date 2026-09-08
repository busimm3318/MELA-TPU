"""Gate J-2: the training path runs end to end under jit on the available devices --
in-graph walk randomness from a typed key, per-block rematerialisation, data-parallel
sharding over the device mesh, donated buffers, persistent compilation cache -- and the
loss on a fixed batch goes down while the holonomy stays a rotation (orth_drift < 1e-3).
Run from the MELA-TPU root:  python tests/test_train_smoke.py
"""
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from melatpu import core, model  # noqa: E402


def main(d=64, T=256, B=None, vocab=65, layers=2, steps=6):
    B = B or 2 * jax.device_count()          # the batch axis is sharded: B must be a multiple of the device count
    model.enable_compilation_cache(os.path.join(tempfile.gettempdir(), "mela_tpu_jax_cache"))
    cfg = core.config(d=d, T=T)
    key = jax.random.key(0)
    P = model.init_lm(key, vocab, cfg, layers)
    mesh = model.data_mesh()
    opt, step = model.make_train_step(cfg, lr=3e-3, mesh=mesh)
    opt_state = opt.init(P)
    x = jax.random.randint(jax.random.key(1), (B, T), 0, vocab)
    y = jnp.roll(x, -1, axis=1)
    losses, drift = [], 0.0
    t0 = time.perf_counter()
    for i in range(steps):
        P, opt_state, loss, insts = step(P, opt_state, x, y, jax.random.fold_in(key, i))
        losses.append(float(loss))
        drift = max(drift, max(float(e["orth_drift"]) for blk in insts for e in blk))
        if i == 0:
            print(f"J-2 compile+first step {time.perf_counter() - t0:.1f} s on {jax.device_count()} device(s) ({jax.devices()[0].platform})")
    print("J-2 losses", [round(l, 4) for l in losses], "| max orth_drift %.2e" % drift)
    ok = all(jnp.isfinite(jnp.array(losses))) and losses[-1] < losses[0] and drift < 1e-3
    print("GATE_J2", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
