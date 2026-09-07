"""Gate J-1: the JAX core equals MELA-260906 (PyTorch, CPU) on identical weights and walk uniforms.
Run from the MELA-TPU root:  python tests/test_equiv_torch.py
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "MELA-260907"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402

from mela260907 import Config, MELALayer  # noqa: E402
from mela260907.walk import walk_uniforms  # noqa: E402
from melatpu import core  # noqa: E402


def torch_params(lay):
    sd = {k: v.detach().cpu().numpy() for k, v in lay.state_dict().items()}
    return dict(to_theta_w=sd["to_theta.weight"], to_theta_b=sd["to_theta.bias"],
                to_k_w=sd["to_k.weight"], to_q_w=sd["to_q.weight"], to_v_w=sd["to_v.weight"],
                to_gate_w=sd["to_gate.weight"], to_gate_b=sd["to_gate.bias"],
                to_out_member_w=sd["to_out_member.weight"], to_out_member_b=sd["to_out_member.bias"],
                to_in_member_w=sd["to_in_member.weight"], to_in_member_b=sd["to_in_member.bias"],
                from_read_w=sd["from_read.weight"], probe=sd["probe"], walk_q_w=sd["walk_q.weight"], walk_k_w=sd["walk_k.weight"],
                gain=sd["gain"], carry_bias=sd["carry_bias"])


def run(d=64, T=256, B=2, seed=0):
    cfg_t = Config(d=d, T=T, chunk=64)
    torch.manual_seed(seed)
    lay = MELALayer(cfg_t)
    with torch.no_grad():
        lay.to_theta.weight.normal_(0, 0.02); lay.gain.fill_(0.3); lay.carry_bias.normal_(0, 0.1)
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(B, T, d, generator=g)
    n_ev = len(range(cfg_t.k_event, T, cfg_t.k_event))
    u = walk_uniforms(n_ev, B, cfg_t.n_walks, cfg_t.walk_len, seed=7, device="cpu")
    lay.eval()
    with torch.no_grad():
        o_t, aux = lay(h, want_aux=True, u=u)
    P = {k: jnp.asarray(v) for k, v in torch_params(lay).items()}
    cfg = core.config(d, T, chunk=64, recompute=False)
    fwd = jax.jit(lambda P, h, us: core.layer_forward(P, cfg, h, us))
    us = [jnp.asarray(u[i].numpy()) for i in range(n_ev)]
    t0 = time.perf_counter()
    o_j, inst = fwd(P, jnp.asarray(h.numpy()), us)
    o_j = np.asarray(o_j); t1 = time.perf_counter() - t0
    rel = float(np.abs(o_j - o_t.numpy()).max() / np.abs(o_t.numpy()).max())
    ti = aux["instruments"][0]; ji = jax.tree_util.tree_map(float, inst[0])
    print("J-1 d=%d T=%d output rel %.2e | K torch %.1f jax %.1f | hol_norm %.4f / %.4f | closed_frac %.3f / %.3f | jit+run %.1f s" % (
        d, T, rel, float(ti["K"]), ji["K"], float(ti["hol_norm"]), ji["hol_norm"], float(ti["closed_frac"]), ji["closed_frac"], t1))
    ok = rel < 1e-4 and abs(float(ti["K"]) - ji["K"]) < 1e-6
    # gradient check on to_out_member (vjp vs torch autograd)
    lay.train(); lay.zero_grad()
    o1 = lay(h, u=u); w = torch.randn_like(o1); (o1 * w).sum().backward()
    g_t = lay.to_out_member.weight.grad.numpy()
    def loss(P):
        o, _ = core.layer_forward(P, cfg, jnp.asarray(h.numpy()), us)
        return (o * jnp.asarray(w.numpy())).sum()
    g_j = np.asarray(jax.grad(loss)(P)["to_out_member_w"])
    grel = float(np.linalg.norm(g_t - g_j) / np.linalg.norm(g_t))
    print("J-1 grad err-ratio to_out_member %.2e" % grel)
    ok = ok and grel < 1e-3
    print("GATE_J1", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
