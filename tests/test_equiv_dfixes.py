"""Gate J-D: the JAX port of the 2026-09-14 walk structure fixes equals the PyTorch
reference (package mela260915) on identical weights and identical walk uniforms.

J-1 checks the port of the frozen design. J-D checks the port of everything added
on top of it -- the rebased write-back, the direction gates with transposed reverse
transport, dead-end death with the loop-only chain, the short convolution, the
withdrawn angle normalisation with a learned scale, the write scale, and the
log-spaced decay initialisation -- against an implementation that is already gated
on the PyTorch side. A port cannot be verified against its own intentions; it can
be verified against the reference, which is what this repository has always done
(the three defects found by the 2026-09-08 audit were all "differs from the
reference").

Run from the MELA-TPU root:  python tests/test_equiv_dfixes.py
Needs the reference repository as a sibling directory (MELA / MELA-260915), or
MELA_REF_DIR.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_REF = os.environ.get("MELA_REF_DIR") or next(
    (p for p in (os.path.join(os.path.dirname(ROOT), d)
                 for d in ("MELA-260915", "MELA-260913", "MELA")) if os.path.isdir(p)), None)
if _REF is None:
    raise SystemExit("J-D needs the PyTorch reference as a sibling directory, or MELA_REF_DIR")
sys.path.insert(0, _REF)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402

from mela260915 import Config  # noqa: E402
from mela260915.layer import MELALayer  # noqa: E402
from mela260915.walk import walk_uniforms  # noqa: E402
from melatpu import core  # noqa: E402

NAMES = dict(to_theta_w="to_theta.weight", to_theta_b="to_theta.bias", to_k_w="to_k.weight",
             to_q_w="to_q.weight", to_v_w="to_v.weight", to_gate_w="to_gate.weight",
             to_gate_b="to_gate.bias", to_out_member_w="to_out_member.weight",
             to_out_member_b="to_out_member.bias", to_in_member_w="to_in_member.weight",
             to_in_member_b="to_in_member.bias", from_read_w="from_read.weight", probe="probe",
             walk_q_w="walk_q.weight", walk_k_w="walk_k.weight", gain="gain", carry_bias="carry_bias",
             to_decay_w="to_decay.weight", to_decay_b="to_decay.bias",
             to_fwd_w="to_fwd.weight", to_fwd_b="to_fwd.bias",
             to_rev_w="to_rev.weight", to_rev_b="to_rev.bias", carry_a="carry_a")


def torch_params(lay):
    sd = {k: v.detach().cpu().numpy() for k, v in lay.state_dict().items()}
    P = {j: sd[t] for j, t in NAMES.items() if t in sd}
    if lay.conv_w is not None:                       # [d,1,w] -> [d,w]
        P["conv_w"] = lay.conv_w.detach().cpu().numpy()[:, 0, :]
    if lay.log_theta0 is not None:
        P["log_theta0"] = lay.log_theta0.detach().cpu().numpy()
    return P


def run(d=64, T=256, B=2, seed=0):
    cfg_t = Config.main(d=d, T=T, chunk=64)
    torch.manual_seed(seed)
    lay = MELALayer(cfg_t)
    with torch.no_grad():                            # move every knob off its init
        lay.to_theta.weight.normal_(0, 0.02)
        lay.gain.fill_(0.3)
        lay.carry_bias.normal_(0, 0.1)
        lay.conv_w.normal_(0, 0.3)
        lay.to_fwd.weight.normal_(0, 0.5)
        lay.to_rev.weight.normal_(0, 0.5)
        lay.to_fwd.bias.fill_(0.4)
        lay.to_rev.bias.fill_(-0.2)
        lay.to_decay.weight.normal_(0, 0.02)
        lay.log_theta0.fill_(1.1)
        lay.carry_a.fill_(0.3)
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(B, T, d, generator=g)
    n_ev = len(range(cfg_t.k_event, T, cfg_t.k_event))
    u = walk_uniforms(n_ev, B, cfg_t.n_walks, cfg_t.walk_len, seed=7, device="cpu")
    lay.eval()
    with torch.no_grad():
        o_t, aux = lay(h, want_aux=True, u=u)
    P = {k: jnp.asarray(v) for k, v in torch_params(lay).items()}
    cfg = core.config_main(d, T, chunk=64, recompute=False)
    fwd = jax.jit(lambda P, h, us: core.layer_forward(P, cfg, h, us))
    us = [jnp.asarray(u[i].numpy()) for i in range(n_ev)]
    t0 = time.perf_counter()
    o_j, inst, lp = fwd(P, jnp.asarray(h.numpy()), us)
    o_j = np.asarray(o_j)
    t1 = time.perf_counter() - t0
    rel = float(np.abs(o_j - o_t.numpy()).max() / np.abs(o_t.numpy()).max())
    ti = aux["instruments"][0]
    ji = jax.tree_util.tree_map(float, inst[0])
    lp_rel = float(np.abs(np.asarray(lp[0]) - aux["logprob"][0].numpy()).max()
                   / max(np.abs(aux["logprob"][0].numpy()).max(), 1e-12))
    print("J-D d=%d T=%d main config: output rel %.2e | log-prob rel %.2e | jit+run %.1f s"
          % (d, T, rel, lp_rel, t1))
    keys = ("K", "closed_frac", "hol_norm", "len_mean", "dead_frac", "no_loop_frac",
            "top_slot_share", "update_norm", "carry_alpha", "theta0", "mean_angle", "fwd_ratio")
    worst = 0.0
    for k in keys:
        a, b = float(ti[k]), ji[k]
        e = abs(a - b) / max(abs(a), 1e-9)
        worst = max(worst, e)
        print("   %-15s torch %10.5f  jax %10.5f  rel %.1e" % (k, a, b, e))
    ok = rel < 1e-4 and lp_rel < 1e-4 and worst < 1e-3

    # gradient: the routing projection, the direction gate and the write scale
    lay.train()
    lay.zero_grad()
    o1 = lay(h, u=u)
    w = torch.randn_like(o1)
    (o1 * w).sum().backward()

    def loss(P):
        o, _, _ = core.layer_forward(P, cfg, jnp.asarray(h.numpy()), us)
        return (o * jnp.asarray(w.numpy())).sum()

    gj = jax.grad(loss)(P)
    for jname, tparam in (("to_out_member_w", lay.to_out_member.weight),
                          ("to_fwd_w", lay.to_fwd.weight),
                          ("carry_a", lay.carry_a),
                          ("to_decay_w", lay.to_decay.weight)):
        gt = tparam.grad.detach().numpy()
        gjj = np.asarray(gj[jname]).reshape(gt.shape)
        den = max(float(np.linalg.norm(gt)), 1e-12)
        grel = float(np.linalg.norm(gt - gjj) / den)
        print("   grad %-16s err-ratio %.2e  (|grad| %.3e)" % (jname, grel, den))
        ok = ok and grel < 1e-3
    print("GATE_JD", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
