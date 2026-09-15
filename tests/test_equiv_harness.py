"""Gate J-H: the JAX LOOPWORD harness equals the PyTorch one.

J-D covers the model. This covers everything wrapped around it -- the task's
token stream, the classifier head that reads at each example's own query
position, the classification loss, the group-stratified baseline of the score
function, and the gradient that comes back through all of it.

Both sides are fed the SAME batch and the SAME walk uniforms, and the JAX
parameters are copied from the PyTorch ones, so nothing here depends on two
random number generators agreeing. A porting defect shows up as a disagreement
in the loss or in the gradient.

Run from the MELA-TPU root:  python tests/test_equiv_harness.py
Needs the reference repository as a sibling directory (MELA-260915 / MELA), or
MELA_REF_DIR.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_REF = os.environ.get("MELA_REF_DIR") or next(
    (p for p in (os.path.join(os.path.dirname(ROOT), d)
                 for d in ("MELA-260915", "MELA-260913", "MELA")) if os.path.isdir(p)), None)
if _REF is None:
    raise SystemExit("J-H needs the PyTorch reference as a sibling directory, or MELA_REF_DIR")
sys.path.insert(0, _REF)
os.environ.setdefault("LOOPWORD_PATH", "0")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402

from mela260915 import Block, Config  # noqa: E402
from mela260915.walk import walk_uniforms  # noqa: E402
from melatpu import core, tasks  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, 'tests'))
from test_equiv_dfixes import NAMES  # noqa: E402

sys.path.insert(0, _REF)
import loopword913 as ref  # noqa: E402


def mix_params(mix):
    sd = {k: v.detach().cpu().numpy() for k, v in mix.state_dict().items()}
    P = {j: sd[t] for j, t in NAMES.items() if t in sd}
    if getattr(mix, "conv_w", None) is not None:
        P["conv_w"] = mix.conv_w.detach().cpu().numpy()[:, 0, :]
    if getattr(mix, "log_theta0", None) is not None:
        P["log_theta0"] = mix.log_theta0.detach().cpu().numpy()
    return {k: jnp.asarray(v) for k, v in P.items()}


def torch_to_jax(m):
    blocks = []
    for b in m.blocks:
        blocks.append(dict(
            n1_g=jnp.asarray(b.n1.weight.detach().numpy()), n1_b=jnp.asarray(b.n1.bias.detach().numpy()),
            mix=mix_params(b.mix),
            n2_g=jnp.asarray(b.n2.weight.detach().numpy()), n2_b=jnp.asarray(b.n2.bias.detach().numpy()),
            mlp_gate=jnp.asarray(b.mlp.gate.weight.detach().numpy()),
            mlp_up=jnp.asarray(b.mlp.up.weight.detach().numpy()),
            mlp_down=jnp.asarray(b.mlp.down.weight.detach().numpy())))
    return dict(emb=jnp.asarray(m.emb.weight.detach().numpy()), blocks=blocks,
                nf_g=jnp.asarray(m.nf.weight.detach().numpy()),
                nf_b=jnp.asarray(m.nf.bias.detach().numpy()),
                head_w=jnp.asarray(m.head.weight.detach().numpy()),
                head_b=jnp.asarray(m.head.bias.detach().numpy()))


def sampler_exact(B=4, M=32, W=64, L=12, seed=0):
    """The two samplers must make the SAME picks when handed the same routing
    matrix and the same uniforms. This is the check that makes the walk-term
    tolerance in run() legitimate: with this exact, any disagreement in the walk
    term can only come from the last-bit difference in the routing matrix, not
    from a porting defect in the sampler."""
    from mela260915 import walk as tw
    rng = np.random.default_rng(seed)
    A = torch.softmax(torch.as_tensor(rng.normal(0, 2.0, (B, M, M)).astype(np.float32)), -1).numpy()
    in_mean = A.mean(1)
    u = rng.random((B, W, L + 1, 2)).astype(np.float32)
    ok = True
    for name, kw in (("escape", dict(dead_end="escape", self_pair_mask=False)),
                     ("die", dict(dead_end="die", self_pair_mask=True, death_c=0.5))):
        t = tw.sample_slots(torch.as_tensor(A), torch.as_tensor(in_mean), torch.as_tensor(u), L, **kw)
        j = core.sample_slots(jnp.asarray(A), jnp.asarray(in_mean), jnp.asarray(u), L, **kw)
        dp = int((t["pairs"].numpy() != np.asarray(j["pairs"])).sum())
        dd = int((t["dead"].numpy() != np.asarray(j["dead"])).sum())
        dc = int((t["closed"].numpy() != np.asarray(j["closed"])).sum())
        dl = float(np.abs(t["logprob"].numpy() - np.asarray(j["logprob"])).max())
        print("J-H sampler %-6s: picks differing %d/%d | dead %d | closed %d | logprob maxabs %.2e"
              % (name, dp, t["pairs"].numel(), dd, dc, dl))
        ok = ok and dp == 0 and dd == 0 and dc == 0 and dl < 1e-4
    return ok


def run(arm="main", d=64, B=8, layers=2, seed=0, mu=0.1):
    # one task instance drives both sides; the batch is converted, not regenerated
    jt = tasks.LoopWord(K=8, k=5, seed=seed, even_only=True)
    T = 3 * (max(jt.L_choices) + jt.n_distract[1]) + 2
    k_event = 3 * (min(jt.L_choices) + jt.n_distract[0]) + 1
    x, y, last, Ls = jt.batch(B, T)

    cfg_t = ref.make_config(arm, d, T, k_event, mu)
    torch.manual_seed(seed)
    mt = ref.Classifier(jt.vocab, cfg_t, layers, jt.G)
    with torch.no_grad():                                  # move the knobs off their init
        for b in mt.blocks:
            b.mix.gain.fill_(0.3)
            b.mix.carry_bias.normal_(0, 0.1)
            if b.mix.conv_w is not None:
                b.mix.conv_w.normal_(0, 0.3)
            if b.mix.to_fwd is not None:
                b.mix.to_fwd.weight.normal_(0, 0.5)
                b.mix.to_rev.weight.normal_(0, 0.5)
                b.mix.to_fwd.bias.fill_(0.4)
                b.mix.to_rev.bias.fill_(-0.2)
            if b.mix.to_decay is not None:
                b.mix.to_decay.weight.normal_(0, 0.02)
            if b.mix.log_theta0 is not None:
                b.mix.log_theta0.fill_(1.1)
            if b.mix.carry_a is not None:
                b.mix.carry_a.fill_(0.3)
    n_ev = len(range(cfg_t.k_event, T, cfg_t.k_event)) or 1
    us_t = [walk_uniforms(n_ev, B, cfg_t.n_walks, cfg_t.walk_len, seed=7 + i, device="cpu")
            for i in range(layers)]

    # the reference Classifier does not forward `u` to the layer, so it would draw
    # fresh walk randomness on every call; inject the same uniforms both sides use
    for i, b in enumerate(mt.blocks):
        b.mix.walk_u = us_t[i]
    xt, yt, lt, Lt = (torch.as_tensor(v.astype(np.int64)) for v in (x, y, last, Ls))
    mt.eval()
    logits_t, aux_t = mt(xt, lt, want_aux=True)
    loss_t, info_t = ref.cls_loss(logits_t, yt, aux_t, cfg_t.mu_walk, stratum=Lt)

    # mirror the reference arm exactly -- `legacy` forces mu_walk 0, so taking mu
    # from the caller instead of from cfg_t compares a loss carrying the walk term
    # against one without it, which is what made this gate fail on 2026-09-15
    build = core.config_main if arm in ("main", "oracle-dir") else core.config
    cfg_j = build(d, T, chunk=k_event, mu_walk=cfg_t.mu_walk)
    cfg_j = dict(cfg_j, k_event=k_event)
    Pj = torch_to_jax(mt)
    us_j = [[jnp.asarray(us_t[i][e].numpy()) for e in range(n_ev)] for i in range(layers)]
    loss_j, (task_j, _, logits_j) = tasks.cls_loss(
        Pj, cfg_j, jnp.asarray(x), jnp.asarray(y), jnp.asarray(last),
        us=us_j, stratum=jnp.asarray(Ls))

    lg = np.abs(np.asarray(logits_j) - logits_t.detach().numpy()).max() / \
        max(np.abs(logits_t.detach().numpy()).max(), 1e-12)
    dl = abs(float(loss_j) - float(loss_t.detach())) / max(abs(float(loss_t.detach())), 1e-12)
    dt = abs(float(task_j) - float(info_t["task"].detach())) / max(abs(float(info_t["task"].detach())), 1e-12)
    print("J-H %-6s d=%d T=%d B=%d: logits rel %.2e | task rel %.2e | total rel %.2e"
          % (arm, d, T, B, lg, dt, dl))
    # the total carries the walk term, which is judged below against its own
    # noise; hold the total to 1e-4 only when there is no walk term in it
    ok = lg < 1e-4 and dt < 1e-4 and (dl < 1e-4 or bool(cfg_t.mu_walk))
    if "walk" in info_t:
        # The walk term is a score-function estimator over SAMPLED slot sequences.
        # The pick is an inverse-CDF lookup, so it is a step function of the routing
        # matrix: the two implementations sum in different orders, their routing
        # matrices differ in the last bits (~6e-07 relative), and about one uniform
        # in 3000 lands inside that gap and picks a different slot. One flipped walk
        # moves that walk's log-probability by a full ~34 nats. Demanding 1e-4 here
        # would be demanding that two float32 summation orders agree bit for bit,
        # which no correct port can deliver. sampler_exact() above pins the sampler
        # itself; what is left is judged against the estimator's OWN noise -- the
        # spread of the term across walk seeds, which is the scale at which any
        # difference in it could matter to training.
        wt = float(info_t["walk"])
        wj = (float(loss_j) - float(task_j)) / cfg_t.mu_walk
        alt = []
        for s_ in range(4):
            u_alt = [walk_uniforms(n_ev, B, cfg_t.n_walks, cfg_t.walk_len, seed=101 + 7 * s_ + i,
                                   device="cpu") for i in range(layers)]
            for i, b in enumerate(mt.blocks):
                b.mix.walk_u = u_alt[i]
            with torch.no_grad():
                lo_a, aux_a = mt(xt, lt, want_aux=True)
                _, inf_a = ref.cls_loss(lo_a, yt, aux_a, cfg_t.mu_walk, stratum=Lt)
            alt.append(float(inf_a["walk"]))
        for i, b in enumerate(mt.blocks):                  # restore the gate's uniforms
            b.mix.walk_u = us_t[i]
        sd = float(np.std(alt + [wt]))
        print("   walk term  torch %.6f  jax %.6f  |diff| %.2e  seed spread (sd) %.2e  ratio %.2f"
              % (wt, wj, abs(wj - wt), sd, abs(wj - wt) / max(sd, 1e-12)))
        ok = ok and abs(wj - wt) < 0.5 * sd

    # gradient through the whole harness
    mt.train()
    mt.zero_grad()
    logits_t, aux_t = mt(xt, lt, want_aux=True)
    loss_t, _ = ref.cls_loss(logits_t, yt, aux_t, cfg_t.mu_walk, stratum=Lt)
    loss_t.backward()
    gj = jax.grad(lambda P: tasks.cls_loss(P, cfg_j, jnp.asarray(x), jnp.asarray(y),
                                           jnp.asarray(last), us=us_j,
                                           stratum=jnp.asarray(Ls))[0])(Pj)
    pairs = [("head_w", mt.head.weight, gj["head_w"]),
             ("emb", mt.emb.weight, gj["emb"]),
             ("b0.to_in_member", mt.blocks[0].mix.to_in_member.weight,
              gj["blocks"][0]["mix"]["to_in_member_w"]),
             ("b0.mlp_down", mt.blocks[0].mlp.down.weight, gj["blocks"][0]["mlp_down"])]
    if mt.blocks[0].mix.to_fwd is not None:
        pairs.append(("b0.to_fwd", mt.blocks[0].mix.to_fwd.weight, gj["blocks"][0]["mix"]["to_fwd_w"]))
    for name, tp, gjj in pairs:
        gt = tp.grad.detach().numpy()
        gv = np.asarray(gjj).reshape(gt.shape)
        den = max(float(np.linalg.norm(gt)), 1e-12)
        rel = float(np.linalg.norm(gt - gv) / den)
        print("   grad %-18s err-ratio %.2e  (|grad| %.3e)" % (name, rel, den))
        ok = ok and rel < 1e-3
    print("GATE_JH", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    good = sampler_exact()
    for arm in ("main", "legacy"):
        good &= run(arm)
    raise SystemExit(0 if good else 1)
