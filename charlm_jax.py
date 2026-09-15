"""Character language-model smoke in JAX (stage T2 of the TPU protocol).

The PyTorch original is charlm913.py in the reference repository; the model the
two harnesses drive is the same function to 1.6e-06 (gate J-D), so the numbers
sit in one coordinate system.

CAVEAT ON 2.194. The registered 2.194 (kernel freeze, 2026-09-06) was measured at
n = 8, M = 256. This sizes by the current rule, n = d/4 and M = d/2, so 2.194 is
an EXTERNAL ANCHOR and not the control. The control is the `legacy` arm here, and
the comparison that decides anything is `main` against `slot+conv+decay` -- the
same configuration with the events switched off, so a short convolution's gain is
never credited to the walk.

Arms:
  main              D1-D6 plus the log-spaced decay (the model-wide setting)
  slot+conv+decay   the same, events off: the control the walk is judged against
  no-decay          main with the decay off
  legacy-walk       the pre-D-fix walk
  legacy            the frozen design

Read at every evaluation, because a loss alone decides nothing here: validation
loss sliced by position and by recall distance (a size axis is never collapsed),
word rate and distinct-4 with a sample of the text (a generation regression is a
FAIL even when the loss passes), the walk instruments, the event's perturbation
of the read, and the walk term's share of the pre-clip gradient norm.

    python charlm_jax.py --arm main --steps 2000
    CHARLM_DATA=gs://bucket/tinyshakespeare.txt python charlm_jax.py --all
"""
from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from melatpu import core, model
from loopword_jax import ckpt_read, ckpt_write, _read, _write

DATA = os.environ.get("CHARLM_DATA", "tinyshakespeare.txt")
ARMS = ("main", "slot+conv+decay", "no-decay", "legacy-walk", "legacy")
STRIP = ".,;:!?'\"-"


def make_cfg(arm, d, T, mu, **kw):
    if arm == "main":
        return core.config_main(d, T, mu_walk=mu, **kw)
    if arm == "slot+conv+decay":                  # k_event past T: no event ever fires
        return core.config_main(d, T, mu_walk=mu, k_event=T + 1, **kw)
    if arm == "no-decay":
        return core.config_main(d, T, mu_walk=mu, decay=False, **kw)
    if arm == "legacy-walk":
        return core.config(d, T, mu_walk=mu, **kw)
    if arm == "legacy":
        return core.config(d, T, mu_walk=0.0, **kw)
    raise ValueError(arm)


def load_text():
    blob = _read(DATA)
    if blob is not None:
        return blob.decode("utf-8")
    # Local fallback so the harness runs with no corpus present. It is a smoke
    # fixture, not the experiment: a verdict is never read off this text.
    print("WARNING: %s not found, using the synthetic fallback corpus" % DATA, flush=True)
    words = ["the", "and", "to", "of", "a", "in", "that", "is", "it", "for"]
    r = np.random.default_rng(0)
    return " ".join(words[i] for i in r.integers(0, len(words), 200_000))


def sliced_eval(P, cfg, va, T, B, key, n_batches=5):
    """Validation loss by position bucket and by recall distance -- how far back
    the token being predicted last appeared. Standing rule: never collapse a size
    axis, because a mechanism that only helps at long recall is invisible in a
    mean."""
    r = np.random.default_rng(99)
    edges = [4, 16, 64, 256, 10 ** 9]
    pos_s, pos_c = np.zeros(8), np.zeros(8)
    d_s, d_c = np.zeros(6), np.zeros(6)
    tot = 0.0
    for b_i in range(n_batches):
        ix = r.integers(0, len(va) - T - 1, B)
        x = np.stack([va[i:i + T] for i in ix])
        y = np.stack([va[i + 1:i + T + 1] for i in ix])
        logits, _, _ = model.lm_forward(P, cfg, jnp.asarray(x),
                                        key=jax.random.fold_in(key, 5000 + b_i))
        lp = jax.nn.log_softmax(logits, axis=-1)
        lo = np.asarray(-jnp.take_along_axis(lp, jnp.asarray(y)[..., None], axis=-1)[..., 0])
        tot += float(lo.mean())
        pb = np.arange(T) * 8 // T
        for k in range(8):
            m = pb == k
            pos_s[k] += lo[:, m].sum()
            pos_c[k] += m.sum() * B
        for bi in range(B):
            last = {}
            for t in range(T):
                tok = int(y[bi, t])
                dd = (t + 1 - last[tok]) if tok in last else None
                k = 5 if dd is None else next(i for i, e in enumerate(edges) if dd <= e)
                d_s[k] += lo[bi, t]
                d_c[k] += 1
                last[int(x[bi, t])] = t
    return (tot / n_batches, (pos_s / np.maximum(pos_c, 1)).tolist(),
            (d_s / np.maximum(d_c, 1)).tolist())


def gen_metrics(P, cfg, va, itos, vocab_words, T, key, n_tok=256):
    """Generation beside the loss at every gate. The window is a fixed T tokens,
    so the sampling step compiles once instead of once per length."""
    buf = np.zeros((1, T), dtype=np.int32)
    buf[0, -64:] = np.asarray(va[:64], dtype=np.int32)
    out = []

    @jax.jit
    def nxt(P, w, k):
        logits, _, _ = model.lm_forward(P, cfg, w, key=k)
        return logits[:, -1]

    for i in range(n_tok):
        lg = nxt(P, jnp.asarray(buf), jax.random.fold_in(key, 9000 + i))
        tok = int(jax.random.categorical(jax.random.fold_in(key, 20000 + i), lg[0]))
        out.append(tok)
        buf = np.concatenate([buf[:, 1:], np.array([[tok]], dtype=np.int32)], axis=1)
    text = "".join(itos[t] for t in out)
    words = [w for w in text.split() if w]
    wr = sum(w.strip(STRIP).lower() in vocab_words for w in words) / max(len(words), 1)
    grams = [text[i:i + 4] for i in range(len(text) - 3)]
    return wr, len(set(grams)) / max(len(grams), 1), text[:110].replace("\n", " / ")


def walk_grad_share(P, cfg, x, y, key):
    """The pre-clip gradient norm with and without the walk term: how much of the
    step the score function is actually driving."""
    cfg0 = dict(cfg, mu_walk=0.0)
    g0 = jax.grad(lambda Q: model.loss_fn(Q, cfg0, x, y, key=key)[0])(P)
    g1 = jax.grad(lambda Q: model.loss_fn(Q, cfg, x, y, key=key)[0])(P)
    return float(optax.global_norm(g0)), float(optax.global_norm(g1))


def flat_inst(insts):
    """Mean each instrument over blocks and events; scalars only."""
    out, n = {}, 0
    for blk in insts:
        for e in blk:
            n += 1
            for k, v in e.items():
                out[k] = out.get(k, 0.0) + float(v)
    return {k: v / max(n, 1) for k, v in out.items()}


def run(arm, seed, steps, d, T, B, layers, lr, mu, out, ckpt_every, eval_every):
    text = load_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab_words = {w.strip(STRIP).lower() for w in text.split() if w}
    data = np.array([stoi[c] for c in text], dtype=np.int32)
    split = int(0.9 * len(data))
    tr, va = data[:split], data[split:]

    cfg = make_cfg(arm, d, T, mu)
    key = jax.random.key(seed)
    P = model.init_lm(key, len(chars), cfg, layers)
    tx, step_fn = model.make_train_step(cfg, lr=lr, clip=1.0)
    opt_state = tx.init(P)
    log = {"arm": arm, "seed": seed, "cfg": {k: str(v) for k, v in cfg.items()},
           "vocab": len(chars), "curve": [], "evals": []}
    start = 0
    ck = (out + ".ckpt") if out else None
    if ck:
        st = ckpt_read(ck)
        if st:
            P, opt_state, start, log = st["P"], st["opt_state"], st["step"], st["log"]
            print("RESUMED at step %d" % start, flush=True)
            if start >= steps:
                # Restarting after a preemption re-runs the whole command, so a finished
                # arm is reached again. Without this the empty training loop leaves x and y
                # unbound and the final evaluate() raises, killing every later arm.
                print("%s already complete at %d steps" % (arm, start), flush=True)
                return log["evals"][-1] if log["evals"] else {}

    rng = np.random.default_rng(seed)

    def batch(src):
        ix = rng.integers(0, len(src) - T - 1, B)
        return (jnp.asarray(np.stack([src[i:i + T] for i in ix])),
                jnp.asarray(np.stack([src[i + 1:i + T + 1] for i in ix])))

    if cfg.get("theta0_learn") and start == 0:
        x0, _ = batch(tr)
        P = model.calibrate_theta0(P, cfg, x0, key=jax.random.fold_in(key, 1))
        print("D5 theta0 ->", [round(float(jnp.exp(b["mix"]["log_theta0"])), 2)
                               for b in P["blocks"] if "log_theta0" in b["mix"]], flush=True)
        opt_state = tx.init(P)

    params = sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(P))
    print("ARM %s | d=%d n=%d M=%d k_event=%d T=%d B=%d layers=%d params %.2fM | decay=%s(%s) "
          "scale=%s gates=%s rebase=%s conv=%d alpha0=%.4f c=%.2f mu=%.3g"
          % (arm, d, cfg["n"], cfg["M"], cfg["k_event"], T, B, layers, params / 1e6,
             cfg["decay"], cfg["decay_init"], cfg["transport_scale"], cfg["direction_gates"],
             cfg["writeback_rebase"], cfg["short_conv"], core.carry_alpha0(cfg),
             cfg["death_c"], cfg["mu_walk"]), flush=True)

    t0 = time.perf_counter()
    for step in range(start + 1, steps + 1):
        x, y = batch(tr)
        P, opt_state, loss, insts = step_fn(P, opt_state, x, y, jax.random.fold_in(key, step))
        if step % 100 == 0 or step == 1:
            log["curve"].append(dict(step=step, loss=float(loss), **flat_inst(insts)))
        if step % 500 == 0 or step == 1:
            el = time.perf_counter() - t0
            print("%-16s s%d %5d/%d loss %.4f (%.0fs, %.3f s/step)"
                  % (arm, seed, step, steps, float(loss), el, el / max(step - start, 1)),
                  flush=True)
        if ckpt_every and ck and step % ckpt_every == 0:
            ckpt_write(ck, step, P, opt_state, log)
        if eval_every and step % eval_every == 0 and step < steps:
            log["evals"].append(evaluate(P, cfg, va, T, B, itos, vocab_words,
                                         x, y, key, step, arm, seed, t0))
    log["evals"].append(evaluate(P, cfg, va, T, B, itos, vocab_words,
                                 x, y, key, steps, arm, seed, t0, final=True))
    if out:
        _write(out + ".json", json.dumps(log, indent=1).encode("utf-8"))
        ckpt_write(out + ".ckpt", steps, P, opt_state, log)
    return log["evals"][-1]


def evaluate(P, cfg, va, T, B, itos, vocab_words, x, y, key, step, arm, seed, t0,
             final=False):
    vl, pos, dist = sliced_eval(P, cfg, va, T, B, key)
    wr, d4, sample = gen_metrics(P, cfg, va, itos, vocab_words, T, key)
    _, insts, _ = model.lm_forward(P, cfg, x, key=jax.random.fold_in(key, 77), probe=True)
    inst = flat_inst(insts)
    g_task, g_tot = (walk_grad_share(P, cfg, x, y, jax.random.fold_in(key, 78))
                     if cfg["mu_walk"] else (0.0, 0.0))
    # the share is the readable number: at initialisation the score function moves
    # the step by about 3e-05 of its norm, and whether that grows is the question
    gshare = abs(g_tot - g_task) / max(g_task, 1e-12)
    row = dict(step=step, val=vl, word_rate=wr, distinct4=d4, pos=pos, dist=dist,
               g_task=g_task, g_total=g_tot, g_walk_share=gshare,
               wall=time.perf_counter() - t0, **inst)
    print("CHARLM EVAL %-16s s%d step %5d val %.4f | word_rate %.3f distinct4 %.3f | "
          "K %.1f K0 %.2f closed %.2f len %.1f | dread %.2f corr %.2f | gnorm %.2f walk share %.2e | %s"
          % (arm, seed, step, vl, wr, d4, inst.get("K", 0.0), inst.get("no_loop_frac", 0.0),
             inst.get("closed_frac", 0.0), inst.get("len_mean", 0.0), inst.get("dread_med", 0.0),
             inst.get("corr_norm", 0.0), g_task, gshare, "FINAL" if final else ""), flush=True)
    print("   sample: %s" % sample, flush=True)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="main", choices=list(ARMS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--out", default="")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--cache", default=os.environ.get("MELA_CACHE", ""),
                    help="compiled-kernel cache, local or gs://; a spot VM restarts often "
                         "and the chunk loop is a Python unroll, so this is not optional "
                         "on preemptible hardware")
    a = ap.parse_args()
    if a.cache:
        model.enable_compilation_cache(a.cache)
        print("compilation cache:", a.cache, flush=True)
    print("devices:", jax.devices(), flush=True)
    for arm in (list(ARMS) if a.all else [a.arm]):
        run(arm, a.seed, a.steps, a.d, a.T, a.B, a.layers, a.lr, a.mu,
            (a.out + "_" + arm.replace("+", "-")) if a.out else "",
            a.ckpt_every, a.eval_every)
    print("CHARLM_JAX_DONE", flush=True)
