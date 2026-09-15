"""LOOPWORD on TPU: the structure diagnosis, in JAX.

Same arms and the same reading rules as the PyTorch harness; the model side is
already gated against it (J-D), and gate J-H covers this harness. Built for a
preemptible machine: the state is checkpointed every `--ckpt-every` steps and a
restart resumes from the last one, so losing the machine costs minutes.

    python loopword_jax.py --arm main --steps 25000 --out gs://BUCKET/run1
    LOOPWORD_PATH=1 LOOPWORD_L=2 python loopword_jax.py --arm main --steps 5000

Arms (each the same configuration except the switch it names):
  main         D1-D6 + the log-spaced decay -- the model-wide setting
  oracle-dir   direction gates frozen at (0, 1): the upper bound on the direction
               search, never a comparison row, and it takes nothing from the task
  legacy-walk  the design before the walk fixes
  dead-hol     main with the transport angle at zero: the same walk, no content
  legacy       the frozen design
  pair         one angle per edge: provably cannot separate elements of different
               order, so it is the counterexample arm
  slot         no events at all: the walk's contribution floor
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from melatpu import core, model, tasks

ARMS = ("main", "oracle-dir", "legacy-walk", "dead-hol", "legacy", "pair", "slot")


def make_cfg(arm: str, d: int, T: int, k_event: int, mu: float, **kw):
    base = dict(chunk=k_event, mu_walk=mu, **kw)
    if arm in ("main", "oracle-dir"):
        cfg = core.config_main(d, T, **base)
    elif arm == "legacy-walk":
        cfg = core.config(d, T, decay=True, transport_scale="example", mu_walk=mu, chunk=k_event, **kw)
    elif arm == "dead-hol":
        cfg = core.config_main(d, T, theta0=0.0, theta0_learn=False, **base)
    elif arm == "legacy":
        cfg = core.config(d, T, chunk=k_event, **kw)
    elif arm == "pair":
        cfg = core.config_main(d, T, transport_scale="pair", theta0=2.35, **base)
    elif arm == "slot":
        cfg = core.config_main(d, T, **base)
        cfg = dict(cfg, k_event=T + 1)                 # no event ever fires
    else:
        raise ValueError(arm)
    if arm != "slot":
        cfg = dict(cfg, k_event=k_event)
    return cfg


def freeze_direction(P, forward_on: bool):
    """oracle-dir: pin the gates at (f, r) = (0, 1) or (1, 0). A global one-bit
    flip that `main` can also reach by learning, so it bounds the search rather
    than adding information."""
    blocks = []
    for blk in P["blocks"]:
        mix = dict(blk["mix"])
        if "to_fwd_w" in mix:
            for nm, on in (("fwd", forward_on), ("rev", not forward_on)):
                mix[f"to_{nm}_w"] = jnp.zeros_like(mix[f"to_{nm}_w"])
                mix[f"to_{nm}_b"] = jnp.full_like(mix[f"to_{nm}_b"], 20.0 if on else -20.0)
        blocks.append(dict(blk, mix=mix))
    return dict(P, blocks=blocks)


FROZEN = ("to_fwd_w", "to_fwd_b", "to_rev_w", "to_rev_b")


def mask_frozen(P, freeze: bool):
    """optax mask: True where the parameter trains."""
    def walk(node, path=()):
        if isinstance(node, dict):
            return {k: walk(v, path + (k,)) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v, path) for v in node]
        return not (freeze and path and path[-1] in FROZEN)
    return walk(P)


# --------------------------------------------------------------------------- checkpoints
def ckpt_write(path, step, P, opt_state, log):
    blob = pickle.dumps(dict(step=step, P=jax.device_get(P),
                             opt_state=jax.device_get(opt_state), log=log))
    tmp = path + ".tmp"
    _write(tmp, blob)
    _move(tmp, path)


def ckpt_read(path):
    blob = _read(path)
    return None if blob is None else pickle.loads(blob)


def _is_gs(p):
    return str(p).startswith("gs://")


def _write(p, blob):
    if _is_gs(p):
        import subprocess
        subprocess.run(["gsutil", "-q", "cp", "-", p], input=blob, check=True)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
        open(p, "wb").write(blob)


def _read(p):
    if _is_gs(p):
        import subprocess
        r = subprocess.run(["gsutil", "-q", "cat", p], capture_output=True)
        return r.stdout if r.returncode == 0 and r.stdout else None
    return open(p, "rb").read() if os.path.exists(p) else None


def _move(a, b):
    if _is_gs(a):
        import subprocess
        subprocess.run(["gsutil", "-q", "mv", a, b], check=True)
    else:
        os.replace(a, b)


# --------------------------------------------------------------------------- run
def run(arm, seed, steps, d, B, layers, lr, mu, K, k, even_only, out, ckpt_every, eval_every):
    task = tasks.LoopWord(K=K, k=k, seed=seed, even_only=even_only)
    T = 3 * (max(task.L_choices) + (0 if task.path else task.n_distract[1])) + 2
    k_event = 3 * (min(task.L_choices) + (0 if task.path else task.n_distract[0])) + 1
    cfg = make_cfg(arm, d, T, k_event, mu)
    key = jax.random.key(seed)
    P = tasks.init_classifier(key, task.vocab, cfg, layers, task.G)
    if arm == "oracle-dir":
        P = freeze_direction(P, forward_on=False)          # against the arrows
    n_ev = max(1, len(range(cfg["k_event"], T, cfg["k_event"])))

    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.masked(optax.adamw(lr, weight_decay=1e-2),
                                  mask_frozen(P, arm == "oracle-dir")))
    opt_state = tx.init(P)
    log = {"arm": arm, "seed": seed, "cfg": {kk: str(v) for kk, v in cfg.items()},
           "T": T, "k_event": k_event, "curve": [], "evals": []}
    start = 0
    ck = (out + ".ckpt") if out else None
    if ck and (st := ckpt_read(ck)):
        P, opt_state, start, log = st["P"], st["opt_state"], st["step"], st["log"]
        print(f"RESUMED at step {start}", flush=True)

    @jax.jit
    def step_fn(P, opt_state, x, y, last, us, stratum):
        (loss, (task_l, insts, _)), g = jax.value_and_grad(
            tasks.cls_loss, has_aux=True)(P, cfg, x, y, last, us=us, stratum=stratum)
        gnorm = optax.global_norm(g)
        upd, opt_state = tx.update(g, opt_state, P)
        return optax.apply_updates(P, upd), opt_state, loss, task_l, insts, gnorm

    @jax.jit
    def eval_fn(P, x, last, us):
        logits, insts, _ = tasks.classifier_forward(P, cfg, x, last, us=us)
        return logits.argmax(-1), insts

    def uniforms(step):
        return model.walk_uniform_tree(jax.random.fold_in(key, step), layers, n_ev,
                                       B, cfg["n_walks"], cfg["walk_len"])

    if cfg.get("theta0_learn") and start == 0:
        x, _, last, _ = task.batch(B, T)
        Pl = dict(P)
        _, insts, _ = tasks.classifier_forward(Pl, cfg, jnp.asarray(x), jnp.asarray(last), us=uniforms(0))
        blocks = []
        for blk, inst in zip(P["blocks"], insts):
            mix = blk["mix"]
            if "log_theta0" in mix and inst:
                gen = jnp.stack([e["gen_norm"] for e in inst]).mean()
                mix = dict(mix, log_theta0=jnp.log(cfg["theta0_target"] / jnp.maximum(gen, 1e-8)))
            blocks.append(dict(blk, mix=mix))
        P = dict(P, blocks=blocks)
        opt_state = tx.init(P)
        print("D5 theta0 ->", [float(jnp.exp(b["mix"]["log_theta0"]))
                               for b in P["blocks"] if "log_theta0" in b["mix"]], flush=True)

    ev = tasks.LoopWord(K=K, k=k, seed=10_000 + seed, even_only=even_only)
    t0 = time.perf_counter()
    for step in range(start + 1, steps + 1):
        x, y, last, Ls = task.batch(B, T)
        P, opt_state, loss, task_l, insts, gnorm = step_fn(
            P, opt_state, jnp.asarray(x), jnp.asarray(y), jnp.asarray(last),
            uniforms(step), jnp.asarray(Ls))
        if step % 100 == 0 or step == 1:
            fl = jax.tree_util.tree_map(float, insts[0][0]) if insts and insts[0] else {}
            log["curve"].append(dict(step=step, loss=float(loss), task=float(task_l),
                                     gnorm=float(gnorm),
                                     **{kk: v for kk, v in fl.items() if np.ndim(v) == 0}))
        if step % 500 == 0 or step == 1:
            print("%-11s s%d %6d/%d loss %.4f gnorm %.2f (%.0fs)"
                  % (arm, seed, step, steps, float(loss), float(gnorm),
                     time.perf_counter() - t0), flush=True)
        if eval_every and (step % eval_every == 0 or step == steps):
            corr = {}
            cnt = {}
            inst_last = None
            for j in range(10):
                xe, ye, le, Le = ev.batch(B, T)
                pred, inst_last = eval_fn(P, jnp.asarray(xe), jnp.asarray(le), uniforms(1_000_000 + step * 16 + j))
                pred = np.asarray(pred)
                for L_, p_, t_ in zip(Le.tolist(), pred.tolist(), ye.tolist()):
                    cnt[L_] = cnt.get(L_, 0) + 1
                    corr[L_] = corr.get(L_, 0) + int(p_ == t_)
            acc = sum(corr.values()) / max(sum(cnt.values()), 1)
            fl = jax.tree_util.tree_map(float, inst_last[0][0]) if inst_last and inst_last[0] else {}
            row = dict(step=step, acc=acc, acc_L={str(L_): corr[L_] / cnt[L_] for L_ in sorted(cnt)},
                       wall=time.perf_counter() - t0,
                       **{kk: v for kk, v in fl.items() if np.ndim(v) == 0})
            log["evals"].append(row)
            print("LOOPWORD_JAX EVAL %-11s s%d step %6d acc %.3f | by L %s | K %.1f K0 %.2f "
                  "closed %.2f dead %.2f len %.1f top %.2f fwd %.2f"
                  % (arm, seed, step, acc,
                     " ".join("L%s %.3f" % (L_, corr[L_] / cnt[L_]) for L_ in sorted(cnt)),
                     row.get("K", 0), row.get("no_loop_frac", 0), row.get("closed_frac", 0),
                     row.get("dead_frac", 0), row.get("len_mean", 0),
                     row.get("top_slot_share", 0), row.get("fwd_ratio", 0)), flush=True)
            if out:
                _write(out + ".json", json.dumps(log, indent=1).encode())
        if ck and step % ckpt_every == 0:
            ckpt_write(ck, step, P, opt_state, log)
    if out:
        _write(out + ".json", json.dumps(log, indent=1).encode())
        ckpt_write(ck, steps, P, opt_state, log)
    print("LOOPWORD_JAX_DONE", arm, seed, flush=True)
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="main", choices=list(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--group", default="A", choices=["S", "A"])
    ap.add_argument("--out", default="")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--cache", default=os.environ.get("MELA_CACHE", ""),
                    help="compiled-kernel cache directory, local or gs://. The chunk loop is a "
                         "Python unroll, so compilation is long and paid again on every restart "
                         "unless this is set -- and a spot VM restarts often.")
    a = ap.parse_args()
    if a.cache:
        model.enable_compilation_cache(a.cache)
        print("compilation cache:", a.cache, flush=True)
    print("devices:", jax.devices(), flush=True)
    run(a.arm, a.seed, a.steps, a.d, a.B, a.layers, a.lr, a.mu, a.K, a.k,
        a.group == "A", a.out, a.ckpt_every, a.eval_every)
