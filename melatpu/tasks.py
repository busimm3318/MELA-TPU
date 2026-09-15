"""LOOPWORD in numpy + a JAX classifier head, so the task the walk is judged on
can run on a TPU.

The generator is a faithful port of the reference harness: a unicyclic labelled
graph (one directed cycle of length L, plus tree edges pointing outward, so
leaves are genuine sinks), the edges shuffled and emitted as triples, then
`[Q] start`; the answer is the ordered product of the group labels around the
cycle, walking from `start` in the arrow direction. The abelian, order-blind
predictor is at chance on A5 because A5 is perfect, so any accuracy above the
order-blind ceiling is order-using by proof.

Random numbers are NOT bit-compatible with the reference generator, and do not
need to be: gate J-H feeds the SAME batches to both harnesses rather than
relying on two random number generators agreeing.
"""
from __future__ import annotations

import itertools
import os

import jax
import jax.numpy as jnp
import numpy as np

from . import core, model

F32 = jnp.float32


def parity(p) -> int:
    s = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                s += 1
    return s % 2


def make_group(k: int = 3, even_only: bool = False):
    """S_k, or A_k when even_only. Returns (elements, multiplication table,
    identity index). Convention (a*b)(x) = a[b[x]], as in the reference."""
    els = [tuple(p) for p in itertools.permutations(range(k))
           if not even_only or parity(p) == 0]
    idx = {e: i for i, e in enumerate(els)}
    table = np.array([[idx[tuple(a[b[x]] for x in range(k))] for b in els] for a in els],
                     dtype=np.int64)
    return els, table, idx[tuple(range(k))]


class LoopWord:
    """vocab layout: [node_0..node_{K-1}] [g_0..g_{G-1}] [QUERY] [PAD]."""

    def __init__(self, K=8, k=4, L_choices=(3, 4, 5, 6), n_distract=(2, 6),
                 seed=0, even_only=False, path=None):
        self.K, self.k = K, k
        els, self.table, self.ident = make_group(k, even_only)
        self.G = len(els)
        self.L_choices = tuple(int(x) for x in os.environ["LOOPWORD_L"].split(",")) \
            if os.environ.get("LOOPWORD_L") else tuple(L_choices)
        self.n_distract = n_distract
        self.path = (os.environ.get("LOOPWORD_PATH", "0") == "1") if path is None else path
        self.QUERY, self.PAD = K + self.G, K + self.G + 1
        self.vocab = K + self.G + 2
        self.rng = np.random.default_rng(seed)

    def sample(self):
        L = int(self.rng.choice(self.L_choices))
        nodes = list(self.rng.permutation(self.K))
        cyc = nodes[:L]
        edges = [(cyc[i], int(self.rng.integers(self.G)), cyc[(i + 1) % L]) for i in range(L)]
        nd = int(self.rng.integers(self.n_distract[0], self.n_distract[1] + 1))
        pool = nodes[L:]
        for j in range(min(nd, len(pool))):
            src = (cyc + pool[:j])[int(self.rng.integers(L + j))]
            edges.append((src, int(self.rng.integers(self.G)), pool[j]))
        edges = [edges[i] for i in self.rng.permutation(len(edges))]
        s_i = int(self.rng.integers(L))
        start = cyc[s_i]
        prod = self.ident
        for i in range(L):
            u = cyc[(s_i + i) % L]
            g = next(e[1] for e in edges if e[0] == u and e[2] == cyc[(s_i + i + 1) % L])
            prod = int(self.table[g][prod])                 # g * prod
        if self.path:                                       # rung 1: the cycle only, in order
            edges = [next(e for e in edges if e[0] == cyc[(s_i + i) % L]
                          and e[2] == cyc[(s_i + i + 1) % L]) for i in range(L)]
        toks = []
        for u, g, v in edges:
            toks += [int(u), self.K + g, int(v)]
        toks += [self.QUERY, int(start)]
        return toks, prod, L

    def batch(self, B, T=None):
        items = [self.sample() for _ in range(B)]
        T = T or max(len(t[0]) for t in items)
        x = np.full((B, T), self.PAD, dtype=np.int32)
        last = np.zeros(B, dtype=np.int32)
        for i, (toks, _, _) in enumerate(items):
            x[i, :len(toks)] = toks                          # LEFT-align, PAD after
            last[i] = len(toks) - 1                          # the query's node token
        y = np.array([t[1] for t in items], dtype=np.int32)
        Ls = np.array([t[2] for t in items], dtype=np.int32)
        return x, y, last, Ls

    # ---- task gates (the 2026-09-05 checks, kept with the generator) ----
    def order_blind_ceiling(self, L, trials=20000):
        """The bag-of-labels oracle: the best an order-blind predictor can do.
        Any accuracy above this is order-using by proof."""
        from collections import Counter
        hit = 0
        for _ in range(trials):
            labs = self.rng.integers(self.G, size=L)
            counts = Counter()
            for perm in itertools.permutations(labs):
                p = self.ident
                for g in perm:
                    p = int(self.table[g][p])
                counts[p] += 1
            hit += max(counts.values()) / sum(counts.values())
        return hit / trials

    def order_flip_rate(self, trials=4000):
        """Swapping two adjacent cycle labels must change the answer often, or the
        task does not test order (A5: 91.7%, S3 50%, D4 37.5%)."""
        changed = 0
        for _ in range(trials):
            L = int(self.rng.choice([c for c in self.L_choices if c >= 3]))
            labs = list(self.rng.integers(self.G, size=L))
            i = int(self.rng.integers(L - 1))
            swapped = labs[:i] + [labs[i + 1], labs[i]] + labs[i + 2:]
            pa = pb = self.ident
            for g in labs:
                pa = int(self.table[g][pa])
            for g in swapped:
                pb = int(self.table[g][pb])
            changed += int(pa != pb)
        return changed / trials


# --------------------------------------------------------------------------- classifier
def init_classifier(key, vocab, cfg, layers, n_cls):
    d = cfg["d"]
    P = model.init_lm(key, vocab, cfg, layers)
    hk = jax.random.fold_in(key, 999)
    P = dict(P, head_w=model._lin(hk, n_cls, d), head_b=jnp.zeros((n_cls,), F32))
    return P


def classifier_forward(P, cfg, x, last, us=None, key=None):
    """Reads the block stack at each example's own query position."""
    B = x.shape[0]
    if us is None:
        us = model.walk_uniform_tree(key, len(P["blocks"]), model.n_events(cfg), B,
                                     cfg["n_walks"], cfg["walk_len"])
    h = P["emb"][x]
    insts, lps = [], []
    for bi, blk in enumerate(P["blocks"]):
        h, inst, lp = model._block(cfg, blk, h, us[bi])
        insts.append(inst)
        lps.append(lp)
    h = model.layernorm(h, P["nf_g"], P["nf_b"])
    h_last = h[jnp.arange(B), last]
    return h_last @ P["head_w"].T + P["head_b"], insts, lps


def cls_loss(P, cfg, x, y, last, us=None, key=None, stratum=None):
    """Classification cross-entropy plus the walk term. One event per sequence,
    so the per-example reward is the negative example loss; the baseline is the
    mean within a cycle-length group when one is given, because the reward varies
    far more with the cycle length than with the routing."""
    logits, insts, lps = classifier_forward(P, cfg, x, last, us=us, key=key)
    logp = jax.nn.log_softmax(logits, axis=-1)
    per = -jnp.take_along_axis(logp, y[:, None].astype(jnp.int32), axis=-1)[:, 0]
    task = per.mean()
    mu = cfg.get("mu_walk", 0.0)
    if not mu:
        return task, (task, insts, logits)
    r = jax.lax.stop_gradient(-per)
    if stratum is None:
        adv = r - r.mean()
    else:
        g = stratum.astype(jnp.int32).reshape(-1)
        # the group count must be static under jit; the cycle length is bounded by
        # the walk length, so a fixed 64 covers every stratum this task produces
        oh = jax.nn.one_hot(g, cfg.get("n_strata", 64), dtype=r.dtype)
        adv = r - oh @ ((oh.T @ r) / jnp.maximum(oh.sum(0), 1.0))
    terms = [-(adv * lp.mean(-1)).mean() for blk in lps for lp in blk]
    walk = jnp.stack(terms).sum() / max(1, len(lps))
    return task + mu * walk, (task, insts, logits)
