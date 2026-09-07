"""MELA-TPU core in JAX (port of MELA-260906, frozen design 2026-09-06).

Functional, jit-able, static shapes; walk randomness is an explicit uniform
array; the sequential sampler and the holonomy chain are lax.scan loops.
dtypes are explicit (float32 activations, int32 indices) so the code does not
depend on the x64 flag.

Parameters (dict of arrays, names follow MELA-260906):
  to_theta_w [n/2, d], to_theta_b [n/2], to_k_w / to_q_w / to_v_w [n, d],
  to_gate_w [M, d], to_gate_b [M], to_out_member_w/b, to_in_member_w/b,
  from_read_w [d, n], probe [n], walk_q_w [n, n], walk_k_w [n, n], gain [], carry_bias [n, n]
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

EPS = 1e-12
_B12 = [
    [9.0198e-16, 0.46932117595418237389, -0.20099424927047284052, -0.04623946134063071740],
    [5.31597895759871264183, 1.19926790417132231573, 0.01179296240992997031, 0.01108844528519167989],
    [0.18188869982170434744, 0.05502798439925399070, 0.09351590770535414968, 0.00610700528898058230],
    [-2.0861320e-13, -0.13181061013830184015, -0.02027855540589259079, -0.00675951846863086359],
]
F32 = jnp.float32
I32 = jnp.int32


# --------------------------------------------------------------------------- sampler
def _pick(logits, u):
    """Inverse-CDF categorical pick per row. logits [R, M], u [R] -> idx [R] int32, logp of pick [R]."""
    logp = jax.nn.log_softmax(logits, axis=-1)
    cdf = jnp.cumsum(jnp.exp(logp), axis=-1)
    # compare-and-sum inverse CDF (== searchsorted left); TPU-friendly for tables <= 512
    idx = (cdf < u.astype(cdf.dtype)[:, None]).sum(-1)
    idx = jnp.minimum(idx, logits.shape[-1] - 1).astype(I32)
    lp = jnp.take_along_axis(logp, idx[:, None], axis=-1)[:, 0]
    return idx, lp


def sample_slots(A, in_mean, u, L):
    """A [B,M,M], in_mean [B,M], u [B,W,L+1,2]. Returns dict with pairs [B,W,L] (slot*M+next),
    live [B,W,L], closed [B,W], length [B,W], member [B,W,M], visited [B,W,L+1], logprob [B,W]."""
    B, M, _ = A.shape
    W = u.shape[1]
    logA = jnp.log(jnp.maximum(A, EPS))
    start_logits = jnp.broadcast_to(jnp.log(jnp.maximum(in_mean, EPS))[:, None, :], (B, W, M))
    slot, lp0 = _pick(start_logits.reshape(B * W, M), u[:, :, 0, 0].reshape(-1))
    slot = slot.reshape(B, W)
    visited0 = jnp.full((B, W, L + 1), -1, dtype=I32).at[:, :, 0].set(slot)
    ar = jnp.arange(M, dtype=I32)[None, None, :]

    def body(carry, inp):
        slot, prev, closed, length, close_slot, logprob, visited = carry
        u_step, k = inp
        row = jax.vmap(lambda la, s: la[s])(logA, slot)      # [B,W,M]
        ban = (ar == prev[..., None]) & (prev >= 0)[..., None]
        has_alt = ((~ban) & (row > -18.0)).any(-1, keepdims=True)
        row = jnp.where(ban & has_alt, -jnp.inf, row)
        nxt, lp = _pick(row.reshape(B * W, M), u_step.reshape(-1))
        nxt, lp = nxt.reshape(B, W), lp.reshape(B, W)
        live = ~closed
        pair = slot * M + nxt
        prev = jnp.where(live, slot, prev)
        logprob = logprob + jnp.where(live, lp, 0.0)
        hit = (visited == nxt[..., None]).any(-1) & live
        close_slot = jnp.where(hit, nxt, close_slot)
        length = jnp.where(hit, k, length)
        closed = closed | hit
        visited = visited.at[:, :, k].set(nxt)
        slot = jnp.where(live, nxt, slot)
        return (slot, prev, closed, length, close_slot, logprob, visited), (pair, live)

    init = (slot, jnp.full((B, W), -1, I32), jnp.zeros((B, W), bool), jnp.zeros((B, W), I32),
            jnp.full((B, W), -1, I32), lp0.reshape(B, W), visited0)
    steps = (jnp.moveaxis(u[:, :, 1:, 0], 2, 0), jnp.arange(1, L + 1, dtype=I32))
    (slot, prev, closed, length, close_slot, logprob, visited), (pairs, lives) = lax.scan(body, init, steps)
    pairs, lives = jnp.moveaxis(pairs, 0, 2), jnp.moveaxis(lives, 0, 2)
    hit_pos = visited == close_slot[..., None]
    first = jnp.argmax(hit_pos.astype(jnp.int8), axis=-1)
    pos = jnp.arange(L + 1, dtype=I32)[None, None, :]
    in_cycle = (pos >= first[..., None]) & (pos <= length[..., None]) & closed[..., None]
    onehot = jax.nn.one_hot(jnp.maximum(visited, 0), M, dtype=F32)
    member = (onehot * in_cycle.astype(F32)[..., None]).sum(2)
    member = member / jnp.maximum(member.sum(-1, keepdims=True), EPS)
    return dict(pairs=pairs, live=lives, closed=closed, length=length, member=member,
                visited=visited, logprob=logprob)


# --------------------------------------------------------------------------- pair state
def pair_mass(out_step, in_member):
    A = jnp.einsum("btm,btn->bmn", out_step, in_member)
    return A + jnp.swapaxes(A, 1, 2)


def pair_gather(out_step, in_member, head, v, k_prev, pairs):
    B, T, M = out_step.shape
    n = v.shape[-1]
    W, L = pairs.shape[1], pairs.shape[2]
    flat = jnp.broadcast_to(pairs.reshape(B, 1, W * L), (B, T, W * L))
    m, m2 = flat // M, flat % M
    g = lambda x, idx: jnp.take_along_axis(x, idx, axis=2)
    o_m, o_m2, i_m, i_m2, h_m = g(out_step, m), g(out_step, m2), g(in_member, m), g(in_member, m2), g(head, m)
    wH = h_m * (o_m * i_m2 + i_m * o_m2)
    wA = o_m * i_m2 + i_m * o_m2
    X = (v[..., :, None] * k_prev[..., None, :]).reshape(B, T, n * n)
    Hp = jnp.einsum("btp,btx->bpx", wH, X)
    return Hp.reshape(B, W, L, n, n), wA.sum(1).reshape(B, W, L)


# --------------------------------------------------------------------------- transport
def _expm_t12(A):
    I = jnp.broadcast_to(jnp.eye(A.shape[-1], dtype=A.dtype), A.shape)
    A2 = A @ A
    A3 = A2 @ A
    P = [I, A, A2, A3]
    Bs = [sum(c * X for c, X in zip(row, P)) for row in _B12]
    A6 = Bs[3] @ Bs[3] + Bs[2]
    return Bs[0] + (Bs[1] + A6) @ A6


def expm_fixed(A, squarings):
    x = A / (2 ** squarings)
    out = _expm_t12(x)
    for _ in range(squarings):
        out = out @ out
    return out


def transport(st, theta0, squarings, angle_clamp):
    A = st - jnp.swapaxes(st, -1, -2)
    raw = jnp.linalg.norm(A.reshape(*A.shape[:-2], -1), axis=-1) / 2 ** 0.5       # [B,W,L]
    sc = jnp.sqrt(jnp.mean(raw ** 2, axis=(0, 1), keepdims=True))
    sc = jnp.maximum(jnp.broadcast_to(sc, raw.shape), 1e-4)
    ang = lax.stop_gradient(theta0 * raw / sc)
    gain = jnp.minimum(angle_clamp / jnp.maximum(ang, 1e-12), 1.0)
    omega = theta0 * A / sc[..., None, None] * gain[..., None, None]
    R = expm_fixed(omega, squarings)
    inst = dict(max_angle=ang.max(), mean_angle=ang.mean(), floor_frac=(raw < 1e-4).astype(F32).mean(),
                clamp_frac=(ang > angle_clamp).astype(F32).mean())
    return R, inst


def chain(R, live):
    B, W, L, n, _ = R.shape
    hol0 = jnp.broadcast_to(jnp.eye(n, dtype=R.dtype), (B, W, n, n))

    def body(hol, inp):
        Rl, ll = inp
        return jnp.where(ll[..., None, None], Rl @ hol, hol), None

    hol, _ = lax.scan(body, hol0, (jnp.moveaxis(R, 2, 0), jnp.moveaxis(live, 2, 0)))
    return hol


# --------------------------------------------------------------------------- dedup (static)
def dedup_static(member, closed):
    """rep [B,W]: lowest-index closed walk of each distinct slot set. Two 16-bit hashes of
    the slot-set bit vector (int32-safe), lexicographic stable sort, first of each run."""
    B, W, M = member.shape
    bits = (member > 0).astype(I32)
    mult = jnp.arange(1, M + 1, dtype=I32)
    h1 = (bits * ((mult * 40503) % 65521)).sum(-1) % 65521          # < 2^16, sums stay < 2^31
    h2 = (bits * ((mult * mult * 4099) % 65519)).sum(-1) % 65519
    h1 = jnp.where(closed, h1, 70000 + jnp.arange(W, dtype=I32)[None, :])   # open walks last, distinct
    widx = jnp.broadcast_to(jnp.arange(W, dtype=I32)[None, :], (B, W))
    order = jax.vmap(lambda a, b, c: jnp.lexsort((c, b, a)))(h1, h2, widx)   # by h1, then h2, then index
    s1 = jnp.take_along_axis(h1, order, axis=-1)
    s2 = jnp.take_along_axis(h2, order, axis=-1)
    first_sorted = jnp.concatenate([jnp.ones((B, 1), bool), (s1[:, 1:] != s1[:, :-1]) | (s2[:, 1:] != s2[:, :-1])], axis=1)
    rep = jnp.zeros((B, W), bool).at[jnp.arange(B)[:, None], order].set(first_sorted)
    return rep & closed


# --------------------------------------------------------------------------- interior
def interior(phi, head, q, k, v, C, corr_at):
    """corr_at: list of (event_index_in_chunks, corr [B,M,n,n]) with static chunk indices."""
    B, T, M = phi.shape
    n = q.shape[-1]
    T0 = T
    if T % C:
        pad = C - T % C
        phi, head, q, k, v = (jnp.pad(x, ((0, 0), (0, pad), (0, 0))) for x in (phi, head, q, k, v))
        T = T + pad
    nC = T // C
    Ph, Hd = phi.reshape(B, nC, C, M), head.reshape(B, nC, C, M)
    Q, K, V = (x.reshape(B, nC, C, n) for x in (q, k, v))
    X = (V[..., :, None] * K[..., None, :]).reshape(B, nC, C, n * n)
    delta = jnp.einsum("bxcm,bxcz->bxmz", Hd, X).reshape(B, nC, M, n, n)
    incl = jnp.cumsum(delta, axis=1)
    s_start = jnp.concatenate([jnp.zeros_like(incl[:, :1]), incl[:, :-1]], axis=1)
    for e_chunk, corr in corr_at:
        mask = (jnp.arange(nC) >= e_chunk).astype(F32)[None, :, None, None, None]
        s_start = s_start + mask * corr[:, None]
    causal = jnp.tril(jnp.ones((C, C), F32))
    G = Ph @ jnp.swapaxes(Hd, -1, -2)
    Aq = Q @ jnp.swapaxes(K, -1, -2)
    intra = (G * Aq * causal) @ V
    U = jnp.einsum("bxcm,bxmz->bxcz", Ph, s_start.reshape(B, nC, M, n * n))
    inter = (U.reshape(B, nC, C, n, n) @ Q[..., None])[..., 0]
    return (intra + inter).reshape(B, T, n)[:, :T0]


# --------------------------------------------------------------------------- layer
def _shift(x):
    return jnp.concatenate([jnp.zeros_like(x[:, :1]), x[:, :-1]], axis=1)


def rotate(x, theta):
    n = x.shape[-1]
    cos, sin = jnp.cos(theta), jnp.sin(theta)
    pair = x.reshape(*x.shape[:-1], n // 2, 2)
    a, b = pair[..., 0], pair[..., 1]
    return jnp.stack([cos * a - sin * b, sin * a + cos * b], axis=-1).reshape(x.shape)


def projections(P, h):
    theta = h @ P["to_theta_w"].T + P["to_theta_b"]
    k = rotate(h @ P["to_k_w"].T, theta)
    q = rotate(h @ P["to_q_w"].T, theta)
    v = h @ P["to_v_w"].T
    phi = jax.nn.softmax(h @ P["to_gate_w"].T + P["to_gate_b"], axis=-1)
    om = jax.nn.softmax(h @ P["to_out_member_w"].T + P["to_out_member_b"], axis=-1)
    im = jax.nn.softmax(h @ P["to_in_member_w"].T + P["to_in_member_b"], axis=-1)
    return dict(k=k, q=q, v=v, phi=phi, im=im, head=_shift(phi), k_prev=_shift(k), out_step=_shift(om))


def walk_event(cfg, o, im, hd, v, kp, in_mean, u):
    A = pair_mass(o, im)
    w = sample_slots(A, in_mean, u, cfg["walk_len"])
    Hp, Ap = pair_gather(o, im, hd, v, kp, w["pairs"])
    st = Hp / jnp.maximum(Ap, EPS)[..., None, None]
    R, tinst = transport(st, cfg["theta0"], cfg["squarings"], cfg["angle_clamp"])
    hol = chain(R, w["live"])
    return hol, w, R, tinst


def layer_forward(P, cfg, h, u_events):
    """h [B,T,d]; u_events: list (per event) of [B,W,L+1,2] uniforms. Returns out [B,T,d], instruments."""
    B, T, d = h.shape
    n, M, C, k_event = cfg["n"], cfg["M"], cfg["chunk"], cfg["k_event"]
    p = projections(P, h)
    events = list(range(k_event, T, k_event))
    corr_at, inst = [], []
    sw_I = None
    walk_fn = jax.checkpoint(lambda *a: walk_event(cfg, *a)) if cfg.get("recompute", True) else (lambda *a: walk_event(cfg, *a))
    for i, t_e in enumerate(events):
        lo = 0 if i == 0 else events[i - 1]
        dI = p["im"][:, lo:t_e].sum(1)
        sw_I = dI if sw_I is None else sw_I + dI
        in_mean = sw_I / t_e
        hol, w, R, tinst = walk_fn(p["out_step"][:, lo:t_e], p["im"][:, lo:t_e], p["head"][:, lo:t_e],
                                   p["v"][:, lo:t_e], p["k_prev"][:, lo:t_e], in_mean, u_events[i])
        rep = dedup_static(w["member"], w["closed"])
        valid = w["closed"] & rep
        chi = jnp.swapaxes(w["member"] * rep[..., None].astype(F32), 1, 2)          # [B,M,W]
        feat = hol @ P["probe"]                                                  # [B,W,n]: holonomy's action on the probe (MELA-260907)
        logits = (feat @ P["walk_q_w"].T) @ jnp.swapaxes(feat @ P["walk_k_w"].T, -1, -2) / n ** 0.5
        mask = valid[:, None, :]
        logits = jnp.where(mask, logits, -jnp.inf)
        logits = jnp.where(mask.any(-1, keepdims=True), logits, 0.0)
        attn = jax.nn.softmax(logits, axis=-1)
        vals = hol + P["carry_bias"]
        delta = jnp.einsum("bkj,bjpq->bkpq", attn, vals) - vals
        update = P["gain"] * jnp.einsum("bmk,bkpq->bmpq", chi, delta)
        wsel = valid.astype(F32) / jnp.maximum(valid.sum(-1, keepdims=True), 1).astype(F32)
        vmean = jnp.einsum("bw,bwpq->bpq", wsel, vals)
        mass = chi.sum(-1)[..., None, None]
        update = update + mass * vmean[:, None]
        corr_at.append((t_e // C, update))
        cf = w["closed"].astype(F32); denom = jnp.maximum(cf.sum(), 1.0)
        eye = jnp.eye(n, dtype=F32)
        inst.append(dict(K=rep.sum(-1).astype(F32).mean(), closed_frac=cf.mean(),
                         hol_norm=(jnp.linalg.norm(hol.reshape(B, -1, n * n), axis=-1) * cf).sum() / denom,
                         orth_drift=((jnp.abs(jnp.swapaxes(hol, -1, -2) @ hol - eye).reshape(B, -1, n * n).max(-1)) * cf).sum() / denom,
                         occupancy=((w["member"] > 0) & w["closed"][..., None]).any(1).astype(F32).mean(),
                         **tinst))
    read = interior(p["phi"], p["head"], p["q"], p["k_prev"], p["v"], C, corr_at)
    return read @ P["from_read_w"].T, inst


def config(d, T, chunk=None, walk_len=32, n_walks=64, theta0=1.0, squarings=3, angle_clamp=12.0, recompute=True):
    n, M = d // 4, d // 2
    return dict(d=d, n=n, M=M, T=T, k_event=max(T // 4, 8), chunk=(chunk or (128 if d >= 1024 else 64)),
                walk_len=walk_len, n_walks=n_walks, theta0=theta0, squarings=squarings, angle_clamp=angle_clamp,
                recompute=recompute)


def init_params(key, cfg):
    d, n, M = cfg["d"], cfg["n"], cfg["M"]
    ks = jax.random.split(key, 8)
    lin = lambda k, o, i: jax.random.uniform(k, (o, i), F32, -1 / i ** 0.5, 1 / i ** 0.5)
    return dict(to_theta_w=jnp.zeros((n // 2, d), F32), to_theta_b=jnp.zeros((n // 2,), F32),
                to_k_w=lin(ks[0], n, d), to_q_w=lin(ks[1], n, d), to_v_w=lin(ks[2], n, d),
                to_gate_w=lin(ks[3], M, d), to_gate_b=jnp.zeros((M,), F32),
                to_out_member_w=lin(ks[4], M, d), to_out_member_b=jnp.zeros((M,), F32),
                to_in_member_w=lin(ks[5], M, d), to_in_member_b=jnp.zeros((M,), F32),
                from_read_w=lin(ks[6], d, n), probe=jax.random.normal(ks[7], (n,), F32) / n ** 0.5,
                walk_q_w=lin(ks[7], n, n), walk_k_w=lin(ks[7], n, n),
                gain=jnp.zeros((), F32), carry_bias=jnp.zeros((n, n), F32))
