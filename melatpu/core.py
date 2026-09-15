"""MELA-TPU core in JAX (port of the MELA PyTorch reference, package mela260907; design frozen 2026-09-06, read-out slimmed 2026-09-07).

Functional, jit-able, static shapes; walk randomness is an explicit uniform
array; the sequential sampler and the holonomy chain are lax.scan loops.
dtypes are explicit (float32 activations, int32 indices) so the code does not
depend on the x64 flag.

Parameters (dict of arrays, names follow the PyTorch reference):
  to_theta_w [n/2, d], to_theta_b [n/2], to_k_w / to_q_w / to_v_w [n, d],
  to_gate_w [M, d], to_gate_b [M], to_out_member_w/b, to_in_member_w/b,
  from_read_w [d, n], probe [n], walk_q_w [n, n], walk_k_w [n, n], gain [], carry_bias [n, n]
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

_SP1 = 0.5413248546129181   # softplus(_SP1) = 1 exactly
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


def sample_slots(A, in_mean, u, L, self_pair_mask=False, dead_end="escape", death_c=0.5):
    """A [B,M,M], in_mean [B,M], u [B,W,L+1,2]. Returns dict with pairs [B,W,L] (slot*M+next),
    live [B,W,L], closed [B,W], length [B,W], member [B,W,M], visited [B,W,L+1], logprob [B,W],
    dead [B,W], first [B,W].

    D3 (2026-09-14). dead_end="escape" is the frozen design: when the only viable
    neighbour is the banned predecessor the ban is LIFTED, the walk backtracks, and
    the returning slot is already in `visited`, so a dead end counts as a length-2
    closure -- a spurious node at every leaf. "die" kills the walk instead (not
    live, no log-probability, no closure) and never lifts the ban; death is judged
    AFTER the ban against a threshold death_c / M times the example's mean row mass.
    The 1/M matters: a near-uniform row puts 1/M of its mass in each entry, so an
    absolute constant tests the slot count rather than the routing."""
    B, M, _ = A.shape
    W = u.shape[1]
    die = dead_end == "die"
    logA = jnp.log(jnp.maximum(A, EPS))
    log_thr = jnp.log(jnp.maximum(death_c / M * A.sum(-1).mean(-1), EPS))[:, None, None]
    start_logits = jnp.broadcast_to(jnp.log(jnp.maximum(in_mean, EPS))[:, None, :], (B, W, M))
    slot, lp0 = _pick(start_logits.reshape(B * W, M), u[:, :, 0, 0].reshape(-1))
    slot = slot.reshape(B, W)
    visited0 = jnp.full((B, W, L + 1), -1, dtype=I32).at[:, :, 0].set(slot)
    ar = jnp.arange(M, dtype=I32)[None, None, :]

    def body(carry, inp):
        slot, prev, closed, length, close_slot, logprob, visited, dead = carry
        u_step, k = inp
        row = jax.vmap(lambda la, s: la[s])(logA, slot)      # [B,W,M]
        ban = (ar == prev[..., None]) & (prev >= 0)[..., None]
        if self_pair_mask:
            ban = ban | (ar == slot[..., None])
        if die:
            dead_now = ~(((~ban) & (row > log_thr)).any(-1))      # judged AFTER the ban
            row = jnp.where(ban, -jnp.inf, row)
        else:
            has_alt = ((~ban) & (row > -18.0)).any(-1, keepdims=True)
            row = jnp.where(ban & has_alt, -jnp.inf, row)
            dead_now = jnp.zeros_like(closed)
        nxt, lp = _pick(row.reshape(B * W, M), u_step.reshape(-1))
        nxt, lp = nxt.reshape(B, W), lp.reshape(B, W)
        live = (~closed) & (~dead) & (~dead_now)
        dead = dead | dead_now
        pair = slot * M + nxt
        hit = (visited == nxt[..., None]).any(-1) & live
        if self_pair_mask:
            hit = hit & (nxt != slot) & ((nxt != prev) | (prev < 0))
        prev = jnp.where(live, slot, prev)
        logprob = logprob + jnp.where(live, lp, 0.0)
        close_slot = jnp.where(hit, nxt, close_slot)
        length = jnp.where(hit, k, length)
        closed = closed | hit
        visited = visited.at[:, :, k].set(nxt)
        slot = jnp.where(live, nxt, slot)
        return (slot, prev, closed, length, close_slot, logprob, visited, dead), (pair, live)

    init = (slot, jnp.full((B, W), -1, I32), jnp.zeros((B, W), bool), jnp.zeros((B, W), I32),
            jnp.full((B, W), -1, I32), lp0.reshape(B, W), visited0, jnp.zeros((B, W), bool))
    steps = (jnp.moveaxis(u[:, :, 1:, 0], 2, 0), jnp.arange(1, L + 1, dtype=I32))
    (slot, prev, closed, length, close_slot, logprob, visited, dead), (pairs, lives) = lax.scan(body, init, steps)
    pairs, lives = jnp.moveaxis(pairs, 0, 2), jnp.moveaxis(lives, 0, 2)
    hit_pos = visited == close_slot[..., None]
    first = jnp.argmax(hit_pos.astype(jnp.int8), axis=-1)
    pos = jnp.arange(L + 1, dtype=I32)[None, None, :]
    in_cycle = (pos >= first[..., None]) & (pos <= length[..., None]) & closed[..., None]
    onehot = jax.nn.one_hot(jnp.maximum(visited, 0), M, dtype=F32)
    member = (onehot * in_cycle.astype(F32)[..., None]).sum(2)
    member = member / jnp.maximum(member.sum(-1, keepdims=True), EPS)
    return dict(pairs=pairs, live=lives, closed=closed, length=length, member=member,
                visited=visited, logprob=logprob, dead=dead, first=first)


def loop_edges_mask(first, length, closed, L):
    """D3. Edge l (visited[l] -> visited[l+1]) lies on the closed loop iff
    first <= l <= length-1 and the walk closed."""
    pos = jnp.arange(L, dtype=I32)[None, None, :]
    return (pos >= first[..., None]) & (pos < length[..., None]) & closed[..., None]


# --------------------------------------------------------------------------- pair state
def pair_mass(out_step, in_member, f=None, r=None):
    """D2: per-token forward / reverse gates (both ~1 at init = undirected = the
    frozen design) weight the two terms, so the layer learns how directed each
    relation is. f = r = None reproduces the symmetrised mass exactly."""
    if f is None and r is None:
        A = jnp.einsum("btm,btn->bmn", out_step, in_member)
        return A + jnp.swapaxes(A, 1, 2)
    of = out_step if f is None else out_step * f
    orv = out_step if r is None else out_step * r
    return (jnp.einsum("btm,btn->bmn", of, in_member)
            + jnp.swapaxes(jnp.einsum("btm,btn->bmn", orv, in_member), 1, 2))


def pair_gather(out_step, in_member, head, v, k_prev, pairs, f=None, r=None,
                reverse_transpose=False):
    """D2: the reverse term is the SAME token pair traversed the other way, so with
    reverse_transpose it carries X^T and is weighted by the head of its own
    departure slot m'. Then st(m'->m) = st(m->m')^T exactly and a reversed edge
    transports the transposed rotation (the connection condition; a backtrack then
    has identity holonomy). Also returns the forward share, the direction instrument."""
    B, T, M = out_step.shape
    n = v.shape[-1]
    W, L = pairs.shape[1], pairs.shape[2]
    flat = jnp.broadcast_to(pairs.reshape(B, 1, W * L), (B, T, W * L))
    m, m2 = flat // M, flat % M
    g = lambda x, idx: jnp.take_along_axis(x, idx, axis=2)
    o_m, o_m2, i_m, i_m2, h_m = g(out_step, m), g(out_step, m2), g(in_member, m), g(in_member, m2), g(head, m)
    wf = o_m * i_m2
    wr = i_m * o_m2
    if f is not None:
        wf = wf * f
    if r is not None:
        wr = wr * r
    wA = wf + wr
    X = (v[..., :, None] * k_prev[..., None, :]).reshape(B, T, n * n)
    if reverse_transpose:
        h_m2 = g(head, m2)
        Xr = (k_prev[..., :, None] * v[..., None, :]).reshape(B, T, n * n)
        Hp = jnp.einsum("btp,btx->bpx", h_m * wf, X) + jnp.einsum("btp,btx->bpx", h_m2 * wr, Xr)
    else:
        Hp = jnp.einsum("btp,btx->bpx", h_m * wA, X)
    tot = jnp.maximum(wA.sum(1), EPS)
    fwd_share = lax.stop_gradient((wf.sum(1) / tot).reshape(B, W, L))
    return Hp.reshape(B, W, L, n, n), wA.sum(1).reshape(B, W, L), fwd_share


# --------------------------------------------------------------------------- transport
_HI = jax.lax.Precision.HIGHEST   # rotation products at full fp32 operand precision on TPU (6 bf16 passes); the rest of the model keeps the default


def _mm(a, b):
    return jnp.matmul(a, b, precision=_HI)


def _expm_t12(A):
    I = jnp.broadcast_to(jnp.eye(A.shape[-1], dtype=A.dtype), A.shape)
    A2 = _mm(A, A)
    A3 = _mm(A2, A)
    P = [I, A, A2, A3]
    Bs = [sum(c * X for c, X in zip(row, P)) for row in _B12]
    A6 = _mm(Bs[3], Bs[3]) + Bs[2]
    return Bs[0] + _mm(Bs[1] + A6, A6)


def expm_fixed(A, squarings):
    x = A / (2 ** squarings)
    out = _expm_t12(x)
    for _ in range(squarings):
        out = _mm(out, out)
    return out


def _edge_rotation(st_l, theta0, squarings, angle_clamp, scale):
    """One edge's rotation, shared by the holonomy scan and the rebasing scan so the
    two cannot drift apart. Returns R and the per-edge angle.

    D5 (2026-09-14): scale="batch" is the frozen design (root-mean-square over the
    other walks at this step), which pins the root-mean-square angle at theta0 and
    scales every angle by a common factor. A rank-1 edge transform makes A rank 2,
    a single-plane rotation, and the faithful representation such a plane can carry
    fixes the angles (A5: 2pi/5, 4pi/5, 2pi/3, pi, root-mean-square 2.347), so a
    common rescale breaks the group relations at once and no exact representation
    exists. scale="none" is the approved rule: the angle is theta0 times the edge's
    own generator norm, a function of the edge content alone. scale="pair"
    normalises by that same norm, giving every edge the angle theta0 -- a function
    of the edge, but one angle cannot separate elements of different order, which is
    why that mode exists only as the counterexample arm."""
    B, W = st_l.shape[0], st_l.shape[1]
    A = st_l - jnp.swapaxes(st_l, -1, -2)
    raw = jnp.linalg.norm(A.reshape(B, W, -1), axis=-1) / 2 ** 0.5             # [B,W]
    if scale == "none":
        sc = jnp.ones_like(raw)
    elif scale == "pair":
        sc = lax.stop_gradient(jnp.maximum(raw, 1e-8))
    else:
        sc = jnp.maximum(jnp.sqrt(jnp.mean(raw ** 2)), 1e-4)                   # per-step RMS over (B,W)
    ang = lax.stop_gradient(theta0 * raw / sc)
    gain = jnp.minimum(angle_clamp / jnp.maximum(ang, 1e-12), 1.0)
    omega = theta0 * A / sc[..., None, None] * gain[..., None, None] if scale == "none" \
        else theta0 * A / sc * gain[..., None, None]
    return expm_fixed(omega, squarings), ang, raw


def transport_chain(st, live, theta0, squarings, angle_clamp, scale="batch", loop_mask=None):
    """Fused SO(n) transport + ordered product: one lax.scan over the L edges whose body
    is checkpointed, so the expm intermediates of an edge are recomputed in the backward
    instead of being stored for all W x L edges at once (measured: peak memory halves).
    st [B,W,L,n,n], live [B,W,L] -> hol [B,W,n,n], instruments.

    D3: with loop_mask the product runs over the LOOP edges only; the frozen design
    multiplied every live step, so a walk that wandered before closing carried its
    tail into the holonomy, and a tail product is not a symmetry of the loop."""
    B, W, L, n, _ = st.shape
    hol0 = jnp.broadcast_to(jnp.eye(n, dtype=st.dtype), (B, W, n, n))
    mask = live if loop_mask is None else loop_mask

    def body(hol, inp):
        st_l, mask_l = inp                                                     # [B,W,n,n], [B,W]
        R, ang, raw = _edge_rotation(st_l, theta0, squarings, angle_clamp, scale)
        hol = jnp.where(mask_l[..., None, None], _mm(R, hol), hol)
        stats = jnp.stack([ang.max(), ang.sum(), (raw < 1e-4).astype(F32).sum(),
                           (ang > angle_clamp).astype(F32).sum(), raw.sum()])
        return hol, stats

    hol, stats = lax.scan(jax.checkpoint(body), hol0, (jnp.moveaxis(st, 2, 0), jnp.moveaxis(mask, 2, 0)))
    cnt = float(B * W * L)
    inst = dict(max_angle=stats[:, 0].max(), mean_angle=stats[:, 1].sum() / cnt,
                floor_frac=stats[:, 2].sum() / cnt, clamp_frac=stats[:, 3].sum() / cnt,
                gen_norm=stats[:, 4].sum() / cnt)
    return hol, inst


def rebase_write(st, loop_mask, visited, weight, values, M, theta0, squarings, angle_clamp, scale):
    """D1. update[b,m] = MEAN over the loop visits (w, j) landing on slot m of
    Q_j values_w Q_j^T, with Q_j the loop prefix (Q at the loop entry is the identity).

    Why: the frozen design wrote ONE matrix into every member slot, so what a slot
    holds does not depend on where it sits on the loop. A closed walk's holonomy is
    based at the walk's entry slot, drawn from the arrival law and unrelated to the
    query, so a query reading its own slot gets the right product only when its slot
    happens to be that base -- about one time in L.

    A second scan recomputes the rotations rather than keeping them: storing R for
    all W x L edges is exactly the [B,W,L,n,n] tensor the fused chain exists to
    avoid, so the port pays one extra expm pass instead of that memory."""
    B, W, L, n, _ = st.shape
    Q0 = jnp.broadcast_to(jnp.eye(n, dtype=st.dtype), (B, W, n, n))
    upd0 = jnp.zeros((B, M, n * n), st.dtype)
    cnt0 = jnp.zeros((B, M, 1), st.dtype)

    def body(carry, inp):
        Q, upd, cnt = carry
        st_l, inloop_l, vis_l = inp
        R, _, _ = _edge_rotation(st_l, theta0, squarings, angle_clamp, scale)
        w_l = inloop_l.astype(st.dtype) * weight
        conj = _mm(_mm(Q, values), jnp.swapaxes(Q, -1, -2)).reshape(B, W, n * n)
        oh = jax.nn.one_hot(jnp.maximum(vis_l, 0), M, dtype=st.dtype) * w_l[..., None]
        upd = upd + jnp.einsum("bwm,bwx->bmx", oh, conj)
        cnt = cnt + oh.sum(1)[..., None]
        Q = jnp.where(inloop_l[..., None, None], _mm(R, Q), Q)
        return (Q, upd, cnt), None

    xs = (jnp.moveaxis(st, 2, 0), jnp.moveaxis(loop_mask, 2, 0), jnp.moveaxis(visited[:, :, :L], 2, 0))
    (_, upd, cnt), _ = lax.scan(jax.checkpoint(body), (Q0, upd0, cnt0), xs)
    return (upd / jnp.maximum(cnt, 1.0)).reshape(B, M, n, n)


# --------------------------------------------------------------------------- dedup (static)
def dedup_static(member, closed):
    """rep [B,W]: lowest-index closed walk of each distinct slot set. Two 16-bit hashes of
    the slot-set bit vector (int32-safe), lexicographic stable sort, first of each run."""
    B, W, M = member.shape
    bits = (member > 0).astype(I32)
    m = np.arange(1, M + 1, dtype=np.int64)                          # constants in int64 at trace time:
    c1 = jnp.asarray((m * 40503) % 65521, dtype=I32)                 # no int32 wrap for any M
    c2 = jnp.asarray((m * m * 4099) % 65519, dtype=I32)
    h1 = (bits * c1).sum(-1) % 65521                                 # < 2^16, sums stay < 2^31 for M < 32768
    h2 = (bits * c2).sum(-1) % 65519
    h1 = jnp.where(closed, h1, 70000 + jnp.arange(W, dtype=I32)[None, :])   # open walks last, distinct
    widx = jnp.broadcast_to(jnp.arange(W, dtype=I32)[None, :], (B, W))
    order = jax.vmap(lambda a, b, c: jnp.lexsort((c, b, a)))(h1, h2, widx)   # by h1, then h2, then index
    s1 = jnp.take_along_axis(h1, order, axis=-1)
    s2 = jnp.take_along_axis(h2, order, axis=-1)
    first_sorted = jnp.concatenate([jnp.ones((B, 1), bool), (s1[:, 1:] != s1[:, :-1]) | (s2[:, 1:] != s2[:, :-1])], axis=1)
    rep = jnp.zeros((B, W), bool).at[jnp.arange(B)[:, None], order].set(first_sorted)
    return rep & closed


# --------------------------------------------------------------------------- interior
def interior(phi, head, q, k, v, C, corr_at, log_gamma=None):
    """corr_at: list of (event_index_in_chunks, corr [B,M,n,n]) with static chunk indices.
    With log_gamma the per-slot forget gate (F1) is applied and the chunks run
    sequentially; without it this is the frozen design's cumulative-sum path."""
    if log_gamma is not None:
        return _interior_decay(phi, head, q, k, v, log_gamma, C, corr_at)
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


def _chunk_terms(Ph, Hd, Q, K, V, cum, cum_end, causal):
    """One chunk of the decay path. Ph/Hd/cum [B,C,M], Q/K/V [B,C,n], cum_end [B,M].
    Returns intra [B,C,n] (reads of this chunk's own writes) and delta [B,M,n,n]
    (this chunk's writes decayed to the chunk end). cum is an inclusive cumulative
    sum of log-gamma inside the chunk, so it is <= 0 and so is cum_t - cum_s."""
    B, C, M = Ph.shape
    n = Q.shape[-1]
    # D[t,s,m] = prod_{r=s+1}^{t} gamma_r[m]; the clip touches only s > t, which the
    # causal mask zeroes (without it the exponential overflows and inf * 0 is NaN)
    D = jnp.exp(jnp.minimum(cum[:, :, None, :] - cum[:, None, :, :], 0.0)) * causal[None, :, :, None]
    G = jnp.einsum("btm,bsm,btsm->bts", Ph, Hd, D)
    Aq = Q @ jnp.swapaxes(K, -1, -2)
    intra = (G * Aq) @ V
    Wd = Hd * jnp.exp(cum_end[:, None, :] - cum)
    X = (V[..., :, None] * K[..., None, :]).reshape(B, C, n * n)
    delta = jnp.einsum("bcm,bcx->bmx", Wd, X).reshape(B, M, n, n)
    return intra, delta


def _interior_decay(phi, head, q, k, v, log_gamma, C, corr_at):
    """F1. S_t[m] = gamma_t[m] (S_{t-1}[m] + corr_t[m]) + head_t[m] v_t k_{t-1}^T.
    Chunks run sequentially because the carry is sequential; the chunk body is
    checkpointed, as in the reference. The chunk loop is unrolled (nC is static)."""
    B, T, M = phi.shape
    n = q.shape[-1]
    T0 = T
    if T % C:
        pad = C - T % C
        phi, head, q, k, v, log_gamma = (jnp.pad(x, ((0, 0), (0, pad), (0, 0)))
                                         for x in (phi, head, q, k, v, log_gamma))
        T = T + pad
    nC = T // C
    Ph, Hd, LG = (x.reshape(B, nC, C, M) for x in (phi, head, log_gamma))
    Q, K, V = (x.reshape(B, nC, C, n) for x in (q, k, v))
    cum = jnp.cumsum(LG, axis=2)
    cum_end = cum[:, :, -1, :]
    causal = jnp.tril(jnp.ones((C, C), F32))
    corr_by_chunk = dict(corr_at)
    body = jax.checkpoint(_chunk_terms)
    S = jnp.zeros((B, M, n, n), F32)
    reads = []
    for c in range(nC):
        if c in corr_by_chunk:
            S = S + corr_by_chunk[c]
        intra, delta = body(Ph[:, c], Hd[:, c], Q[:, c], K[:, c], V[:, c],
                            cum[:, c], cum_end[:, c], causal)
        Pd = Ph[:, c] * jnp.exp(cum[:, c])
        U = jnp.einsum("bcm,bmx->bcx", Pd, S.reshape(B, M, n * n))
        inter = (U.reshape(B, C, n, n) @ Q[:, c][..., None])[..., 0]
        reads.append(intra + inter)
        S = jnp.exp(cum_end[:, c])[..., None, None] * S + delta
    return jnp.concatenate(reads, axis=1)[:, :T0]


# --------------------------------------------------------------------------- layer
def _shift(x):
    return jnp.concatenate([jnp.zeros_like(x[:, :1]), x[:, :-1]], axis=1)


def rotate(x, theta):
    n = x.shape[-1]
    cos, sin = jnp.cos(theta), jnp.sin(theta)
    pair = x.reshape(*x.shape[:-1], n // 2, 2)
    a, b = pair[..., 0], pair[..., 1]
    return jnp.stack([cos * a - sin * b, sin * a + cos * b], axis=-1).reshape(x.shape)


def short_conv(P, h, width):
    """D4: depthwise causal convolution on the routing/value inputs. Without it no
    projection sees the previous token, so a relation whose endpoints are two tokens
    cannot be routed at all. Identity-initialised: on == off at step 0."""
    if not width or "conv_w" not in P:
        return h
    acc = None
    for j in range(width):                      # tap j looks back (width-1-j) tokens
        shift = width - 1 - j
        x = h if shift == 0 else jnp.concatenate([jnp.zeros_like(h[:, :shift]), h[:, :-shift]], axis=1)
        term = x * P["conv_w"][None, None, :, j]
        acc = term if acc is None else acc + term
    return acc


def projections(P, h, cfg=None):
    theta = h @ P["to_theta_w"].T + P["to_theta_b"]
    k = rotate(h @ P["to_k_w"].T, theta)        # q, k stay current-token:
    q = rotate(h @ P["to_q_w"].T, theta)        # the induction-read contract
    c = short_conv(P, h, (cfg or {}).get("short_conv", 0))
    v = c @ P["to_v_w"].T
    phi = jax.nn.softmax(c @ P["to_gate_w"].T + P["to_gate_b"], axis=-1)
    om = jax.nn.softmax(c @ P["to_out_member_w"].T + P["to_out_member_b"], axis=-1)
    im = jax.nn.softmax(c @ P["to_in_member_w"].T + P["to_in_member_b"], axis=-1)
    fwd = jax.nn.sigmoid(c @ P["to_fwd_w"].T + P["to_fwd_b"]) if "to_fwd_w" in P else None
    rev = jax.nn.sigmoid(c @ P["to_rev_w"].T + P["to_rev_b"]) if "to_rev_w" in P else None
    lg = jax.nn.log_sigmoid(h @ P["to_decay_w"].T + P["to_decay_b"]) if "to_decay_w" in P else None
    return dict(k=k, q=q, v=v, phi=phi, im=im, fwd=fwd, rev=rev, log_gamma=lg,
                head=_shift(phi), k_prev=_shift(k), out_step=_shift(om))


def walk_event(cfg, P, o, im, hd, v, kp, in_mean, u, f=None, r=None):
    theta0 = jnp.exp(P["log_theta0"]) if "log_theta0" in P else cfg["theta0"]
    A = pair_mass(o, im, f, r)
    w = sample_slots(A, in_mean, u, cfg["walk_len"], cfg["self_pair_mask"],
                     cfg["walk_dead_end"], cfg["death_c"])
    Hp, Ap, fwd_share = pair_gather(o, im, hd, v, kp, w["pairs"], f, r,
                                    cfg["reverse_transport"] == "transpose")
    st = Hp / jnp.maximum(Ap, EPS)[..., None, None]
    lm = (loop_edges_mask(w["first"], w["length"], w["closed"], cfg["walk_len"])
          if cfg["walk_loop_only"] else None)
    hol, tinst = transport_chain(st, w["live"], theta0, cfg["squarings"], cfg["angle_clamp"],
                                 cfg["transport_scale"], lm)
    tinst = dict(tinst, fwd_ratio=(fwd_share * w["live"].astype(F32)).sum()
                 / jnp.maximum(w["live"].astype(F32).sum(), 1.0),
                 theta0=jnp.asarray(theta0, F32))
    return hol, w, tinst, st, lm


def layer_forward(P, cfg, h, u_events, probe=False):
    """h [B,T,d]; u_events: list (per event) of [B,W,L+1,2] uniforms. Returns out [B,T,d], instruments.

    probe=True adds the events-off read and the write-against-state ratio to the
    instruments. It costs a second pass over the interior, so it belongs at
    evaluation, not in the training step. These are the two numbers the write
    scale was chosen on: the event's perturbation of the read, and the size of
    what it writes measured against the state it is added to."""
    B, T, d = h.shape
    n, M, C, k_event = cfg["n"], cfg["M"], cfg["chunk"], cfg["k_event"]
    p = projections(P, h, cfg)
    events = list(range(k_event, T, k_event))
    corr_at, inst, logprobs = [], [], []
    sw_I = None
    walk_fn = jax.checkpoint(lambda *a: walk_event(cfg, P, *a)) if cfg.get("recompute", True) else (lambda *a: walk_event(cfg, P, *a))
    sl = lambda x, lo, hi: None if x is None else x[:, lo:hi]
    for i, t_e in enumerate(events):
        lo = 0 if i == 0 else events[i - 1]
        dI = p["im"][:, lo:t_e].sum(1)
        sw_I = dI if sw_I is None else sw_I + dI
        in_mean = sw_I / t_e
        hol, w, tinst, st, lm = walk_fn(p["out_step"][:, lo:t_e], p["im"][:, lo:t_e], p["head"][:, lo:t_e],
                                        p["v"][:, lo:t_e], p["k_prev"][:, lo:t_e], in_mean, u_events[i],
                                        sl(p["fwd"], lo, t_e), sl(p["rev"], lo, t_e))
        logprobs.append(w["logprob"])
        rep = dedup_static(w["member"], w["closed"])
        valid = w["closed"] & rep
        chi = jnp.swapaxes(w["member"] * rep[..., None].astype(F32), 1, 2)          # [B,M,W]
        feat = hol @ P["probe"]                                                  # [B,W,n]: holonomy's action on the probe (MELA-260907)
        logits = (feat @ P["walk_q_w"].T) @ jnp.swapaxes(feat @ P["walk_k_w"].T, -1, -2) / n ** 0.5
        mask = valid[:, None, :]
        logits = jnp.where(mask, logits, -jnp.inf)
        logits = jnp.where(mask.any(-1, keepdims=True), logits, 0.0)
        attn = jax.nn.softmax(logits, axis=-1)
        mass_m = chi.sum(-1)                                                   # [B,M]
        if cfg["writeback_rebase"]:
            # D1: each loop slot receives the loop as seen from itself, averaged over
            # visits. The learned constant and the node-attention mixture are added
            # OUTSIDE the conjugation: a constant is not frame-dependent, and the
            # mixture is of other loops' holonomies, based at their own entry slots.
            delta = jnp.einsum("bkj,bjpq->bkpq", attn, hol) - hol
            theta0 = jnp.exp(P["log_theta0"]) if "log_theta0" in P else cfg["theta0"]
            upd = rebase_write(st, lm, w["visited"], valid.astype(F32), hol, M,
                               theta0, cfg["squarings"], cfg["angle_clamp"], cfg["transport_scale"])
            pres = (mass_m > 0).astype(F32)[..., None, None]
            update = upd + pres * P["carry_bias"] + P["gain"] * jnp.einsum("bmk,bkpq->bmpq", chi, delta)
        else:
            vals = hol + P["carry_bias"]
            delta = jnp.einsum("bkj,bjpq->bkpq", attn, vals) - vals
            update = P["gain"] * jnp.einsum("bmk,bkpq->bmpq", chi, delta)
            wsel = valid.astype(F32) / jnp.maximum(valid.sum(-1, keepdims=True), 1).astype(F32)
            vmean = jnp.einsum("bw,bwpq->bpq", wsel, vals)
            update = update + mass_m[..., None, None] * vmean[:, None]
        if "carry_a" in P:
            # D6: a holonomy is orthogonal at any angle, so the write is sqrt(n) in
            # Frobenius norm regardless -- measured at several times the interior
            # state it is added to. alpha = alpha0 * softplus(a + 0.5413), a init 0.
            update = update * (carry_alpha0(cfg) * jax.nn.softplus(P["carry_a"] + _SP1))
        corr_at.append((t_e // C, update))
        cf = w["closed"].astype(F32); denom = jnp.maximum(cf.sum(), 1.0)
        eye = jnp.eye(n, dtype=F32)
        mass_sum = jnp.maximum(mass_m.sum(-1, keepdims=True), EPS)
        inst.append(dict(dead_frac=w["dead"].astype(F32).mean(),
                         no_loop_frac=(~valid.any(-1)).astype(F32).mean(),
                         top_slot_share=(mass_m / mass_sum).max(-1).mean(),
                         len_mean=w["length"].astype(F32).mean(),
                         update_norm=jnp.linalg.norm(update.reshape(B, M, n * n), axis=-1).mean(),
                         carry_alpha=(carry_alpha0(cfg) * jax.nn.softplus(P["carry_a"] + _SP1)
                                      if "carry_a" in P else jnp.asarray(1.0, F32)),
                         K=rep.sum(-1).astype(F32).mean(), closed_frac=cf.mean(),
                         hol_norm=(jnp.linalg.norm(hol.reshape(B, -1, n * n), axis=-1) * cf).sum() / denom,
                         orth_drift=((jnp.abs(_mm(jnp.swapaxes(hol, -1, -2), hol) - eye).reshape(B, -1, n * n).max(-1)) * cf).sum() / denom,   # HIGHEST: the instrument must not add its own bf16 error on TPU
                         occupancy=((w["member"] > 0) & w["closed"][..., None]).any(1).astype(F32).mean(),
                         **tinst))
    read = interior(p["phi"], p["head"], p["q"], p["k_prev"], p["v"], C, corr_at,
                    log_gamma=p["log_gamma"])
    if probe and corr_at:
        off = interior(p["phi"], p["head"], p["q"], p["k_prev"], p["v"], C, [],
                       log_gamma=p["log_gamma"])
        dr = jnp.linalg.norm(read - off, axis=-1) / jnp.maximum(jnp.linalg.norm(off, axis=-1), EPS)
        cn = jnp.stack([jnp.linalg.norm(c.reshape(B, M, n * n), axis=-1) for _, c in corr_at]).mean(0)
        inst.append(dict(dread_med=jnp.median(dr), dread_p95=jnp.quantile(dr.reshape(-1), 0.95),
                         corr_norm=cn.mean()))
    return read @ P["from_read_w"].T, inst, logprobs


def config(d, T, chunk=None, walk_len=32, n_walks=64, theta0=1.0, squarings=3, angle_clamp=12.0,
           recompute=True, **sw):
    """Frozen-design defaults. Every switch added on 2026-09-14 defaults to the old
    behaviour, so config(...) alone is the frozen MELA-260907 design and gate J-1
    still holds; config_main(...) turns the approved set on."""
    n, M = d // 4, d // 2
    cfg = dict(d=d, n=n, M=M, T=T, k_event=max(T // 4, 8), chunk=(chunk or (128 if d >= 1024 else 64)),
               walk_len=walk_len, n_walks=n_walks, theta0=theta0, squarings=squarings, angle_clamp=angle_clamp,
               recompute=recompute,
               # ---- 260913 / 2026-09-14 switches (off = the frozen design) ----
               decay=False, decay_init_bias=4.0, decay_init="uniform", decay_tau=(32.0, 4096.0),
               transport_scale="batch", theta0_learn=False, theta0_target=2.35,
               direction_gates=False, direction_init_bias=4.0, reverse_transport="same",
               walk_dead_end="escape", walk_loop_only=False, self_pair_mask=False, death_c=0.5,
               writeback_rebase=False, carry_gain=False, carry_gain_init="inv_sqrt_n",
               short_conv=0, mu_walk=0.0)
    cfg.update(sw)
    if cfg["writeback_rebase"] and not cfg["walk_loop_only"]:
        raise ValueError("writeback_rebase requires walk_loop_only")
    return cfg


def config_main(d, T, **sw):
    """The 2026-09-14 approved configuration: D1-D6 plus the log-spaced decay init.
    Mirrors mela260913.Config.main() field for field -- gate J-D compares the two."""
    base = dict(decay=True, decay_init="log_spaced", transport_scale="none",
                theta0_learn=True, direction_gates=True, reverse_transport="transpose",
                walk_dead_end="die", walk_loop_only=True, self_pair_mask=True,
                writeback_rebase=True, carry_gain=True, short_conv=2, mu_walk=0.1)
    base.update(sw)
    return config(d, T, **base)


def carry_alpha0(cfg):
    v = cfg["carry_gain_init"]
    if not isinstance(v, str):
        return float(v)
    n = cfg["n"]
    return {"inv_sqrt_n": n ** -0.5, "inv_n": 1.0 / n, "one": 1.0, "two_inv_sqrt_n": 2.0 * n ** -0.5}[v]


def init_params(key, cfg):
    d, n, M = cfg["d"], cfg["n"], cfg["M"]
    ks = jax.random.split(key, 10)
    lin = lambda k, o, i: jax.random.uniform(k, (o, i), F32, -1 / i ** 0.5, 1 / i ** 0.5)
    return dict(to_theta_w=jnp.zeros((n // 2, d), F32), to_theta_b=jnp.zeros((n // 2,), F32),
                to_k_w=lin(ks[0], n, d), to_q_w=lin(ks[1], n, d), to_v_w=lin(ks[2], n, d),
                to_gate_w=lin(ks[3], M, d), to_gate_b=jnp.zeros((M,), F32),
                to_out_member_w=lin(ks[4], M, d), to_out_member_b=jnp.zeros((M,), F32),
                to_in_member_w=lin(ks[5], M, d), to_in_member_b=jnp.zeros((M,), F32),
                from_read_w=lin(ks[6], d, n), probe=jax.random.normal(ks[7], (n,), F32) / n ** 0.5,
                walk_q_w=lin(ks[8], n, n), walk_k_w=lin(ks[9], n, n),   # distinct keys: sharing one made q == k
                gain=jnp.zeros((), F32), carry_bias=jnp.zeros((n, n), F32),
                **_extra_params(cfg))


def _extra_params(cfg):
    """Parameters of the 2026-09-14 switches. Present only when the switch is on, so
    a frozen-design parameter tree is byte-for-byte what it was."""
    d, n, M = cfg["d"], cfg["n"], cfg["M"]
    P = {}
    if cfg["decay"]:
        if cfg["decay_init"] == "log_spaced":
            lo, hi = cfg["decay_tau"]
            tau = jnp.exp(jnp.linspace(jnp.log(lo), jnp.log(hi), M))
            b = jnp.log(tau - 1.0).astype(F32)
        else:
            b = jnp.full((M,), cfg["decay_init_bias"], F32)
        P.update(to_decay_w=jnp.zeros((M, d), F32), to_decay_b=b)
    if cfg["short_conv"]:
        w = jnp.zeros((d, cfg["short_conv"]), F32).at[:, -1].set(1.0)   # identity: current tap 1
        P.update(conv_w=w)
    if cfg["direction_gates"]:
        b_val = cfg["direction_init_bias"]
        # two SEPARATE buffers holding the same value: aliasing them makes the two
        # gates one donated buffer, which XLA rejects, and hides that they are meant
        # to be independent parameters that start equal (equal gates = undirected)
        P.update(to_fwd_w=jnp.zeros((1, d), F32), to_fwd_b=jnp.full((1,), b_val, F32),
                 to_rev_w=jnp.zeros((1, d), F32), to_rev_b=jnp.full((1,), b_val, F32))
    if cfg["theta0_learn"]:
        P.update(log_theta0=jnp.array(jnp.log(cfg["theta0"]), F32))
    if cfg["carry_gain"]:
        P.update(carry_a=jnp.zeros((), F32))
    return P
