# MELA-TPU

JAX port of the frozen MELA design (PyTorch reference: https://github.com/busimm3318/MELA, package `mela260907`), built for Cloud TPUs.
Functional core (`melatpu/core.py`), LM + optax step (`melatpu/model.py`), Colab notebooks for the
JAX-vs-PyTorch/XLA measurement (`colab/`).

Gate J-1 (`tests/test_equiv_torch.py`): identical outputs (rel 6e-7) and gradients (1.5e-6) to the
PyTorch reference on identical weights and walk uniforms (needs the reference repository as a sibling directory named `MELA` or `MELA-260907`, or `MELA_REF_DIR` set).

Precision contract: the transport's products need fp32 accuracy. The rotation products (Taylor expm, squarings, the ordered chain) call `jnp.matmul(..., precision=HIGHEST)` so they run at full fp32 operand precision on TPU (6 bf16 passes) while the rest of the model keeps the default; no global precision flag is needed (measured cost in the notebook).

Author: Dongsu Shim. License: Apache-2.0 (see LICENSE, NOTICE). Cite with CITATION.cff.

## j0.5 (2026-09-07): TPU memory/precision/training-path changes (design unchanged; J-1 4.8e-7)
* transport + ordered chain fused into one `lax.scan` over the L edges with a checkpointed body
  (`core.transport_chain`): the expm intermediates of an edge are recomputed in the backward instead
  of being stored for all W x L edges at once (measured on CPU-XLA: peak memory of the event halves);
* `orth_drift` instrument computed at HIGHEST (otherwise it measures its own bf16 rounding on TPU);
* dedup hash constants built in int64 at trace time (no int32 wrap for any M);
* training path (`model.py`): walk uniforms drawn in-graph from one typed key (`fold_in` per
  step / block / event), per-block `jax.checkpoint`, optional data-parallel mesh
  (`make_train_step(cfg, mesh=data_mesh())`), donated parameter/optimizer buffers,
  `enable_compilation_cache(path)` for local or `gs://` persistent caches;
* gate J-2 (`tests/test_train_smoke.py`, batch = 2 x device count): the jitted training step on the device mesh, loss falls on a
  fixed batch, orth_drift < 1e-3.
Not yet done (needs a TPU to measure): scan over blocks/events for compile time, the reversible-chain
custom_vjp, chunk-scan interior, Precision.HIGH trial for the transport, Pallas expm kernel with
VMEM-resident intermediates.

## j0.7 (2026-09-08): three defects found by an audit against the PyTorch reference
* `init_params` drew `walk_q_w`, `walk_k_w` (and the probe) from ONE key, so the two read-out
  matrices were identical at initialisation and the node-attention logits were symmetric;
  they now use distinct keys. No gate caught this (J-1 loads reference weights).
* `make_train_step` used optax's AdamW default weight decay 1e-4; the PyTorch reference uses
  torch.optim.AdamW's default 1e-2. Now an explicit argument defaulting to 1e-2.
* J-2 and the Colab timing cells used a fixed batch that is not divisible by the chip count of a
  multi-chip slice (v5e-8, v6e-8); batch is now a multiple of `jax.device_count()`.

## 2026-09-15: the walk structure fixes, ported and gated (status: NOT frozen)

This repository now carries the 2026-09-14 fixes as well as the frozen design.
`core.config(...)` is the frozen design and is unchanged; `core.config_main(...)`
turns the new set on. Every fix is a switch whose off state is the old behaviour,
and the new parameters exist only when their switch is on, so a frozen-design
parameter tree is what it always was.

What the fixes are, in one line each. **D1**: each loop slot receives the loop as
seen from itself rather than one matrix written into every member slot -- a closed
walk's holonomy is based at the walk's entry slot, which is drawn from the arrival
law and unrelated to the query, so the old write capped an exact non-abelian
answer at about one time in L. **D2**: per-token forward and reverse gates on the
pair mass, both at one on initialisation so the graph starts undirected exactly as
before, and a reversed traversal now transports the transposed rotation, which
makes an edge and its reverse exact transposes and gives a backtrack the identity
holonomy. **D3**: a dead end kills the walk instead of lifting the non-backtracking
ban and counting the backtrack as a closure, the chain multiplies only the loop
part, and death is judged against a threshold relative to one over the slot count.
**D4**: a short depthwise causal convolution on the routing and value inputs,
identity-initialised; without it no projection sees the previous token, so a
relation whose endpoints are two tokens cannot be routed at all. **D5**: the angle
normalisation is withdrawn -- normalising by a root-mean-square over other walks
pins the angle scale, and since a single-plane rotation can only carry a
representation whose angles are fixed, that made an exact group representation
impossible; the angle is now a learned scale times the edge's own generator norm.
**D6**: one global learned scale on the write-back, because a holonomy is
orthogonal at any angle and the write measured several times the interior state it
is added to. **F1** and **F2**, absent from the earlier port, are here too: the
per-slot forget gate with a log-spaced initialisation, and the score-function term
that is the only thing giving the routing distributions a gradient.

Gates (CPU, d=64, T=256):

```
tests/test_equiv_torch.py    J-1  frozen path == mela260907        output rel 4.2e-07
tests/test_equiv_dfixes.py   J-D  main config == mela260915        output rel 1.6e-06
                                  twelve instruments to 1.3e-06, gradients to 2.4e-05
tests/test_train_smoke.py    J-2  jitted step, loss falls, orth drift 4.3e-06
```

J-D is the load-bearing one: it says the JAX and PyTorch implementations are the
same function, so a verdict reached on either applies to both.

**Status: unfrozen.** The frozen design stays frozen (`core.config()`, and
`mela260907` in the reference repository). The current version is the
in-development design and is meant to move; the plan of record is to re-harden
the kernel and freeze again after the LOOPWORD verification, so that what gets
frozen is a mechanism that has been shown to do something. Unfrozen does not mean
the engineering contract lapses -- static shapes, no host synchronisation inside a
step and the full-graph trace are gated on every commit.

**What is not known.** Whether the walk works. The fixes are a corrected
mechanism, not a demonstrated one. `ROLLBACK.md` records what each fix asserts,
what would falsify it, and how to undo it.
