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
* gate J-2 (`tests/test_train_smoke.py`): the jitted training step on the device mesh, loss falls on a
  fixed batch, orth_drift < 1e-3.
Not yet done (needs a TPU to measure): scan over blocks/events for compile time, the reversible-chain
custom_vjp, chunk-scan interior, Precision.HIGH trial for the transport, Pallas expm kernel with
VMEM-resident intermediates.
