# MELA-TPU

JAX port of the frozen MELA design (MELA-260907), built for Cloud TPUs (TPU Research Cloud).
Functional core (`melatpu/core.py`), LM + optax step (`melatpu/model.py`), Colab notebooks for the
JAX-vs-PyTorch/XLA measurement (`colab/`).

Gate J-1 (`tests/test_equiv_torch.py`): identical outputs (rel 6e-7) and gradients (1.5e-6) to the
PyTorch reference on identical weights and walk uniforms (needs ../MELA-260907).

Precision contract: the transport's products need fp32 accuracy -- run with
the rotation products (Taylor expm, squarings, the ordered chain) call `jnp.matmul(..., precision=HIGHEST)` so they run at full fp32 operand precision on TPU (6 bf16 passes) while the rest of the model keeps the default; no global precision flag is needed (measured cost in the notebook).
