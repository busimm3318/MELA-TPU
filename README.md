# MELA-TPU

JAX port of the frozen MELA design (MELA-260906), built for Cloud TPUs (TPU Research Cloud).
Functional core (`melatpu/core.py`), LM + optax step (`melatpu/model.py`), Colab notebooks for the
JAX-vs-PyTorch/XLA measurement (`colab/`).

Gate J-1 (`tests/test_equiv_torch.py`): identical outputs (rel 6e-7) and gradients (1.5e-6) to the
PyTorch reference on identical weights and walk uniforms (needs ../MELA-260906).

Precision contract: the transport's products need fp32 accuracy -- run with
`jax.config.update("jax_default_matmul_precision", "highest")` on TPU (measured cost in the notebook).
