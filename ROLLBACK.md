# Rollback log — the 2026-09-15 port of the walk structure fixes

This repository was a port of the frozen design. It now also carries the
2026-09-14 fixes, ported from the PyTorch reference and gated against it.
Written **before** the experiment that judges those fixes, so that a
disappointing result cannot be reinterpreted afterwards.

## The cheap fact first

**Every fix is a switch whose off state is the frozen behaviour**, so a rollback
is normally a configuration change, not a revert:

    core.config(d, T)        # the frozen design -- unchanged, gate J-1 holds
    core.config_main(d, T)   # everything below ON; the only configuration at risk

A frozen-design parameter tree is also unchanged: the new parameters exist only
when their switch is on (`_extra_params`).

## What was ported, and the switch that turns it off

| fix | what it changes | switch (off = frozen) |
|---|---|---|
| **D1** | each loop slot receives the loop as seen from itself (the prefix taken from the loop entry), averaged over visits, instead of one matrix into every member slot | `writeback_rebase=False` |
| **D2** | per-token forward and reverse gates on the pair mass, both at one on initialisation; a reversed traversal transports the transposed rotation | `direction_gates=False`, `reverse_transport="same"` |
| **D3** | a dead end kills the walk instead of lifting the ban and counting the backtrack as a closure; the chain multiplies only the loop part; a self-pair can never close; death is judged against a threshold relative to one over the slot count | `walk_dead_end="escape"`, `walk_loop_only=False`, `self_pair_mask=False` |
| **D4** | a short depthwise causal convolution on the routing and value inputs, identity-initialised | `short_conv=0` |
| **D5** | the angle normalisation is withdrawn: the angle is a learned scale times the edge's own generator norm | `transport_scale="batch"`, `theta0_learn=False` |
| **D6** | a single global learned scale on the write-back | `carry_gain=False` |
| **F1** | the per-slot forget gate, with a log-spaced initialisation | `decay=False` |
| **F2** | the score-function term on the walk's routing log-probability | `mu_walk=0.0` |

## How the port is verified

A port cannot be verified against its own intentions. It is verified against the
reference, which is what this repository has always done — the three defects the
2026-09-08 audit found were all "differs from the reference".

    tests/test_equiv_torch.py    J-1: the frozen path equals mela260907
    tests/test_equiv_dfixes.py   J-D: the main configuration equals mela260915
    tests/test_train_smoke.py    J-2: the jitted training step runs, loss falls,
                                 the holonomy stays a rotation

Recorded at the time of writing (CPU, d=64, T=256):

    GATE_J1  output rel 4.21e-07 | gradient err-ratio 1.61e-06
    GATE_JD  output rel 1.64e-06 | walk log-probability rel 6.22e-08
             twelve instruments match to 1.3e-06 or better
             gradients to the routing projection, the direction gate, the write
             scale and the decay gate: 1.4e-06 to 2.4e-05
    GATE_J2  loss falls over six steps, maximum orthogonality drift 4.27e-06

J-D is the one that matters for a rollback decision: it says the JAX and PyTorch
implementations are the same function, so a verdict reached on either applies to
both, and a rollback on one is a rollback on both.

## What is measured and what is not

Measured (initialisation only, no training): the write scale's exponent is
width-invariant at one over the square root of the read rank; the unit scale,
which is what the model has without D6, moves the initial loss from 0.30 to 2.25;
the death constant is flat across its whole range once routing has separated.

**Not measured: whether the walk works.** That is what the LOOPWORD verification
is for, and until it runs the fixes are a corrected mechanism, not a better one.

## How to undo

Partial, and preferred:

    core.config_main(d, T, writeback_rebase=False, walk_loop_only=False)  # D1 out
    core.config_main(d, T, direction_gates=False, reverse_transport="same")
    core.config_main(d, T, walk_dead_end="escape", self_pair_mask=False)
    core.config_main(d, T, short_conv=0)
    core.config_main(d, T, transport_scale="batch", theta0_learn=False)
    core.config_main(d, T, carry_gain=False)
    core.config_main(d, T, decay=False)
    core.config_main(d, T, mu_walk=0.0)

Full revert of this repository to the pre-port state:

    git log --oneline            # find the commit before "port of the walk structure fixes"
    git revert --no-commit <that range>

The interface changes a full revert also takes with it: `layer_forward` returns
three values instead of two (the third is the walks' log-probabilities, which the
score-function term needs), `pair_gather` returns three instead of two, and
`sample_slots` returns two extra keys.

## The freeze

The frozen design stays frozen: `mela260907` in the reference repository, and
`core.config()` here. The current version is **not** frozen — it is the
in-development design, and it is meant to move. The plan of record is to
re-harden the kernel and freeze again **after** the LOOPWORD verification, so
that what gets frozen is a mechanism that has been shown to do something.

Unfrozen means the design may change. It does not mean the engineering contract
lapses: static shapes, no host synchronisation inside a step, and the
full-graph trace are gated on every commit, here and in the reference.
