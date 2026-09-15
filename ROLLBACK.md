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


## 2026-09-15: the harness port, and one thing it cannot promise

`melatpu/tasks.py` (the LOOPWORD generator and the classifier), `loopword_jax.py`
(the arms, checkpointing and resume) and gate J-H complete the port. J-H passes on
both arms.

Two defects were found and fixed while building it, both in the gate rather than
in the port:

1. The reference `Classifier.forward` does not pass `u` down to the layer, so it
   drew fresh walk randomness on every call and the two sides were never compared
   on the same walks. Fixed by injecting the uniforms.
2. The gate built the JAX side with `mu_walk=0.1` on every arm, while the
   reference forces `mu_walk=0.0` on `legacy`. It was comparing a loss carrying
   the walk term against one without it. Fixed by reading `mu_walk` from the
   reference config. The legacy arm then agreed to 1e-06 on every gradient, up
   from 1.7e-02.

**What J-H cannot promise, and why this is not a defect.** The walk term is a
score-function estimator over sampled slot sequences; the slot is an inverse-CDF
lookup, hence a step function of the routing matrix. Two float32 summation orders
give routing matrices that differ by about 6e-07 relative, about one uniform in
three thousand falls inside that gap, and a single flipped walk shifts its
log-probability by roughly 34 nats. Measured on the main arm: three of four
log-probability tensors agree to 7.6e-06, the fourth has one walk in 512 differing,
and the walk term lands 1.5e-02 apart against a seed-to-seed spread of 5.2e-02.

The gate therefore certifies it in two parts. `sampler_exact()` hands both
samplers the same routing matrix and the same uniforms and requires every pick,
death and closure to be identical -- measured 0 of 3072 picks differing, in both
the escape and the die regimes. The walk term itself is then held to half its own
seed spread. Rolling this back means restoring a 1e-04 test that no correct port
can pass; if the tolerance is ever to be tightened, the way to do it is float64 on
both sides, not a smaller number.

## The tpu/ operations layer

Seven shell scripts, added the same day. They hold no credentials and read none.
`provision.sh`, `run.sh` and `t1_loopword.sh` refuse to act without an explicit
`--yes`, and `teardown.sh` is the only script that stops a bill. Removing the
layer removes no model code: nothing in `melatpu/` imports it.


## 2026-09-15, later: three defects found by porting the language-model harness

1. **Two parameters were one buffer.** `to_fwd_b` and `to_rev_b` were initialised
   from the same array object, so the direction gates shared a buffer. Training
   still worked -- JAX arrays are immutable, so the first gradient separates them
   -- but the donated-buffer path in the compiled training step rejected it, and
   the aliasing hid that these are meant to be two independent parameters that
   merely start equal. Fixed by allocating both.
2. **The events-off arm could not compute a loss.** With `k_event` past `T` no
   walk is ever sampled, so the list of score-function terms is empty and stacking
   it raised. Fixed by returning the task loss alone, which is what the arm means.
3. **Gate J-1 had been broken by the package rename.** It imported `mela260907`,
   which no longer exists, and would have fallen back to the plain `Config()` --
   the current development defaults, not the frozen design. It now resolves the
   renamed package and takes `Config.legacy()`. Restored at output rel 4.21e-07,
   the registered value.

Also: the stratified baseline in the language-model loss sized its one-hot from
`int(stratum.max()) + 1`, which cannot be traced under jit. It now uses the same
static width the task loss uses. Extra empty columns contribute nothing, so this
is exact rather than an approximation.

`layer_forward(..., probe=True)` is new and off by default: it reruns the interior
with the events removed and reports the read perturbation and the size of the
write. It costs a second pass, so it belongs at evaluation. Removing it removes no
training behaviour.


## 2026-09-15, third pass: an adversarial audit of the cloud layer

Eighty-six agents reviewed the operations scripts and the harnesses along five
dimensions before any money could be spent, and each finding was then attacked by three
independent skeptics. Five findings survived; all five were reproduced by hand.

1. **The default batch could not run at all.** The stage-1 fan-out inherited `B=256`.
   The per-event edge tensor is `[B, W, L, n, n]`, which is 2.00 GiB per copy at that
   batch, and XLA's memory analysis puts the compiled step at 1.57 / 3.15 / 6.30 GiB for
   B = 8 / 16 / 32 -- exactly linear at 0.197 GiB per example, so about 50 GiB at 256. A
   v5e chip has 16 GiB. All eight chains would have died at the first step with eight VMs
   already provisioned and billing. The default is now 32, `t1_loopword.sh` passes it
   explicitly rather than inheriting, and `core.hbm_estimate` prints the figure at
   startup with its constant taken from the measurement rather than assumed.
2. **The money switch deleted one resource in eight.** gcloud on Windows is a Python
   program, and Python writes CRLF even into a pipe -- verified on this machine. A
   multi-line capture therefore carries embedded carriage returns, `for` does not split
   on CR, and every name but the last was handed back to gcloud with a CR attached. Both
   deletes then failed and the `||` chain reported "already gone". After a fan-out,
   `teardown.sh --all` would have deleted one VM and left seven billing overnight while
   saying it was done. Fixed by stripping CR at capture and per name, by replacing the
   `||` chain with an explicit check that confirms the resource is actually gone, and by
   exiting non-zero when a delete fails. Verified against a fake gcloud that emits CRLF.
3. **The `slot` arm could not compute a loss.** The same empty-terms defect fixed in the
   language-model loss was still present in the classifier copy.
4. **The language-model harness was never shipped.** `run.sh` tarred a hand-kept file
   list that had gone stale, so stage T2 would have failed on a paid VM. It now ships the
   repository, which is the fix that does not drift.
5. **Resuming a finished arm crashed.** `charlm_jax.py` called the final evaluation with
   batch variables that are bound only inside the training loop, so re-running after a
   preemption died on the first already-complete arm and never reached the rest.

Two further gaps, raised by the operator rather than the audit, are closed in the same
pass: the VM ran as the project default service account (now `MELA_SA`, with preflight
refusing to stay quiet about it), and `env.sh` carried a dead `REMOTE_DIR` holding a
literal backslash.

**And one design gap that no single script could fix.** A preempted TPU Spot VM is
deleted, not stopped, and nothing recreates it. Checkpointing to a bucket was only half
the answer. `watchdog.sh` is the other half: it watches the node, relaunches from the
checkpoint when the process dies, re-provisions when the VM disappears, and is capped by
restart count and wall clock so that a supervisor which can spend money cannot spend it
indefinitely.

Rolling back the audit fixes means restoring a fan-out that cannot run and a teardown
that lies about having stopped the billing. Do not.
