# Driving Cloud TPU from a shell

Seven scripts. One of them can spend money, one stops the spending, the rest are
read-only or run on a machine you already pay for.

| script | spends | what it does |
|---|---|---|
| `env.sh` | no | every setting, in one place. Sourced by the others |
| `preflight.sh` | no | checks tools, identity, billing, APIs, quota, zone stock, bucket |
| `status.sh` | no | what exists right now and what it costs per hour |
| `provision.sh` | **YES** | creates one queued TPU resource. Refuses without `--yes` |
| `startup.sh` | — | runs on the VM at boot: installs JAX, points the compile cache at the bucket |
| `run.sh` | **yes** | ships the repo to a live VM, launches detached, streams the log |
| `sync_results.sh` | no | pulls logs and checkpoints into `results/` |
| `teardown.sh` | no | deletes the resource. **A TPU VM bills for existing, not for computing** |
| `watchdog.sh` | **yes** | keeps one chain alive across preemptions; capped restarts and a wall-clock stop |
| `t1_loopword.sh` | **yes** | the eight-chain stage-1 fan-out; dry-runs unless given `--yes` |

## What only you can do

1. **Install the Google Cloud SDK**, then `gcloud auth login`. It opens a browser.
   No script here will ever read a password, a token or a key.
2. **Set the project and the bucket**:
   ```
   gcloud config set project YOUR_PROJECT
   gsutil mb -l us-central1 gs://your-mela-bucket
   export MELA_BUCKET=gs://your-mela-bucket
   ```
3. **Make a narrow service account** and export `MELA_SA`. Without it the VM runs as
   the project default account, which can reach the rest of the project.
4. **Create the budget alert** — changing a billing setting is yours to do.
   `preflight.sh` prints the exact command.
5. **Answer the three risks** that `preflight.sh` surfaces but cannot decide:
   free-trial accelerator quota, the 90-day credit expiry date, and v5e stock
   in the zone.

## The order

```
tpu/preflight.sh                     # free
tpu/provision.sh mela-t0 --yes       # one chip, spot, a few dollars
tpu/run.sh mela-t0 --yes -- python tests/test_equiv_dfixes.py
tpu/run.sh mela-t0 --yes -- python loopword_jax.py --arm main --steps 200 --B 64
tpu/sync_results.sh mela-t0
tpu/teardown.sh mela-t0              # do not skip this line
```

T0 exists to produce one number, seconds per step, which is what turns the T1
estimate from a guess into arithmetic. Put it in `MELA_SEC_PER_STEP` and
`t1_loopword.sh` will print the bill before it spends anything.

## Why one chip at a time

Cloud TPU bills per chip-hour for as long as the VM exists. Eight one-chip VMs
and one eight-chip slice cost the same. The eight separate VMs are preempted
independently, so a spot preemption costs one chain rather than all eight, and a
one-chip slice is much easier to get when the zone is tight. Slice size is a
scheduling decision here, not a cost decision.

## Preemption deletes the VM

A preempted TPU Spot VM is deleted, not stopped. Google's own documentation is blunt
about it: Spot TPUs cannot be restarted and must be recreated. Nothing restarts the
workload for you. A checkpoint in the bucket is therefore necessary and not sufficient
-- something has to notice the VM is gone, ask for another, and relaunch from the
checkpoint. That is `watchdog.sh`. Without it an overnight spot run stops at the first
preemption and the fan-out returns half its chains.

Because the watchdog can provision on its own, it is capped: `MELA_MAX_RESTARTS`
(default 5), `MELA_MAX_HOURS` (default 12, after which it tears down), `MELA_POLL`.

## Batch size is a hardware constraint here, not a taste

The per-event edge tensor is `[B, W, L, n, n]`. At the stage-1 shape that is 2.00 GiB
per copy at `B=256`, and the compiled training step holds about twenty-five copies once
the backward pass keeps its residuals. Measured with XLA's own memory analysis:

| batch | peak |
|---|---|
| 8 | 1.57 GiB |
| 16 | 3.15 GiB |
| 32 | 6.30 GiB |
| 256 | about 50 GiB |

One v5e chip has 16 GiB. `B=32` is the working default; the harness prints its own
estimate at startup so a batch that cannot run is caught before a VM is provisioned.

## Why the compilation cache matters

The chunk loop is a Python unroll, so compilation is long. A spot VM restarts
often. `startup.sh` points `MELA_CACHE` at the bucket and `loopword_jax.py
--cache` uses it, so a restart pays for the compile once rather than every time.
