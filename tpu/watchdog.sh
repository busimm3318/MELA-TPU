#!/usr/bin/env bash
# Keep ONE chain alive across spot preemptions.
#
# A preempted TPU Spot VM is DELETED, not stopped: "You can't restart TPU Spot VMs,
# and you must recreate them after preemption." Nothing in Google Cloud restarts the
# workload. A checkpoint in the bucket is therefore necessary but not sufficient --
# something has to notice the VM is gone, ask for another one, and relaunch. That is
# this script, and without it an overnight spot run silently stops at the first
# preemption and the eight-chain fan-out comes back with four results.
#
# Usage:
#   tpu/watchdog.sh NAME --yes -- python loopword_jax.py --arm main --out gs://b/name
#
# Caps, because this script can spend money on its own:
#   MELA_MAX_RESTARTS   how many times it may re-provision            (default 5)
#   MELA_MAX_HOURS      hard wall-clock stop, then it tears down      (default 12)
#   MELA_POLL           seconds between checks                        (default 120)
set -u
cd "$(dirname "$0")"; . ./env.sh

NAME="${1:-}"; shift || true
[ -n "$NAME" ] || die "usage: watchdog.sh NAME [--yes] -- COMMAND..."
ASSUME_YES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --yes) ASSUME_YES=1; shift ;;
    --) shift; break ;;
    *) die "unknown flag $1 (the command goes after --)" ;;
  esac
done
[ $# -gt 0 ] || die "nothing to run: put the command after --"
need_gcloud
CMD="$*"
MAX_RESTARTS="${MELA_MAX_RESTARTS:-5}"
MAX_HOURS="${MELA_MAX_HOURS:-12}"
POLL="${MELA_POLL:-120}"
[ -n "$BUCKET" ] || die "MELA_BUCKET unset: a relaunch with no checkpoint restarts from step 0"

rate="$(awk -v p="$(price_now)" -v c="$CHIPS" 'BEGIN{printf "%.2f", p*c}')"
confirm_spend "supervised run '$NAME' for up to $MAX_HOURS h with up to $MAX_RESTARTS restarts: $CMD" "$rate"
echo "  worst case about \$$(awk -v r="$rate" -v h="$MAX_HOURS" 'BEGIN{printf "%.0f", r*h}') before the wall-clock stop"

ssh_q() { gcloud compute tpus tpu-vm ssh "$NAME" --zone "$ZONE" --project "$PROJECT" \
            --command "$1" 2>/dev/null | tr -d '\r'; }
state()  { gcloud compute tpus tpu-vm describe "$NAME" --zone "$ZONE" --project "$PROJECT" \
            --format='value(state)' 2>/dev/null | tr -d '\r'; }

restarts=0
deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
launched=0

while :; do
  now=$(date +%s)
  if [ "$now" -ge "$deadline" ]; then
    echo "WATCHDOG: wall-clock cap of ${MAX_HOURS}h reached -- tearing down so it stops billing"
    ./teardown.sh "$NAME"
    exit 2
  fi

  st="$(state)"
  case "$st" in
    READY|ACTIVE)
      if [ "$launched" = "0" ]; then
        echo "WATCHDOG: node $st, launching (restart $restarts)"
        ./run.sh "$NAME" --yes --no-tail -- $CMD && launched=1
      else
        done_code="$(ssh_q 'cat ~/mela/run.done 2>/dev/null')"
        if [ -n "$done_code" ]; then
          echo "WATCHDOG: run finished with exit code $done_code"
          ./sync_results.sh "$NAME" || true
          ./teardown.sh "$NAME"
          exit "$done_code"
        fi
        alive="$(ssh_q 'kill -0 $(cat ~/mela/run.pid 2>/dev/null) 2>/dev/null && echo yes')"
        if [ "$alive" != "yes" ]; then
          echo "WATCHDOG: process gone with no completion marker -- relaunching from the checkpoint"
          launched=0
        fi
      fi
      ;;
    "")
      # No node. Either the queued resource is still waiting for capacity (free), or the
      # spot VM was preempted and deleted (also free, but nothing will come back on its
      # own unless the queued resource is still holding a request).
      qs="$(gcloud compute tpus queued-resources describe "$NAME" --zone "$ZONE" \
             --project "$PROJECT" --format='value(state.state)' 2>/dev/null | tr -d '\r')"
      if [ -n "$qs" ] && [ "$qs" != "FAILED" ] && [ "$qs" != "SUSPENDED" ]; then
        echo "WATCHDOG: queued resource is $qs, waiting for capacity (this costs nothing)"
      else
        if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
          echo "WATCHDOG: hit the restart cap of $MAX_RESTARTS -- stopping rather than spending more"
          ./teardown.sh "$NAME" || true
          exit 3
        fi
        restarts=$((restarts + 1)); launched=0
        echo "WATCHDOG: no node and queued resource is '${qs:-gone}' -- re-provisioning (restart $restarts/$MAX_RESTARTS)"
        ./teardown.sh "$NAME" >/dev/null 2>&1 || true
        ./provision.sh "$NAME" --yes || echo "WATCHDOG: provision failed, will retry next poll"
      fi
      ;;
    *)
      echo "WATCHDOG: node state '$st', waiting"
      ;;
  esac
  sleep "$POLL"
done
