#!/usr/bin/env bash
# Create ONE queued TPU resource. This is the only script that starts a bill.
# Usage: tpu/provision.sh NAME [--yes] [--on-demand]
set -eu
cd "$(dirname "$0")"; . ./env.sh
NAME="${1:-}"; shift || true
[ -n "$NAME" ] || die "usage: provision.sh NAME [--yes] [--on-demand]"
ASSUME_YES=0
for a in "$@"; do
  case "$a" in
    --yes) ASSUME_YES=1 ;;
    --on-demand) SPOT=0 ;;
    *) die "unknown flag $a" ;;
  esac
done
need_gcloud
[ -n "$BUCKET" ] || die "MELA_BUCKET unset: without it a preempted run loses everything"

rate="$(awk -v p="$(price_now)" -v c="$CHIPS" 'BEGIN{printf "%.2f", p*c}')"
kind="$([ "$SPOT" = "1" ] && echo 'SPOT (preemptible, cheap)' || echo 'ON-DEMAND (not preemptible, dear)')"
confirm_spend "$ACCEL in $ZONE as '$NAME', $kind" "$rate"

set -x
gcloud compute tpus queued-resources create "$NAME" \
  --node-id="$NAME" --zone="$ZONE" --project="$PROJECT" \
  --accelerator-type="$ACCEL" --runtime-version="$RUNTIME" \
  --metadata-from-file=startup-script=./startup.sh \
  --metadata="mela-bucket=$BUCKET" \
  $([ "$SPOT" = "1" ] && echo --spot)
set +x
echo
echo "queued. It may sit in PENDING until the zone has stock -- that part is free."
echo "watch:   tpu/status.sh"
echo "when ACTIVE: tpu/run.sh $NAME -- python loopword_jax.py --arm main --steps 100"
echo "STOP THE BILL: tpu/teardown.sh $NAME"
