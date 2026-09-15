#!/usr/bin/env bash
# Pull logs and checkpoints off a VM (or straight out of the bucket) into ./results.
# Usage: tpu/sync_results.sh NAME
set -eu
cd "$(dirname "$0")"; . ./env.sh
NAME="${1:-}"; [ -n "$NAME" ] || die "usage: sync_results.sh NAME"
need_gcloud
OUT="../results/$NAME"; mkdir -p "$OUT"
gcloud compute tpus tpu-vm scp "$NAME:~/mela/run.log" "$OUT/run.log" \
  --zone "$ZONE" --project "$PROJECT" 2>/dev/null || echo "(no run.log on the VM)"
gcloud compute tpus tpu-vm scp --recurse "$NAME:~/mela/results" "$OUT/" \
  --zone "$ZONE" --project "$PROJECT" 2>/dev/null || true
[ -n "$BUCKET" ] && gsutil -m cp -r "$BUCKET/$NAME" "$OUT/" 2>/dev/null || true
echo "into $OUT:"; ls -la "$OUT"
