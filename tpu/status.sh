#!/usr/bin/env bash
# What exists right now, and what it costs per hour. Read-only.
set -u
cd "$(dirname "$0")"; . ./env.sh
need_gcloud
echo "  zone $ZONE, project $PROJECT"
q="$(gcloud compute tpus queued-resources list --zone "$ZONE" --project "$PROJECT" \
     --format='table(name,state.state)' 2>/dev/null)"
n="$(gcloud compute tpus tpu-vm list --zone "$ZONE" --project "$PROJECT" \
     --format='table(name,acceleratorType,state)' 2>/dev/null)"
echo "  -- queued resources --"; echo "${q:-  (none)}" | sed 's/^/  /'
echo "  -- tpu vms --";          echo "${n:-  (none)}" | sed 's/^/  /'
live="$(gcloud compute tpus tpu-vm list --zone "$ZONE" --project "$PROJECT" \
        --format='value(name)' 2>/dev/null | wc -l | tr -d ' ')"
if [ "${live:-0}" -gt 0 ]; then
  echo "  $live VM(s) alive at ~\$$(price_now) per chip-hour each. Delete with tpu/teardown.sh NAME."
else
  echo "  nothing running, nothing accruing."
fi
