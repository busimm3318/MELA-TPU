#!/usr/bin/env bash
# The money switch. Deletes the queued resource AND its node. Run it the moment a
# run finishes -- a TPU VM bills for existing, not for computing.
# Usage: tpu/teardown.sh NAME [--all]
set -u
cd "$(dirname "$0")"; . ./env.sh
need_gcloud
if [ "${1:-}" = "--all" ]; then
  names="$(gcloud compute tpus queued-resources list --zone "$ZONE" --project "$PROJECT" \
           --format='value(name)' 2>/dev/null)"
  echo "deleting EVERY queued resource in $ZONE:"; echo "${names:-  (none)}"
else
  names="${1:-}"
  [ -n "$names" ] || die "usage: teardown.sh NAME [--all]"
fi
for n in $names; do
  echo "-- $n"
  gcloud compute tpus queued-resources delete "$n" --zone "$ZONE" --project "$PROJECT" --quiet --force \
    || gcloud compute tpus tpu-vm delete "$n" --zone "$ZONE" --project "$PROJECT" --quiet \
    || echo "   (already gone)"
done
echo; ./status.sh
