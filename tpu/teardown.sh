#!/usr/bin/env bash
# The money switch. Deletes the queued resource AND its node. Run it the moment a
# run finishes -- a TPU VM bills for existing, not for computing.
# Usage: tpu/teardown.sh NAME [--all]
set -u
cd "$(dirname "$0")"; . ./env.sh
need_gcloud
if [ "${1:-}" = "--all" ]; then
  names="$(gcloud compute tpus queued-resources list --zone "$ZONE" --project "$PROJECT" --format='value(name)' 2>/dev/null | tr -d '\r')"
  echo "deleting EVERY queued resource in $ZONE:"; echo "${names:-  (none)}"
else
  names="${1:-}"
  [ -n "$names" ] || die "usage: teardown.sh NAME [--all]"
fi
fail=0
for n in $names; do
  n="$(printf %s "$n" | tr -d '\r')"      # gcloud on Windows writes CRLF even into a pipe,
  [ -n "$n" ] || continue                 # and CR is not in IFS, so every name but the last
                                          # would otherwise be passed back to gcloud with a CR
  echo "-- $n"
  if gcloud compute tpus queued-resources delete "$n" --zone "$ZONE" --project "$PROJECT" --quiet --force 2>/dev/null; then
    echo "   deleted"
  elif gcloud compute tpus tpu-vm delete "$n" --zone "$ZONE" --project "$PROJECT" --quiet 2>/dev/null; then
    echo "   deleted (node)"
  elif ! gcloud compute tpus queued-resources describe "$n" --zone "$ZONE" --project "$PROJECT" >/dev/null 2>&1; then
    echo "   already gone"
  else
    echo "   DELETE FAILED for $n -- IT IS STILL BILLING" >&2
    fail=1
  fi
done
echo; ./status.sh
exit "$fail"
