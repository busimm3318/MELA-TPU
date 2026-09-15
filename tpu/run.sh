#!/usr/bin/env bash
# Ship the repo to a live TPU VM and run a command on it, detached, with the log
# streamed back. Usage:
#   tpu/run.sh NAME --yes -- python loopword_jax.py --arm main --steps 25000 --out gs://.../main_s0
set -eu
cd "$(dirname "$0")"; . ./env.sh
NAME="${1:-}"; shift || true
[ -n "$NAME" ] || die "usage: run.sh NAME [--yes] -- COMMAND..."
ASSUME_YES=0
TAIL=1
while [ $# -gt 0 ]; do
  case "$1" in
    --yes) ASSUME_YES=1; shift ;;
    --no-tail) TAIL=0; shift ;;
    --) shift; break ;;
    *) die "unknown flag $1 (the command goes after --)" ;;
  esac
done
[ $# -gt 0 ] || die "nothing to run: put the command after --"
need_gcloud
CMD="$*"
rate="$(awk -v p="$(price_now)" -v c="$CHIPS" 'BEGIN{printf "%.2f", p*c}')"
confirm_spend "compute on '$NAME': $CMD" "$rate"

REPO="$(cd .. && pwd)"
TAR="/tmp/mela-$NAME.tgz"
# ship the whole repository, not a hand-kept list: the list had already gone stale
# (charlm_jax.py was missing, so stage T2 could not have run on a paid VM)
tar --exclude-vcs --exclude='__pycache__' --exclude='*.pyc' --exclude='results' \
    --exclude='*.ckpt' --exclude='*.tgz' -czf "$TAR" -C "$REPO" .

ssh() { gcloud compute tpus tpu-vm ssh "$NAME" --zone "$ZONE" --project "$PROJECT" --command "$1"; }
gcloud compute tpus tpu-vm scp "$TAR" "$NAME:/tmp/mela.tgz" --zone "$ZONE" --project "$PROJECT"
ssh "mkdir -p ~/mela && tar xzf /tmp/mela.tgz -C ~/mela && ls ~/mela"

# wait for the boot install, then launch detached so a dropped ssh does not kill the run
ssh "for i in \$(seq 1 120); do [ -f /tmp/mela-startup-done ] && break; sleep 5; done; \
     test -f /tmp/mela-startup-done || { echo 'startup script never finished'; tail -40 /var/log/mela-startup.log; exit 1; }"
ssh "cd ~/mela && rm -f ~/mela/run.done; . /etc/profile.d/mela.sh 2>/dev/null; \
     nohup sh -c '$CMD; echo \$? > ~/mela/run.done' > ~/mela/run.log 2>&1 & \
     echo \$! > ~/mela/run.pid; sleep 2; cat ~/mela/run.pid"
echo
echo "launched. pid above. streaming the log -- Ctrl-C detaches, it keeps running."
echo "REMEMBER: tpu/teardown.sh $NAME when it finishes, or it bills all night."
[ "$TAIL" = "1" ] || { echo "detached; poll with tpu/watchdog.sh or sync_results.sh"; exit 0; }
ssh "tail -f ~/mela/run.log"
