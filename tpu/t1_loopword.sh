#!/usr/bin/env bash
# Stage T1 of the protocol: the LOOPWORD structure diagnosis, eight independent
# chains on eight ONE-chip VMs. Billing is per chip-hour, so eight 1-chip VMs cost
# exactly what one 8-chip slice costs -- but a preemption takes one chain instead
# of all eight, and a 1-chip slice is far easier to get.
#
#   tpu/t1_loopword.sh                 # print the plan and the bill, touch nothing
#   tpu/t1_loopword.sh --yes           # actually provision and launch
set -eu
cd "$(dirname "$0")"; . ./env.sh
ARMS="${MELA_T1_ARMS:-main oracle-dir legacy-walk dead-hol}"
SEEDS="${MELA_T1_SEEDS:-0 1}"
STEPS="${MELA_T1_STEPS:-25000}"
SEC_PER_STEP="${MELA_SEC_PER_STEP:-}"        # fill in from T0; without it no estimate is honest
GO=0; for a in "$@"; do [ "$a" = "--yes" ] && GO=1; done

echo "T1 plan: arms [$ARMS] x seeds [$SEEDS] = chains, $STEPS steps each, $ACCEL per chain"
n=0; for arm in $ARMS; do for s in $SEEDS; do n=$((n+1)); done; done
echo "chains: $n   rate: \$$(price_now) per chip-hour   spot: $SPOT"
if [ -n "$SEC_PER_STEP" ]; then
  awk -v n="$n" -v st="$STEPS" -v sp="$SEC_PER_STEP" -v p="$(price_now)" 'BEGIN{
    h = st*sp/3600; printf "  %.1f h per chain, %.0f chip-hours total, about $%.0f, wall clock %.1f h if all run at once\n",
    h, n*h, n*h*p, h }'
else
  echo "  no cost estimate: MELA_SEC_PER_STEP is unset. Run T0 first and measure it."
  echo "  (T0: tpu/provision.sh mela-t0 --yes ; tpu/run.sh mela-t0 --yes -- python tests/test_equiv_dfixes.py)"
fi
echo "  stop rule from the protocol: if at $STEPS every arm is at or below the order-blind"
echo "  ceiling in the (N=8, L=3) cell, stop and fall back to the registered ladder."
[ "$GO" = "1" ] || { echo; echo "dry run. Re-run with --yes to spend."; exit 0; }
[ -n "$BUCKET" ] || die "MELA_BUCKET unset: a preempted spot chain with no checkpoint is money burnt"

for arm in $ARMS; do
  for s in $SEEDS; do
    name="mela-t1-$(echo "$arm" | tr -d '-')-s$s"
    echo "== $name"
    ./provision.sh "$name" --yes
    ./run.sh "$name" --yes -- python loopword_jax.py --arm "$arm" --seed "$s" \
      --steps "$STEPS" --out "$BUCKET/$name" &
  done
done
wait
echo "all chains returned. PULL RESULTS THEN TEAR DOWN:"
for arm in $ARMS; do for s in $SEEDS; do
  n="mela-t1-$(echo "$arm" | tr -d '-')-s$s"; echo "  tpu/sync_results.sh $n && tpu/teardown.sh $n"
done; done
