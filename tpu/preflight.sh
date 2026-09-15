#!/usr/bin/env bash
# Read-only. Spends nothing, changes nothing. Answers the three risks in the
# protocol before any money moves: is the accelerator quota actually usable on a
# free-trial account, when do the credits expire, and does the zone have v5e.
set -u
cd "$(dirname "$0")"; . ./env.sh

fail=0
say()  { printf '  %-34s %s\n' "$1" "$2"; }
bad()  { printf '  %-34s %s\n' "$1" "$2"; fail=1; }

echo "== tools =="
if command -v gcloud >/dev/null 2>&1; then say "gcloud" "$(gcloud version 2>/dev/null | head -1)"
else bad "gcloud" "NOT INSTALLED -- install the Google Cloud SDK"; fi
command -v gsutil >/dev/null 2>&1 && say "gsutil" "present" || bad "gsutil" "missing (ships with the SDK)"
[ "$fail" = "1" ] && { echo; echo "Install the SDK first, then run 'gcloud auth login' yourself."; exit 1; }

echo "== identity =="
acct="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null)"
[ -n "$acct" ] && say "active account" "$acct" || bad "active account" "none -- run: gcloud auth login"
proj="$(gcloud config get-value project 2>/dev/null | grep -v '^(unset)$')"
[ -n "$proj" ] && say "project" "$proj" || bad "project" "unset -- gcloud config set project ID"
[ -n "$proj" ] || exit 1

echo "== billing and credit =="
ba="$(gcloud billing projects describe "$proj" --format='value(billingAccountName)' 2>/dev/null)"
if [ -n "$ba" ]; then
  say "billing account" "$ba"
  en="$(gcloud billing projects describe "$proj" --format='value(billingEnabled)' 2>/dev/null)"
  [ "$en" = "True" ] && say "billing enabled" "yes" || bad "billing enabled" "NO -- TPUs will not create"
else
  bad "billing account" "not linked, or no permission to read it"
fi
echo "  credit balance and expiry are not exposed to the CLI. Check them in the console:"
echo "    https://console.cloud.google.com/billing -> Credits"
echo "  RISK 2: the \$300 trial credit expires 90 days from sign-up. Note the date."

echo "== APIs =="
for api in tpu.googleapis.com compute.googleapis.com storage.googleapis.com; do
  if gcloud services list --enabled --project "$proj" --format='value(config.name)' 2>/dev/null | grep -qx "$api"
  then say "$api" "enabled"
  else bad "$api" "DISABLED -- gcloud services enable $api --project $proj"; fi
done

echo "== accelerator quota (RISK 1) =="
echo "  free-trial accounts have historically been denied GPU/TPU quota. If the numbers"
echo "  below are zero, upgrade to a paid account (the credit still applies) and request quota."
gcloud compute regions describe "${ZONE%-*}" --project "$proj" \
  --format='table(quotas.metric,quotas.limit,quotas.usage)' 2>/dev/null \
  | grep -i -E 'TPU|metric' || echo "  (no TPU quota rows returned for ${ZONE%-*})"
echo "  console view: https://console.cloud.google.com/iam-admin/quotas?project=$proj"

echo "== zone stock (RISK 3) =="
for z in "$ZONE" us-central1-a us-east1-c us-east5-a europe-west4-b asia-east1-a; do
  t="$(gcloud compute tpus accelerator-types list --zone "$z" --project "$proj" \
        --format='value(type)' 2>/dev/null | grep -c '^v5litepod' || true)"
  [ -n "$t" ] && [ "$t" != "0" ] && say "$z" "$t v5litepod types offered" || say "$z" "none offered / no access"
done
echo "  'offered' is not 'in stock'. Stock shows up as a queued resource that stays PENDING."

echo "== storage =="
if [ -n "$BUCKET" ]; then
  gsutil ls -b "$BUCKET" >/dev/null 2>&1 && say "bucket" "$BUCKET ok" \
    || bad "bucket" "$BUCKET missing -- gsutil mb -l ${ZONE%-*} $BUCKET"
else
  bad "bucket" "MELA_BUCKET unset -- checkpoints and the compilation cache need one"
fi

echo "== budget alarm =="
echo "  a budget alert is an account setting, so run it yourself once:"
echo "    gcloud billing budgets create --billing-account=BILLING_ID \\"
echo "      --display-name=mela-tpu --budget-amount=300USD \\"
echo "      --threshold-rule=percent=0.17 --threshold-rule=percent=0.5 --threshold-rule=percent=0.83"

echo "== live resources (anything here is costing money right now) =="
./status.sh 2>/dev/null || true

echo
[ "$fail" = "0" ] && echo "PREFLIGHT OK" || echo "PREFLIGHT INCOMPLETE -- fix the lines marked above"
exit "$fail"
