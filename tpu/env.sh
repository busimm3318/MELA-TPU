# Settings every tpu/ script reads. Override any of them in the environment.
# Nothing here spends money; provision.sh is the only script that can.

PROJECT="${MELA_PROJECT:-}"                  # gcloud project id; preflight tells you if unset
ZONE="${MELA_ZONE:-us-central1-a}"           # cheapest v5e zone with stock; preflight lists alternates
ACCEL="${MELA_ACCEL:-v5litepod-1}"           # ONE chip. Billing is per chip-hour, so 8x1 == 1x8 in
                                             # money but 8x1 loses one chain, not eight, to preemption.
RUNTIME="${MELA_RUNTIME:-v2-alpha-tpuv5-lite}"
SPOT="${MELA_SPOT:-1}"                       # 1 = --spot (about 55-70% off, preemptible)
BUCKET="${MELA_BUCKET:-}"                    # gs://... for checkpoints, cache and results
REMOTE_DIR="${MELA_REMOTE_DIR:-/home/\$USER/mela}"

# List price per chip-hour, USD. Confirm against the pricing page before trusting any
# estimate a script prints -- the console showed "Pricing information not available".
PRICE_ONDEMAND="${MELA_PRICE_ONDEMAND:-1.20}"
PRICE_SPOT="${MELA_PRICE_SPOT:-0.54}"

CHIPS="${ACCEL##*-}"                         # v5litepod-8 -> 8
[ "$CHIPS" = "$ACCEL" ] && CHIPS=1

price_now() { [ "$SPOT" = "1" ] && echo "$PRICE_SPOT" || echo "$PRICE_ONDEMAND"; }

die() { echo "ERROR: $*" >&2; exit 1; }

need_gcloud() {
  command -v gcloud >/dev/null 2>&1 || die "gcloud not found. Install the Google Cloud SDK, then run 'gcloud auth login'."
  [ -n "$PROJECT" ] || PROJECT="$(gcloud config get-value project 2>/dev/null | grep -v '^(unset)$')"
  [ -n "$PROJECT" ] || die "no project set. Run: gcloud config set project YOUR_PROJECT_ID"
}

# Every spending action funnels through this. It never reads a password or a token.
confirm_spend() {
  local what="$1" rate="$2"
  echo
  echo "  ABOUT TO SPEND: $what"
  echo "  rate: \$$rate per hour while the resource exists (billed on existence, not on use)"
  echo "  stop it with: tpu/teardown.sh <name>"
  echo
  [ "$ASSUME_YES" = "1" ] || die "refusing to spend without --yes (this is the approval gate, do not remove it)"
}
