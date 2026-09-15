#!/usr/bin/env bash
# Runs ON the TPU VM at boot, as root, via --metadata-from-file=startup-script.
# Keep it short: every second here is billed.
set -eux
exec > >(tee -a /var/log/mela-startup.log) 2>&1

BUCKET="$(curl -fsH 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/mela-bucket || true)"

pip install --upgrade pip
# JAX must match the TPU runtime, so take the libtpu release index rather than PyPI alone.
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install --upgrade optax numpy

# lever 4: never compile the same kernel twice, across restarts and across VMs
if [ -n "$BUCKET" ]; then
  echo "export MELA_CACHE=$BUCKET/jaxcache" >> /etc/profile.d/mela.sh
  echo "export MELA_BUCKET=$BUCKET"          >> /etc/profile.d/mela.sh
fi

python - <<'PY' || true
import jax
print("jax", jax.__version__, "devices", jax.devices())
PY

touch /tmp/mela-startup-done
