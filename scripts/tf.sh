#!/usr/bin/env bash
# Run terraform against the yapper k3s cluster WITHOUT hand-managing the SSH tunnel.
#
# The k3s API cert is only valid for 127.0.0.1, so the kubernetes/kustomization providers reach
# it through an `ssh -L 6443` local-forward to the EC2 box (hubstream). This wrapper brings that
# tunnel up if it isn't already, runs `terraform "$@"` in deploy/terraform, then tears down ONLY
# the tunnel it opened (a pre-existing one — e.g. for kubectl — is left running).
#
#   bash scripts/tf.sh plan
#   bash scripts/tf.sh apply
#   TF_VAR_image_tag=$(git rev-parse --short HEAD) bash scripts/tf.sh apply
#
# Env overrides: HOST (ssh alias, default hubstream), PORT (k3s API, default 6443).
set -euo pipefail
cd "$(dirname "$0")/../deploy/terraform"

HOST="${HOST:-hubstream}"
PORT="${PORT:-6443}"
CTRL="${TMPDIR:-/tmp}/yapper-k3s-tunnel.sock"

port_open() { nc -z -w2 127.0.0.1 "$PORT" >/dev/null 2>&1; }
# A bound port is NOT proof the tunnel works: a stale/wedged forward still accepts the connection
# but never reaches the k3s API, which surfaces downstream as a cryptic "TLS handshake timeout".
# Confirm the API actually answers — any HTTP status (even 401) means the forward is live.
api_up() { [ "$(curl -sk --max-time 5 -o /dev/null -w '%{http_code}' "https://127.0.0.1:${PORT}/version" 2>/dev/null)" != "000" ]; }

# Drop a stale control socket left by a previously-crashed run.
if [ -S "$CTRL" ] && ! ssh -S "$CTRL" -O check "$HOST" >/dev/null 2>&1; then
  rm -f "$CTRL"
fi

opened=0
if api_up; then
  echo ">> k3s API already reachable on 127.0.0.1:${PORT} — reusing existing tunnel"
elif port_open; then
  # Something holds the port but the API doesn't answer => a dead tunnel. Don't silently reuse it
  # (that's the TLS-handshake-timeout trap); fail with the one-line fix instead.
  echo "!! 127.0.0.1:${PORT} is occupied but the k3s API doesn't answer — a stale/dead SSH tunnel." >&2
  echo "   Close it and retry:  pkill -f '${PORT}:localhost:${PORT}'" >&2
  exit 1
else
  echo ">> opening ssh tunnel: 127.0.0.1:${PORT} -> ${HOST}:localhost:${PORT}"
  ssh -fN -M -S "$CTRL" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
      -L "${PORT}:localhost:${PORT}" "$HOST"
  opened=1
  for _ in $(seq 1 20); do port_open && break; sleep 0.5; done   # wait up to ~10s for the forward
  port_open || { echo "!! tunnel did not come up on :${PORT}" >&2; ssh -S "$CTRL" -O exit "$HOST" 2>/dev/null || true; exit 1; }
fi

cleanup() {
  if [ "$opened" = 1 ]; then
    echo ">> closing ssh tunnel"
    ssh -S "$CTRL" -O exit "$HOST" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

terraform "$@"
