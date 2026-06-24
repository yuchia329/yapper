#!/usr/bin/env bash
# Local NON-DOCKER GPU access: idempotently ensure gpud is up on the box AND a host ssh tunnel to
# it exists, so the CLI / a bare `yapper_web` can reach gpud at localhost:50050 (set
# GPU_SUPERVISOR_TARGET=localhost:50050). Re-running is always safe — gpud is shared and the
# tunnel is opened only "if not exist".
#
#   bash scripts/gpu_tunnel.sh              # ensure gpud + open a BACKGROUND tunnel, then return
#   bash scripts/gpu_tunnel.sh nlp          # same, against ssh host 'nlp' (legacy positional form)
#   bash scripts/gpu_tunnel.sh fg           # ensure gpud + run the tunnel in the FOREGROUND (blocks)
#   bash scripts/gpu_tunnel.sh down         # tear down the background tunnel this script opened
#
# Env: GPU_SSH_HOST (default nlp), GPUD_PORT (50050), GPUD_PORT_RANGE (50060-50099),
#      GPUD_METRICS_PORT (9050), BIND (127.0.0.1; set 0.0.0.0 for a host.docker.internal fallback),
#      ENSURE_GPUD=0 (skip the gpud check). See scripts/lib/gpu.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/gpu.sh
. "$SCRIPT_DIR/lib/gpu.sh"

# First arg may be an ACTION (up|fg|down) or, for backward-compat, the HOST (e.g. `gpu_tunnel.sh nlp`).
ACTION=up
case "${1:-}" in
  up|fg|down) ACTION="$1"; shift ;;
esac
export GPU_SSH_HOST="${GPU_SSH_HOST:-${1:-nlp}}"

if [ "$ACTION" = down ]; then
  gpu_tunnel_down
  exit 0
fi

# Both up + fg first make sure gpud is actually running on the box (non-fatal — tunnelling to a
# down gpud is pointless, but we still surface the tunnel for partial use).
gpu_ensure_gpud || true

if [ "$ACTION" = up ]; then
  gpu_ensure_tunnel
  echo
  echo ">> ready. point the app at it:  export GPU_SUPERVISOR_TARGET=localhost:${GPUD_PORT:-50050}"
  echo "   stop the tunnel:             bash scripts/gpu_tunnel.sh down"
  exit 0
fi

# ACTION = fg: foreground, blocking tunnel (Ctrl-C to stop). Idempotent: bail if one's already up.
PORT="${GPUD_PORT:-50050}"
if gpu_tunnel_port_open "$PORT"; then
  echo ">> gpud tunnel already up on 127.0.0.1:${PORT} — nothing to do (Ctrl-C a previous run to replace it)"
  exit 0
fi
RANGE="${GPUD_PORT_RANGE:-50060-50099}"; LO="${RANGE%-*}"; HI="${RANGE#*-}"
BIND="${BIND:-127.0.0.1}"; METRICS="${GPUD_METRICS_PORT:-9050}"
flags=( -N -L "${BIND}:${PORT}:localhost:${PORT}" -L "${BIND}:${METRICS}:localhost:${METRICS}" )
for p in $(seq "$LO" "$HI"); do flags+=( -L "${BIND}:${p}:localhost:${p}" ); done
echo ">> tunneling gpud ${PORT} + pool ${LO}-${HI} + metrics ${METRICS} to ${GPU_SSH_HOST} on ${BIND} (foreground; Ctrl-C to stop)…"
exec ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 "${flags[@]}" "$GPU_SSH_HOST"
