#!/usr/bin/env bash
# Shared, IDEMPOTENT helpers for reaching the GPU box's gpud — sourced by the three deploy
# entry points so "ensure gpud if not exist" and "ensure the ssh tunnel if not exist" mean the
# exact same thing everywhere:
#   • scripts/run_local_compose.sh  (docker compose — tunnel is the in-compose sidecar)
#   • scripts/deploy.sh             (EC2/k3s — tunnel is the in-cluster gpu-tunnel pod)
#   • scripts/gpu_tunnel.sh         (local, no docker — tunnel is a host ssh -L forward)
#
# Not executable on its own. `. scripts/lib/gpu.sh` then call the functions below.
#
# Env knobs (all optional, sane defaults):
#   GPU_SSH_HOST       ssh alias / host for the GPU box           (default: nlp)
#   GPUD_REMOTE_DIR    where ~/jieshuo lives on the box           (default: ~/jieshuo)
#   GPUD_PORT          gpud control port                          (default: 50050)
#   GPUD_PORT_RANGE    on-demand instance pool                    (default: 50060-50099)
#   GPUD_METRICS_PORT  gpud Prometheus port                       (default: 9050)
#   BIND               local bind addr for host forwards          (default: 127.0.0.1)
#   ENSURE_GPUD=0      skip the gpud check entirely (you manage it yourself)
#
# Source this UNDER BASH (the entry points all use `#!/usr/bin/env bash`); the port probe's
# /dev/tcp fallback is bash-only.

# Control-socket path for the background tunnel master, shared by gpu_ensure_tunnel + gpu_tunnel_down
# so the two can never drift to different paths — a TMPDIR difference between the open and the
# teardown (sudo / cron / a different login session) would otherwise orphan the master. A per-uid
# name avoids collisions in world-writable /tmp; override with GPU_TUNNEL_CTRL.
GPU_TUNNEL_CTRL="${GPU_TUNNEL_CTRL:-/tmp/yapper-gpud-tunnel-$(id -u 2>/dev/null || echo u).sock}"

# True iff something is already listening on the local gpud control port — i.e. a tunnel
# (sidecar, host forward, or k8s port-forward) is already established. Prefers nc; falls back to
# bash's /dev/tcp on hosts without netcat. NOTE: this proves the port is BOUND, not that the
# forward is live (cf. tf.sh's stale-tunnel guard) — callers treat OUR control socket as the
# authoritative liveness signal and use this only as a fast hint.
gpu_tunnel_port_open() {
  local port="${1:-${GPUD_PORT:-50050}}"
  if command -v nc >/dev/null 2>&1; then
    nc -z -w2 127.0.0.1 "$port" >/dev/null 2>&1
  elif [ -n "${BASH_VERSION:-}" ]; then
    (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null
  else
    # No nc and not bash: can't probe. Report closed so callers (re)open rather than skip silently.
    return 1
  fi
}

# Idempotently ensure gpud is RUNNING ON THE BOX. The launch needs box-side secrets, so it's
# delegated to server/start_gpud.sh on the box (itself a no-op when gpud is already up — prod and
# local dev SHARE one gpud). NON-FATAL: a GPU box that's unreachable shouldn't block a deploy of
# the CPU/LLM/render stages, so callers run this as `gpu_ensure_gpud || true` and we just warn.
# Returns non-zero if it could not confirm gpud, so a caller MAY choose to react.
gpu_ensure_gpud() {
  local host="${GPU_SSH_HOST:-${1:-nlp}}"
  if [ "${ENSURE_GPUD:-1}" != "1" ]; then
    echo ">> ENSURE_GPUD=0 — skipping gpud check on ${host}"
    return 0
  fi
  echo ">> ensuring gpud on ${host} (idempotent; prod + local share one gpud)…"
  # The literal ~ stays unexpanded in the quotes and expands on the BOX.
  # shellcheck disable=SC2029
  if ssh -o ConnectTimeout=10 "$host" "bash ${GPUD_REMOTE_DIR:-~/jieshuo}/server/start_gpud.sh"; then
    echo ">> gpud is up on ${host}"
    return 0
  fi
  echo "!! could not ensure gpud on ${host} — ASR/TTS stages will fail until it's running." >&2
  echo "   check 'ssh ${host}', then: ssh ${host} 'bash ~/jieshuo/server/start_gpud.sh'  (see server/README_deploy.md)" >&2
  return 1
}

# Idempotently ensure a BACKGROUND host ssh tunnel to gpud exists (control port + instance pool +
# metrics). No-op if the control port is already open (so re-running is safe). Managed via an ssh
# control socket so gpu_tunnel_down() can tear down exactly what we opened, while a tunnel started
# some other way is left alone. Used by the no-docker flow; compose/k8s forward gpud their own way.
gpu_ensure_tunnel() {
  local host="${GPU_SSH_HOST:-${1:-nlp}}"
  local port="${GPUD_PORT:-50050}"
  local range="${GPUD_PORT_RANGE:-50060-50099}"
  local metrics="${GPUD_METRICS_PORT:-9050}"
  local bind="${BIND:-127.0.0.1}"
  local ctrl="$GPU_TUNNEL_CTRL"

  # 1. Drop a stale control socket left by a crashed run so we don't reuse a dead master.
  if [ -S "$ctrl" ] && ! ssh -S "$ctrl" -O check "$host" >/dev/null 2>&1; then
    rm -f "$ctrl"
  fi
  # 2. If OUR managed master is alive, that's the authoritative "already up" — no-op. This MUST come
  #    before any open: re-issuing `ssh -M -S "$ctrl"` against a live socket makes ssh print
  #    "ControlSocket already exists, disabling multiplexing" and fork a SECOND, non-multiplexed
  #    connection that gpu_tunnel_down can't reap (a port-holding leak).
  if [ -S "$ctrl" ] && ssh -S "$ctrl" -O check "$host" >/dev/null 2>&1; then
    echo ">> gpud tunnel already up (managed master -> ${bind}:${port}) — reusing"
    return 0
  fi
  # 3. Otherwise, if something else already holds the control port, treat it as an externally
  #    managed tunnel (compose sidecar / a foreground gpu_tunnel.sh / another session) and reuse it
  #    — the idempotent "if not exist" contract. Port-bound != proven-live, so hint at the
  #    stale-tunnel possibility (cf. tf.sh) instead of failing opaquely later.
  if gpu_tunnel_port_open "$port"; then
    echo ">> gpud control port ${bind}:${port} already open (externally-managed tunnel) — reusing"
    echo "   (if ASR/TTS later hang, that forward may be stale: 'gpu_tunnel.sh down' or kill the holder, then retry)" >&2
    return 0
  fi

  # 4. Open a fresh background master: control + metrics + the instance pool.
  local lo="${range%-*}" hi="${range#*-}" p
  local flags=( -L "${bind}:${port}:localhost:${port}" -L "${bind}:${metrics}:localhost:${metrics}" )
  for p in $(seq "$lo" "$hi"); do flags+=( -L "${bind}:${p}:localhost:${p}" ); done

  echo ">> opening gpud tunnel -> ${host}: control ${port} + pool ${lo}-${hi} + metrics ${metrics} (bind ${bind})"
  # GUARD the launch. `ssh -f` returns nonzero if any forward bind fails (ExitOnForwardFailure) or
  # auth fails; left bare, that failure trips the caller's `set -e` and aborts the script BEFORE
  # the diagnostic/return below ever runs. Capturing it keeps this function's contract intact.
  if ! ssh -fN -M -S "$ctrl" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        "${flags[@]}" "$host"; then
    echo "!! could not open the gpud tunnel to ${host}." >&2
    echo "   A forwarded port (${port}, ${metrics}, or pool ${lo}-${hi}) may be held by a leftover/foreign" >&2
    echo "   forward, or ssh/auth failed. Try: 'bash scripts/gpu_tunnel.sh down', or find the holder with" >&2
    echo "   'lsof -nP -iTCP:${port} -sTCP:LISTEN', then check 'ssh ${host}'." >&2
    return 1
  fi

  # -f already backgrounds only after the forwards are up, so this is belt-and-suspenders.
  local i
  for i in $(seq 1 20); do
    gpu_tunnel_port_open "$port" && { echo ">> gpud tunnel up on ${bind}:${port}"; return 0; }
    sleep 0.5
  done
  echo "!! gpud tunnel launched but :${port} never became reachable within ~10s" >&2
  return 1
}

# Tear down the tunnel gpu_ensure_tunnel() opened (via its control socket). A tunnel opened some
# other way (a foreground gpu_tunnel.sh, the compose sidecar, kubectl port-forward) is untouched.
gpu_tunnel_down() {
  local host="${GPU_SSH_HOST:-${1:-nlp}}"
  local ctrl="$GPU_TUNNEL_CTRL"
  if [ -S "$ctrl" ] && ssh -S "$ctrl" -O check "$host" >/dev/null 2>&1; then
    echo ">> closing managed gpud tunnel"
    ssh -S "$ctrl" -O exit "$host" >/dev/null 2>&1 || true
  else
    rm -f "$ctrl" 2>/dev/null || true
    echo ">> no managed gpud tunnel to close (if you started one another way: pkill -f '${GPUD_PORT:-50050}:localhost:${GPUD_PORT:-50050}')"
  fi
}
