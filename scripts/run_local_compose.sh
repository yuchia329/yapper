#!/usr/bin/env bash
# Quick local run of the yapper app via Docker Compose (base + local overlay).
#   app:           http://localhost:8080
#   MinIO console: http://localhost:9001   (minioadmin / minioadmin)
#
#   bash scripts/run_local_compose.sh            # build image + (re)start the whole stack, detached
#   bash scripts/run_local_compose.sh fast       # restart WITHOUT rebuilding (reuse current image)
#   bash scripts/run_local_compose.sh down       # stop + remove containers (KEEPS volumes/data)
#   bash scripts/run_local_compose.sh logs [svc] # follow logs (default svc: api)
#   bash scripts/run_local_compose.sh ps         # service status
#   bash scripts/run_local_compose.sh nginx      # force-recreate nginx (after editing nginx.conf)
#
# NOTE: the source is baked into the image (no bind-mount), so code edits only take effect after a
# rebuild — use the default `up`, not `fast`, when you've changed yapper / yapper_web.
set -euo pipefail
cd "$(dirname "$0")/.."

DC=(docker compose
    -f deploy/compose/docker-compose.base.yml
    -f deploy/compose/docker-compose.local.yml)

cmd="${1:-up}"
case "$cmd" in
  up)    BUILD=1 ;;
  fast)  BUILD=0 ;;
  down)  exec "${DC[@]}" down ;;
  ps)    exec "${DC[@]}" ps ;;
  logs)  shift; exec "${DC[@]}" logs -f --tail=100 "${1:-api}" ;;
  nginx) exec "${DC[@]}" up -d --force-recreate nginx ;;   # bind-mounted nginx.conf needs a recreate
  *)     echo "usage: $(basename "$0") [up|fast|down|logs [svc]|ps|nginx]" >&2; exit 2 ;;
esac

# --- preflight: warn (don't block) on missing local secrets / GPU key -------
[ -f deploy/env/local.env.mine ] || \
  echo "!! deploy/env/local.env.mine missing — no LLM_API_KEY / GPU targets; LLM + ASR/TTS stages will fail." >&2
[ -f "${HOME}/.ssh/nlp_ed25519" ] || \
  echo "!! ~/.ssh/nlp_ed25519 missing — the gpu-tunnel sidecar can't reach the GPU box; non-GPU stages still run." >&2

# All 7 app services (api + workers + beat) share ONE image tag (yapper:latest). A plain
# `up --build` fans out 7 parallel builds that race to export that SAME tag into Docker's
# containerd image store and fail with: image "...yapper:latest": already exists. So build
# the shared image exactly ONCE (via the api service), then start WITHOUT --build.
if [ "$BUILD" = 1 ]; then
  echo ">> building shared image yapper:latest (once; code changes picked up)…"
  "${DC[@]}" build api
fi

echo ">> starting stack"
"${DC[@]}" up -d

echo
"${DC[@]}" ps
cat <<'EOF'

>> ready:
     app             http://localhost:8080
     MinIO console   http://localhost:9001   (minioadmin / minioadmin)
   tail logs:  bash scripts/run_local_compose.sh logs api
   stop:       bash scripts/run_local_compose.sh down
EOF
