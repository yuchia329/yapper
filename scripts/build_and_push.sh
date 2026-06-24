#!/usr/bin/env bash
# Build the yapper image for linux/arm64 (prod is a t4g.large = ARM/Graviton box) and PUSH it
# to Docker Hub as yuchia329/yapper. This is the REGISTRY-PULL deploy flow: k3s on the prod box
# pulls the image from Docker Hub — vs scripts/build_and_load.sh which side-loads it into
# containerd over SSH (no registry). The prod kustomize overlay already references
# docker.io/yuchia329/yapper, so nothing in the manifests changes.
#
#   docker login                                    # once, as the user that owns the repo (yuchia329)
#   bash scripts/build_and_push.sh                  # pushes :latest + :<git-sha>
#   IMAGE_TAG=v1 bash scripts/build_and_push.sh     # also pushes :v1
#   PLATFORM=linux/amd64,linux/arm64 bash scripts/build_and_push.sh   # multi-arch (slower)
#
# IMPORTANT — imagePullPolicy is IfNotPresent (deploy/k8s/base/*.yaml), so re-pushing :latest
# does NOT cause the node to re-pull. Deploy the immutable :<git-sha> tag instead; Terraform
# rewrites yapper:latest -> yapper:<image_tag> so a new tag always forces a fresh pull:
#     export TF_VAR_image_tag=<git-sha>  &&  (cd deploy/terraform && terraform apply)
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-yuchia329/yapper}"
PLATFORM="${PLATFORM:-linux/arm64}"        # prod box is ARM64 (Graviton t4g.large)
BUILDER="${BUILDER:-yapper-builder}"
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
IMAGE_TAG="${IMAGE_TAG:-}"                 # optional extra human tag (e.g. v1)

# tags: :latest (as requested) + the immutable git sha for reproducible, re-pullable deploys.
TAGS=(-t "${IMAGE}:latest")
[ "${GIT_SHA}" != "unknown" ] && TAGS+=(-t "${IMAGE}:${GIT_SHA}")
[ -n "${IMAGE_TAG}" ] && TAGS+=(-t "${IMAGE}:${IMAGE_TAG}")

# --- preflight: buildx + a builder that can push / cross-build -------------
command -v docker >/dev/null 2>&1 || { echo "!! docker not found on PATH" >&2; exit 1; }
docker buildx version >/dev/null 2>&1 || { echo "!! docker buildx unavailable (need Docker >= 19.03)" >&2; exit 1; }

# A docker-container builder reliably supports --push and multi-platform builds (the default
# 'docker' driver can't push a multi-arch manifest). Created once, reused thereafter.
if ! docker buildx inspect "${BUILDER}" >/dev/null 2>&1; then
  echo ">> creating buildx builder '${BUILDER}' (one-time; bootstraps a buildkit container)"
  docker buildx create --name "${BUILDER}" --driver docker-container --bootstrap >/dev/null
fi

# --- preflight: warn if not logged in to Docker Hub (push needs auth) ------
# Detect any of: a stored auth entry, a credsStore, or a credHelper. Non-fatal — if this is a
# false negative (Docker Desktop keychain), the push itself surfaces the authoritative error.
CFG="${HOME}/.docker/config.json"
if ! grep -Eq '"(https://index\.docker\.io/v1/|auths|credsStore|credHelpers)"' "${CFG}" 2>/dev/null; then
  echo "!! You may not be logged in to Docker Hub. If the push 401s, run:  docker login" >&2
fi

echo ">> building + pushing ${IMAGE}  [${PLATFORM}]"
echo "   tags: latest${GIT_SHA:+, ${GIT_SHA}}${IMAGE_TAG:+, ${IMAGE_TAG}}"
docker buildx build \
  --builder "${BUILDER}" \
  --platform "${PLATFORM}" \
  -f deploy/Dockerfile \
  "${TAGS[@]}" \
  --push \
  .

echo
echo ">> pushed to Docker Hub:"
echo "     ${IMAGE}:latest"
[ "${GIT_SHA}" != "unknown" ] && echo "     ${IMAGE}:${GIT_SHA}   <- deploy this immutable tag"
echo
echo ">> deploy via Terraform (registry pull):"
echo "     export TF_VAR_image_tag=${GIT_SHA}"
echo "     (cd deploy/terraform && terraform apply)"
echo ">> verify on the cluster:"
echo "     kubectl -n yapper rollout status deploy/api"
echo "     kubectl -n yapper get pods -o wide"
