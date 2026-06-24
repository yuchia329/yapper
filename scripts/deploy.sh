#!/usr/bin/env bash
# One-command release: build the yapper image, deliver it to the prod k3s box, and roll the
# Deployments onto it.
#
# WHY a plain `terraform apply` doesn't update the app: the prod overlay tags the image :latest
# and the pods use imagePullPolicy: IfNotPresent. So `apply` with the default image_tag=latest
# renders a manifest identical to state -> ZERO changes -> no rollout; and even a forced restart
# would reuse the cached :latest instead of pulling the new one. The fix is a UNIQUE tag (the git
# short sha): it changes the rendered image string -> the Deployment template changes -> a rolling
# update -> the node pulls the new, not-yet-present tag.
#
#   bash scripts/deploy.sh                 # build + push to Docker Hub, then terraform apply @ <sha>
#   DELIVERY=load bash scripts/deploy.sh   # side-load into k3s containerd instead (no registry pull)
#   IMAGE_TAG=hotfix1 bash scripts/deploy.sh   # override the tag (default: git short sha)
#
# Env: DELIVERY=push|load (default push). A dirty tree auto-tags <sha>-<timestamp> and still ships.
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck source=lib/gpu.sh
. scripts/lib/gpu.sh

DELIVERY="${DELIVERY:-push}"

# Pick the image tag. Priority: explicit IMAGE_TAG > clean git sha > git sha + UTC timestamp (dirty).
# The image is built from the WORKING TREE, so a dirty tree must NOT be tagged with the bare commit
# sha: it would (a) misrepresent provenance and (b) collide with a future clean build of that sha,
# which IfNotPresent would then refuse to re-pull. A timestamp suffix keeps the tag unique + honest.
if [ -n "${IMAGE_TAG:-}" ]; then
  TAG="$IMAGE_TAG"
else
  SHA="$(git rev-parse --short HEAD 2>/dev/null || true)"
  [ -n "$SHA" ] || { echo "!! no git HEAD and no IMAGE_TAG — set IMAGE_TAG=<tag> and retry" >&2; exit 1; }
  if git diff --quiet HEAD 2>/dev/null; then
    TAG="$SHA"
  else
    TAG="${SHA}-$(date -u +%Y%m%d-%H%M%S)"
    echo ">> working tree is dirty — building from it and tagging :${TAG} (sha + UTC timestamp, unique)." >&2
    echo "   tip: commit for a clean :${SHA} tag." >&2
  fi
fi
[ "$TAG" = "latest" ] && { echo "!! refusing to deploy tag 'latest' (IfNotPresent won't re-pull it)" >&2; exit 1; }

echo "==> 1/3  build + deliver image   (DELIVERY=${DELIVERY}, tag=${TAG})"
case "$DELIVERY" in
  push) IMAGE_TAG="$TAG" bash scripts/build_and_push.sh ;;
  # side-load must import under the SAME name the manifest references (docker.io/yuchia329/yapper),
  # not build_and_load.sh's default 'yapper', or IfNotPresent won't match the imported image.
  load) IMAGE="docker.io/yuchia329/yapper" IMAGE_TAG="$TAG" bash scripts/build_and_load.sh ;;
  *) echo "!! DELIVERY must be 'push' or 'load'" >&2; exit 1 ;;
esac

echo "==> 2/3  terraform apply @ ${TAG}   (rolls the pods + applies the in-cluster gpu-tunnel; via the auto :6443 tunnel)"
# Pass the tag as -var (HIGHEST precedence) so it always wins over any image_tag pinned in
# terraform.tfvars. A tfvars value silently overrides TF_VAR_*, which would make this a no-op:
# the manifest's `yapper:latest` never gets rewritten -> IfNotPresent -> the pods never roll.
# tf.sh brings up the `ssh -L 6443` k3s-API tunnel if it isn't already (idempotent); terraform
# (re)applies the gpu-tunnel Deployment so the in-cluster ssh forward to gpud exists.
bash scripts/tf.sh apply -var "image_tag=${TAG}"

echo "==> 3/3  ensure gpud on the GPU box   (so the freshly-rolled workers' ASR/TTS leases succeed)"
# The in-cluster gpu-tunnel pod forwards to gpud, but only if gpud is actually running on the box.
# Idempotent + non-fatal: a down/unreachable box doesn't fail the deploy of the CPU/LLM/render path.
gpu_ensure_gpud || true

echo
echo ">> deployed ${TAG}. watch the rollout (needs the :6443 tunnel up):"
echo "     KUBECONFIG=~/.kube/yapper-k3s.yaml kubectl -n yapper rollout status deploy/api"
echo "     KUBECONFIG=~/.kube/yapper-k3s.yaml kubectl -n yapper rollout status deploy/worker-cpu"
echo "     KUBECONFIG=~/.kube/yapper-k3s.yaml kubectl -n yapper get pods -o wide"
