#!/usr/bin/env bash
# Build the yapper image for linux/arm64 and load it into the k3s containerd on the prod box
# (single node → no registry needed). Run from the repo root on an Apple-Silicon Mac (Docker
# Desktop builds arm64 natively) or any arm64 host; the t4g box is also arm64 if you'd rather
# build there (REMOTE_BUILD=1).
#
#   bash scripts/build_and_load.sh                         # tag = git short sha
#   IMAGE_TAG=v1 bash scripts/build_and_load.sh
#   HOST=hubstream bash scripts/build_and_load.sh
#
# The k3s kubelet resolves the manifest's `yapper:<tag>` to docker.io/library/yapper:<tag>,
# which is what `k3s ctr images import` stores, so imagePullPolicy: IfNotPresent finds it.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${HOST:-hubstream}"
IMAGE="${IMAGE:-yapper}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo latest)}"
REF="${IMAGE}:${IMAGE_TAG}"
PLATFORM="linux/arm64"

echo ">> building ${REF} for ${PLATFORM}"
docker build --platform "${PLATFORM}" -f deploy/Dockerfile -t "${REF}" .

echo ">> streaming ${REF} into k3s containerd on ${HOST} (namespace k8s.io)"
docker save "${REF}" | ssh "${HOST}" "sudo k3s ctr images import -"

echo
echo ">> loaded. Point the deploy at this tag, then apply:"
echo "     export TF_VAR_image_tag=${IMAGE_TAG}      # if deploying via Terraform"
echo "     # or: (cd deploy/k8s/overlays/prod && kustomize edit set image yapper=${IMAGE}:${IMAGE_TAG})"
echo ">> verify on the box:"
echo "     ssh ${HOST} 'sudo k3s ctr images ls | grep ${IMAGE}'"
