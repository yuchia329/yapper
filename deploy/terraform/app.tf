# Apply the existing kustomize prod overlay (single source of truth for the workloads) via
# the kustomization provider, so every app resource is tracked in Terraform state. The image
# tag rendered as `yapper:latest` is rewritten to the tag built by scripts/build_and_load.sh.
data "kustomization_build" "prod" {
  path = "${path.module}/../k8s/overlays/prod"
}

resource "kustomization_resource" "prod" {
  for_each = data.kustomization_build.prod.ids

  manifest = replace(
    data.kustomization_build.prod.manifests[each.value],
    "yapper:latest",
    "yapper:${var.image_tag}",
  )

  # Namespace + Secrets (TF-owned) must exist before the workloads that reference them.
  depends_on = [
    kubernetes_namespace.yapper,
    kubernetes_secret.app,
    kubernetes_secret.gpu_ssh,
  ]
}
