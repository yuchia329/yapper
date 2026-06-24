# Terraform owns the namespace + the Secrets the kustomize overlay deliberately omits, so
# real credentials never live in the repo.
resource "kubernetes_namespace" "yapper" {
  metadata {
    name   = "yapper"
    labels = { "app.kubernetes.io/part-of" = "yapper" }
  }
}

resource "kubernetes_secret" "app" {
  metadata {
    name      = "yapper-secrets"
    namespace = kubernetes_namespace.yapper.metadata[0].name
  }
  type = "Opaque"
  data = {
    LLM_API_KEY           = var.llm_api_key
    SESSION_SECRET        = var.session_secret
    POSTGRES_PASSWORD     = var.postgres_password
    DATABASE_URL          = "postgresql+psycopg://jieshuo:${var.postgres_password}@postgres:5432/jieshuo"
    AWS_ACCESS_KEY_ID     = aws_iam_access_key.yapper.id
    AWS_SECRET_ACCESS_KEY = aws_iam_access_key.yapper.secret
  }
}

# SSH key for the gpu-tunnel (same key video-search uses to reach the GPU box).
resource "kubernetes_secret" "gpu_ssh" {
  metadata {
    name      = "gpu-ssh-key"
    namespace = kubernetes_namespace.yapper.metadata[0].name
  }
  type = "Opaque"
  data = {
    id_rsa = var.gpu_ssh_private_key
  }
}
