# --- cluster ---------------------------------------------------------------
variable "kubeconfig_path" {
  description = "Path to a kubeconfig pointing at the k3s API (e.g. via the ssh -L 6443 tunnel)."
  type        = string
  default     = "~/.kube/yapper-k3s.yaml"
}

variable "image_tag" {
  description = "yapper image tag loaded into k3s by scripts/build_and_load.sh (e.g. the git short sha). Substituted into the kustomize-rendered manifests."
  type        = string
  default     = "latest"
}

# --- AWS S3 (artifact store) ----------------------------------------------
variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "s3_bucket" {
  description = "Bucket for movie inputs + yapper outputs (must match S3_BUCKET in the overlay)."
  type        = string
  default     = "recap-artifacts-prod"
}

variable "artifact_retention_days" {
  description = "Objects older than this are deleted by the S3 lifecycle rule (the scheduled cleanup)."
  type        = number
  default     = 30
}

# --- DNS (Cloudflare) ------------------------------------------------------
variable "cloudflare_api_token" {
  description = "Cloudflare API token with DNS edit on the yuchia.dev zone."
  type        = string
  sensitive   = true
}

variable "cloudflare_zone_id" {
  description = "Zone ID for yuchia.dev."
  type        = string
}

variable "ec2_public_ip" {
  description = "Public IP the yapper.yuchia.dev record points at."
  type        = string
  default     = "34.195.244.172"
}

# --- app secrets -----------------------------------------------------------
variable "llm_api_key" {
  description = "MiniMax (OpenAI-compatible) API key."
  type        = string
  sensitive   = true
}

variable "session_secret" {
  description = "Long random string for signing the session cookie."
  type        = string
  sensitive   = true
}

variable "postgres_password" {
  description = "Password for the in-cluster Postgres (used in DATABASE_URL too)."
  type        = string
  sensitive   = true
}

variable "gpu_ssh_private_key" {
  description = "SSH private key (id_rsa) the gpu-tunnel uses to reach the GPU box. Reuse the same key as video-search."
  type        = string
  sensitive   = true
}
