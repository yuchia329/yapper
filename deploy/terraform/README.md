# Terraform — yapper.yuchia.dev production deploy

Deploys the Yapper commentary platform onto the existing **k3s/ARM EC2** (`hubstream`) and wires
the cloud bits around it, in one `terraform apply`:

| Provider | Manages |
|---|---|
| `aws` | S3 artifact bucket + **lifecycle retention** (the scheduled cleanup) + CORS + a scoped IAM user/key |
| `cloudflare` | `yapper.yuchia.dev` A record (proxied → edge TLS) |
| `kubernetes` | `yapper` namespace + the `yapper-secrets` / `gpu-ssh-key` Secrets + the Grafana dashboard ConfigMap (in `monitoring`) |
| `kustomization` | the app workloads + observability CRs by applying `../k8s/overlays/prod` (api, worker, postgres, redis, gpu-tunnel, ingress, **ServiceMonitors + PrometheusRule**) |

**Out of scope (by design):** the physical GPU box (`nlp-gpu-01.be.ucsc.edu`). Terraform
can't provision hardware — run gpud there once (Part C below). The in-cluster `gpu-tunnel`
connects to it at runtime.

## Prerequisites
- `terraform >= 1.5`, `docker` (on an arm64 host — Apple Silicon or the box itself), `kubectl`.
- AWS credentials in your environment (`AWS_PROFILE` or `AWS_ACCESS_KEY_ID`/`…SECRET…`).
- A Cloudflare API token with **DNS edit** on the `yuchia.dev` zone, and the zone id.
- The GPU box SSH **private key** (the same one `video-search` uses).

## 0. Point kubectl/Terraform at the k3s API (SSH tunnel)
The k3s cert isn't valid for the public IP, so forward the API over SSH and use a local kubeconfig:
```bash
ssh -fN -L 6443:localhost:6443 hubstream
scp hubstream:/etc/rancher/k3s/k3s.yaml ~/.kube/yapper-k3s.yaml   # server is already https://127.0.0.1:6443
KUBECONFIG=~/.kube/yapper-k3s.yaml kubectl get ns                  # sanity check
```

**To skip the manual tunnel for Terraform**, use the wrapper — it opens the `:6443` forward only
if it isn't already up, runs terraform, then closes the tunnel it opened:
```bash
bash ../../scripts/tf.sh plan
TF_VAR_image_tag=$(git rev-parse --short HEAD) bash ../../scripts/tf.sh apply
```

## 1. Build the ARM image and load it into k3s
```bash
IMAGE_TAG=$(git rev-parse --short HEAD) bash ../../scripts/build_and_load.sh
export TF_VAR_image_tag=$(git rev-parse --short HEAD)
```

## 2. Configure
```bash
cp terraform.tfvars.example terraform.tfvars     # fill it in (gitignored)
export TF_VAR_gpu_ssh_private_key="$(cat ~/.ssh/your_gpu_key)"   # keep keys out of files
```

## 3. Apply
```bash
terraform init
terraform plan      # review: S3 bucket+lifecycle, IAM user, Cloudflare record, ns/secrets, ~8 k8s resources
terraform apply
```

## Part C — gpud on the GPU box (one-time, separate)
```bash
scp -r ../../server ../../yapper_rpc nlp-gpu-01.be.ucsc.edu:~/jieshuo/
ssh nlp-gpu-01.be.ucsc.edu 'bash ~/jieshuo/server/setup_gpu.sh'
# run gpud under systemd (Restart=always) with GPUD_PORT_RANGE=50060-50099 + HF_TOKEN + TTS voice
# — see server/README_deploy.md
```

## Verify
```bash
kubectl -n yapper get pods,ingress
dig yapper.yuchia.dev +short
open https://yapper.yuchia.dev          # upload a short clip; front half + "play original" work via S3
kubectl -n yapper logs deploy/gpu-tunnel   # forwards up; ASR/TTS stages complete
# Prometheus → Status/Targets shows yapper-api + yapper-gpud UP; Grafana has "Yapper · Overview".
```

## Notes
- **Footprint:** the box is 2 vCPU / 8 GB shared with monitoring + hubstream + video-search
  (~4.9 GB free at last check). yapper is right-sized to ~2 GB. Watch `kubectl top node` after
  apply; if tight, the pushgateway is already off and you can bump to t4g.xlarge.
- **Image tag:** `var.image_tag` is string-substituted into the rendered `yapper:latest` so a
  unique tag (git sha) forces a rollout. Re-run step 1 + `terraform apply` to ship a new build.
- **Secrets** live only in `terraform.tfvars` / `TF_VAR_*` + Terraform state — never in git or
  the kustomize manifests (the overlay deletes the placeholder Secrets).
- **Alternative to the kustomization provider:** `kubectl apply -k ../k8s/overlays/prod` after
  `terraform apply` has created the namespace + secrets.
