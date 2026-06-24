# 解說 Platform — deployment

Turns the `yap` CLI into a multi-user web service on one EC2 control plane. The
GPU box (WhisperX ASR + CosyVoice2 TTS) stays where it is; the CLI keeps working
unchanged (it uses `LocalStorage` and never touches any of this).

## Topology

```
Browser ──cookie──▶ nginx ─▶ FastAPI (api)            object store (per session)
                              │ enqueue                ▲ presigned PUT/GET   local: MinIO
                              ▼                         │                    prod:  AWS S3
                      Redis (broker + budget) ─▶ Celery workers (cpu / asr / tts / llm / render)
                              │                         │ gRPC over WireGuard
                      Postgres (sessions/movies/runs)   ▼
                                          GPU box: ASR :50051  TTS :50052  (gRPC + /metrics 9101/9102, dcgm)
```

## Profiles: local vs production

Same code, two profiles that differ only by env. Files are object-stored in **both**
(boto3 + `endpoint_url`), so local/prod share one code path:

|           | files            | datastores                               | GPU services        | cookies         |
| --------- | ---------------- | ---------------------------------------- | ------------------- | --------------- |
| **local** | MinIO (in-stack) | redis+postgres containers / StatefulSets | external box (gRPC) | insecure (http) |
| **prod**  | AWS S3           | in-cluster StatefulSets                  | external box (gRPC) | secure (https)  |

The one wrinkle: presigned URLs embed the host, and the browser reaches MinIO under a
different name than the in-cluster services do — so `S3_PUBLIC_ENDPOINT_URL` (browser)
is signed separately from `S3_ENDPOINT_URL` (internal). Prod (AWS) leaves both unset.

## Docker Compose

```bash
cp deploy/env/local.env deploy/env/local.env.mine   # fill LLM_API_KEY, point *_GRPC_TARGET at the box
# LOCAL (MinIO):
docker compose -f deploy/compose/docker-compose.base.yml -f deploy/compose/docker-compose.local.yml up -d --build
#   app:  http://localhost:8080    MinIO console: http://localhost:9001
# PROD (AWS S3, single EC2) — host-metrics profile enables node-exporter (Linux host only):
COMPOSE_PROFILES=host-metrics docker compose -f deploy/compose/docker-compose.base.yml -f deploy/compose/docker-compose.prod.yml up -d --build
```

`deploy/docker-compose.yml` is the original single-file stack (still valid). The
`docker-compose.observability.yml` overlay adds Prometheus/Grafana/Loki if you don't
already run them. Tables auto-create on first start (`init_db`); swap in Alembic later.

## Kubernetes (kustomize)

```bash
# build + make the image available to the cluster
docker build -f deploy/Dockerfile -t yapper:latest .
kind load docker-image yapper:latest            # local (kind)

kubectl apply -k deploy/k8s/overlays/local            # MinIO + in-cluster pg/redis; app on :30800
kubectl apply -k deploy/k8s/overlays/prod             # AWS S3; in-cluster pg/redis StatefulSets; TLS ingress + HPA
```

Render without applying: `kubectl kustomize deploy/k8s/overlays/{local,prod}`.

- **Worker pods use an emptyDir `WORK_ROOT`** — no shared RWX volume, because S3 is the
  source of truth and each phase task re-`materialize`s from it. (Optional EFS RWX in
  prod avoids re-downloading the movie between phases.)
- **GPU box is external**: a selector-less `asr`/`tts` Service + manual `Endpoints` maps
  `asr:50051` / `tts:50052` to the box's WireGuard IP — patch the IP in
  `base/gpu-external.yaml`. `worker-asr` and `worker-tts` are separate (different GPUs ->
  they overlap), each `replicas: 1` (one model per GPU serializes).
- **Prod secrets**: replace the placeholders in `base/config.yaml`'s Secret (or use
  sealed-secrets / external-secrets / SSM); for S3, attach an IRSA role to the pods
  instead of static keys.

## gRPC (ASR + TTS)

The worker↔GPU calls are gRPC (`yapper_rpc.Asr` / `yapper_rpc.Tts`); the browser↔API
stays REST and API↔workers stays on the Redis broker. Stubs are generated/committed in
`yapper_rpc/` (`bash scripts/gen_protos.sh` to regenerate). Run the servers on the box
per `server/README_deploy.md`. Verify:

```bash
grpcurl -plaintext <box>:50051 grpc.health.v1.Health/Check
curl -s <box>:9101/metrics | head      # ASR Prometheus side port (TTS: 9102)
```

## EC2 ⇄ GPU box link (WireGuard, recommended)

The current Mac→`nlp` flow uses an SSH tunnel; for an always-on server use a persistent
**WireGuard** VPN so `ASR_SERVER_URL` / `TTS_SERVER_URL` can point at a stable peer IP
(e.g. `http://10.10.0.2:8900`). An `autossh -N -L 8900:localhost:8900 -L 8901:localhost:8901 nlp`
tunnel from the EC2 is a fine fallback. Demucs (`scripts/separate_score.py`) keeps using
ssh/scp to the box.

On the GPU box, also run **dcgm-exporter** (`:9400`) for GPU utilization/memory, and make
sure the ASR/TTS services expose `/metrics` (they do via `server/_metrics.py` once
`prometheus-client` is installed in their venvs).

## Scaling notes

- The **GPU box is the bottleneck and a SPOF.** Two movies overlap fine on cpu/llm/render,
  but ASR and TTS each serialize on their own queue (concurrency 1 + a Redis lock),
  while overlapping each other across their two GPUs. To go
  faster, add GPU workers + point them at more GPUs — not more EC2.
- **$3 LLM cap** is enforced in `yapper_web/budget.py`: an atomic Redis reserve at
  enqueue rejects a run that can't afford the worst case, and a mid-run hard check aborts
  cleanly. Watch `解說 · Cost` in Grafana; the `LLMBudgetNearlyExhausted` alert fires at 80%.
- **Disk**: render scratch + the per-run working cache live on the `work` volume. `celery
beat` prunes after S3 persist; size the volume for a few movies' worth of re-encodes and
  watch the disk-free panel.
- **Storage**: S3 is durable; the local `work` volume is just a cache. On a single host
  it stays warm so `materialize`/`persist` are near-no-ops; a second host refills on miss.

## Metrics & logs

- `prometheus/scrape.yml` — jobs for api, pushgateway, node-exporter, GPU services, dcgm.
- `prometheus/alerts.yml` — budget / GPU-down / disk / backlog / failure alerts.
- `promtail/promtail.yml` — ships JSON logs (run_id/session_id/stage/lang labels) to Loki.
- `grafana/dashboards/` — Fleet Overview + Cost. Run drill-down: filter Loki by `run_id`
  and the pushgateway series (`yapper_stage_duration_seconds`, grouping_key run_id).
