# Yapper — funny AI commentary for movies & short clips (短片解說)

Turn any video into a punchy Mandarin **funny-commentary** track in the Douyin/Kuaishou
短片解說 style: condensed footage plays under an AI-generated voiceover that retells the
story *and* riffs on it. Works on full-length **movies** (a ~10–12 min funny recap) and
short **clips** like vlogs (a 1–3 min riff). English (YouTube-style) commentary is supported too.

🔗 **[Live demo → yapper.yuchia.dev](https://yapper.yuchia.dev)**  · 
📊 **[Grafana dashboard](https://grafana.yuchia.dev/d/yapper-overview/yapper-c2b7-overview?orgId=1&from=now-24h&to=now&timezone=America%2FLos_Angeles&var-datasource=prometheus&refresh=auto)**

---

## Demo

A 25-second taste (plays inline) — then the full recap on YouTube:



  
*▶ Watch the full recap on YouTube*



---

## How it works

A video goes through **12 deterministic, resumable stages**. Each stage reads and writes one
validated artifact, so any step can be inspected or re-run. The hosted platform splits them
into a **front half** (run once per uploaded video) and a **back half** (run per commentary, so
you can regenerate in another language or a fresh take without re-doing ASR/scene detection).

```
            ┌──────────────── front half (per video, shared) ─────────────────┐
video ─▶ probe ─▶ extract audio ─▶ ASR ─▶ detect shots ─▶ scenes+keyframes ─▶ clip_index (SQLite)
            └─────────────────────────────────────────────────────────────────┘
            ┌──────────────── back half (per commentary / language) ──────────┐
         ─▶ MAP: watch & outline ─▶ REDUCE: 解說 script ─▶ runtime budget
         ─▶ TTS (cloned voice) ─▶ audio-driven EDL ─▶ subtitles ─▶ ffmpeg render ─▶ final.mp4
            └─────────────────────────────────────────────────────────────────┘
```


| #   | Stage              | What it does                                                            | Compute |
| --- | ------------------ | ----------------------------------------------------------------------- | ------- |
| 1   | `ingest` (probe)   | ffprobe dims/rotation/duration → `probe.json`                           | CPU     |
| 2   | `audio`            | extract a mono track for ASR                                            | CPU     |
| 3   | `asr`              | transcribe with **WhisperX** (word timings)                             | **GPU** |
| 4   | `shots`            | shot-boundary detection (PySceneDetect)                                 | CPU     |
| 5   | `scenes`           | group shots → scenes, pick keyframes, build the **clip_index**          | CPU     |
| 6   | `understand` (MAP) | LLM reads transcript + keyframes → beat sheet                           | **LLM** |
| 7   | `script` (REDUCE)  | LLM writes the 解說 narration grounded to `clip_ref`s                     | **LLM** |
| 8   | `budget`           | fit the script to the target runtime (spoken-rate model)                | CPU     |
| 9   | `tts`              | synthesize the voiceover with a **cloned voice** (CosyVoice2/IndexTTS2) | **GPU** |
| 10  | `edl`              | build an audio-driven edit decision list                                | CPU     |
| 11  | `subs`             | timed, width-fit subtitle pieces (ASS)                                  | CPU     |
| 12  | `render`           | ffmpeg: conform footage to the VO, burn subs, mix score bed             | CPU     |


Output preserves the source aspect ratio (portrait stays portrait, longest side capped at 1080)
and plays the film's own score/SFX as a bed under the narration (dialogue stripped via Demucs).

  
*The web UI — every upload runs the 12 stages with live per-stage timings; colors mark CPU / GPU / LLM work.*

### The load-bearing idea: `clip_ref` grounding

The LLM never invents timestamps. Every shot/transcript chunk gets a stable `clip_id` in a SQLite
index; the model emits `clip_refs[]` per narration line, and code resolves the real timecodes.
Editing is **audio-driven**: footage is conformed to the measured voiceover duration, never the
reverse — so the cut always matches what's being said.

## LLM choices

The script "brain" is any **OpenAI-compatible** chat model — only `LLM_BASE_URL` + `LLM_MODEL` change.

- **Default: MiniMax-M3** (`https://api.minimax.io/v1`) — a vision-capable reasoning model that
accepts up to ~200 keyframe images per request, so it can actually *see* the footage it narrates.
- **MAP vs REDUCE thinking budget:** the `understand` (MAP) pass runs with reasoning **on**
(`thinking_map = "adaptive"`) to grasp the plot; the `script` (REDUCE) pass runs with reasoning
**off** (`thinking_reduce = "disabled"`) so the full 32K output budget goes to narration instead
of being eaten by `<think>` tokens.
- **Swappable:** point it at **Anthropic Claude**, **OpenAI**, or a **self-hosted vLLM** endpoint by
changing `LLM_BASE_URL`/`LLM_MODEL` (set `vision = false` in `config/pipeline.toml` for text-only models).

Speech models run on the GPU box: **WhisperX** (ASR, `large-v3`) and **CosyVoice2** (TTS, zero-shot
voice cloning; IndexTTS2 optional). Diarization uses **pyannote** (`HF_TOKEN`).

## System architecture

Yapper runs two ways from one codebase:

**1. CLI (single machine).** The Mac orchestrates and renders (Python + ffmpeg + PySceneDetect),
calls the GPU box's ASR/TTS gRPC services over an SSH tunnel, and calls the LLM API. The source
film never leaves the Mac — only audio, keyframes, text, and voiceover WAVs cross the network.

**2. Hosted web platform** ([yapper.yuchia.dev](https://yapper.yuchia.dev)) — the `yap` core wrapped
in a multi-user, no-sign-in web service:

```
Browser ─cookie─▶ Cloudflare (edge TLS) ─▶ Traefik ─▶ FastAPI (api)        artifacts
                                                         │ enqueue          ▲ presigned PUT/GET
                                                         ▼                  │  prod: AWS S3
                                              Redis (broker + $ budget) ─▶ Celery workers
                                                         │                  │  local: MinIO
                                              Postgres (sessions/movies/runs)
                                                         │ gRPC over an SSH-tunnel sidecar
                                                         ▼
                              GPU box (nlp-gpu-01, 6× RTX 3090): gpud leases ASR/TTS on demand
```

- **Cluster:** single-node **k3s** on a `t4g.large` (ARM64) EC2; one collapsed Celery worker drains
all queues (`cpu,asr,tts,llm,render`). Local dev runs the same images via Docker Compose.
- **On-demand GPU:** a supervisor (`gpud`) on the shared GPU box leases ASR/TTS instances per task and
releases them when idle, so the box isn't pinned. The cluster reaches it through an SSH-forward sidecar.
- **Artifacts:** per-session objects in S3 (prod) / MinIO (local) via presigned URLs; the source video
never transits the app servers.
- **Observability:** Prometheus + **Grafana** + Loki — see the
[Yapper overview dashboard](https://grafana.yuchia.dev/d/yapper-overview/yapper-c2b7-overview?orgId=1&from=now-24h&to=now&timezone=America%2FLos_Angeles&var-datasource=prometheus&refresh=auto)
(per-stage timings, LLM tokens/cost, GPU lease activity).

## Repository layout

```
yapper/          # CLI + pipeline core (the 12 stages, ffmpeg graph, LLM/TTS/ASR clients)
yapper_web/      # FastAPI + Celery web platform (api.py, tasks.py, db.py, static/index.html)
yapper_rpc/      # generated gRPC stubs (ride PYTHONPATH; shared by web + GPU server)
server/          # GPU-box gRPC services (WhisperX ASR, CosyVoice2 TTS) + gpud supervisor
deploy/          # Dockerfile, docker-compose overlays, k8s (kustomize), Terraform, observability
config/          # pipeline.toml — all tunables
scripts/         # setup_check, run_local_compose, build_and_push, deploy, tf, ...
```

## Setup & usage

### A. CLI (local pipeline)

```bash
uv sync
cp .env.example .env          # fill LLM_API_KEY (+ optionally LLM_BASE_URL/LLM_MODEL),
                              #      ASR_GRPC_TARGET / TTS_GRPC_TARGET, HF_TOKEN
bash scripts/setup_check.sh   # checks ffmpeg+libass, CJK font, disk, GPU-server reachability
```

```bash
uv run yap run /path/to/movie.mkv                 # full pipeline → final .mp4
uv run yap run /path/to/clip.mp4 --lang en        # English commentary
uv run yap run /path/to/movie.mkv --until shots   # stop after a stage (resumable)
```

The CLI dials **fixed** ASR/TTS endpoints (e.g. over `ssh -N -L 50051:localhost:50051 -L 50052:localhost:50052 nlp`).
On-demand `gpud` leasing is a platform feature, not used by the CLI.

### B. Local web platform (Docker Compose)

```bash
bash scripts/run_local_compose.sh        # build once + start the whole stack
# app:           http://localhost:8080
# MinIO console: http://localhost:9001   (minioadmin / minioadmin)
```

Details, profiles, and gotchas: **[deploy/README.md](deploy/README.md)**.

### C. Production deploy (k3s + Terraform)

```bash
bash scripts/deploy.sh                    # build+push the image, then terraform apply (rolling update)
```

`deploy.sh` builds `yuchia329/yapper:<git-sha>`, pushes to Docker Hub, and applies the kustomize
overlay via Terraform (which also manages S3, the Cloudflare DNS record, and secrets). The Terraform
wrapper auto-manages the k3s API SSH tunnel. See **[deploy/terraform/README.md](deploy/terraform/README.md)**.

GPU-box bring-up (WhisperX + CosyVoice2 + gpud) is documented separately in
**[server/README_deploy.md](server/README_deploy.md)**.

## Configuration

All tunables live in `**[config/pipeline.toml](config/pipeline.toml)`** — LLM provider/model and
thinking budgets, ASR model, scene-detection thresholds, keyframe limits, TTS voice/reference,
narration target runtime, render preset/CRF, and subtitle styling. Code reads this file; nothing is
hard-coded.

## License

Personal project — see [github.com/yuchia329/movie_narrative](https://github.com/yuchia329/movie_narrative).