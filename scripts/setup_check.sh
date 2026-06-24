#!/usr/bin/env bash
# Preflight: verify the host can run the pipeline. Non-fatal — prints status.
set -uo pipefail

ok()   { printf "  \033[32m✓\033[0m %s\n" "$1"; }
bad()  { printf "  \033[31m✗\033[0m %s\n" "$1"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$1"; }

# load .env so secret/URL checks reflect the real config
[ -f .env ] && { set -a; . ./.env; set +a; }

echo "== ffmpeg =="
# resolve the same way yapper.ffmpeg.run does: FFMPEG_BIN -> vendored -> PATH
FF="${FFMPEG_BIN:-}"
[ -z "$FF" ] && [ -x vendor/ffmpeg/ffmpeg ] && FF=vendor/ffmpeg/ffmpeg
[ -z "$FF" ] && FF="$(command -v ffmpeg 2>/dev/null)"
if [ -n "$FF" ]; then
  ok "ffmpeg: $("$FF" -version | head -1 | awk '{print $3}')  ($FF)"
  if "$FF" -hide_banner -filters 2>/dev/null | awk '{print $2}' | grep -qx subtitles; then
    ok "libass subtitles filter present (CJK burn-in works)"
  else
    bad "no 'subtitles' filter — this ffmpeg lacks libass; subtitle burn-in (s11/s12) will be SKIPPED."
    warn "fix: vendored static build at vendor/ffmpeg/ffmpeg (re-download from ffmpeg.martin-riedl.de), or set FFMPEG_BIN."
  fi
else
  bad "ffmpeg not found"
fi

echo "== CJK font =="
if compgen -G "config/fonts/*.otf" >/dev/null || compgen -G "config/fonts/*.ttf" >/dev/null || compgen -G "config/fonts/*.ttc" >/dev/null; then
  ok "bundled CJK font in config/fonts/ ($(basename "$(compgen -G 'config/fonts/*.otf' 'config/fonts/*.ttf' 'config/fonts/*.ttc' | head -1)")"
elif fc-list 2>/dev/null | grep -qiE "noto sans cjk|source han"; then
  ok "CJK font installed system-wide"
else
  bad "no Noto Sans CJK / Source Han font — subtitles would render as boxes."
  warn "fix: download Noto Sans CJK SC into config/fonts/"
fi

echo "== disk =="
avail=$(df -g . 2>/dev/null | awk 'NR==2{print $4}')
if [ -n "${avail:-}" ]; then
  if [ "$avail" -ge 40 ]; then ok "${avail} GiB free"; else
    bad "only ${avail} GiB free — a feature film + normalized segments needs ~30-80 GiB."
    warn "fix: free space or point [paths] artifacts_dir/scratch_dir in config/pipeline.toml at an external volume."
  fi
fi

echo "== python / uv =="
command -v uv >/dev/null && ok "uv: $(uv --version)" || bad "uv not found"

echo "== secrets / services =="
[ -n "${LLM_API_KEY:-}" ] && ok "LLM_API_KEY set (script brain)" || warn "LLM_API_KEY not set (needed for script generation)"
# gRPC targets (host:port). Probe a TCP connect; use grpcurl for a real health check if present.
for var in ASR_GRPC_TARGET TTS_GRPC_TARGET; do
  target="${!var:-}"
  if [ -z "$target" ]; then warn "$var not set"; continue; fi
  host="${target%%:*}"; port="${target##*:}"
  if command -v grpcurl >/dev/null 2>&1; then
    if grpcurl -plaintext -max-time 4 "$target" grpc.health.v1.Health/Check >/dev/null 2>&1; then
      ok "$var serving ($target)"; else bad "$var not serving ($target)"; fi
  elif (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; then
    ok "$var TCP reachable ($target) — install grpcurl for a real health check"
  else
    bad "$var unreachable ($target)"
  fi
done

echo "done."
