# JieshuoForge — 电影解说 pipeline

Turn a full-length movie into a ~10–12 minute Mandarin **narrated recap** in the
Douyin/Kuaishou 电影解说 style: condensed footage from the film plays under an
AI-generated voiceover that tells the story.

## How it works

A movie goes through 12 deterministic, resumable stages. Each stage reads and
writes one validated artifact, so you can inspect or re-run any step.

```
movie ─▶ probe ─▶ audio ─▶ ASR ─▶ shots ─▶ scenes+keyframes ─▶ clip_index (SQLite)
       ─▶ Claude MAP (beat sheet) ─▶ Claude REDUCE (解说 script) ─▶ runtime budget
       ─▶ TTS (cloned voice) ─▶ audio-driven EDL ─▶ subtitles ─▶ ffmpeg render ─▶ recap.mp4
```

### Topology
- **This Mac** orchestrates and renders (Python + ffmpeg + PySceneDetect).
- **GPU server (6× RTX 3090)** hosts ASR (WhisperX) and TTS (CosyVoice2/IndexTTS2)
  as HTTP services; the Mac calls them over an SSH tunnel.
- **Claude Opus 4.8 (API)** reads the transcript + keyframes and writes the Mandarin script.

The source film never leaves the Mac — only audio, keyframes, text, and voiceover
WAVs cross the network.

### The load-bearing idea: `clip_ref` grounding
The LLM never invents timestamps. Every shot/transcript chunk gets a stable
`clip_id` in a SQLite index; the model emits `clip_refs[]` per narration line, and
code resolves the real timecodes. Editing is **audio-driven**: footage is conformed
to the measured voiceover duration, never the reverse.

## Setup

```bash
uv sync
cp .env.example .env        # fill in ANTHROPIC_API_KEY, server URLs
```

Then check prerequisites (ffmpeg+libass, CJK font, disk, server reachability):

```bash
bash scripts/setup_check.sh
```

## Usage

```bash
uv run jieshuo run /path/to/movie.mkv          # full pipeline
uv run jieshuo run /path/to/movie.mkv --until shots   # stop after a stage
```

See `config/pipeline.toml` for all tunables. See the plan in
`~/.claude/plans/` for the full design and roadmap.
