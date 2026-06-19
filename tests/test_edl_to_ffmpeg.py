"""EDL build is audio-driven; the ffmpeg translation is deterministic and carries
the conform + ducking semantics."""

from pathlib import Path

from jieshuoforge import db
from jieshuoforge.ffmpeg import graph
from jieshuoforge.schemas import (
    Clip,
    Script,
    ScriptLine,
    Transcript,
    TranscriptSegment,
    VOLine,
    VOManifest,
)
from jieshuoforge.stages import s10_edl


def _setup():
    conn = db.connect(":memory:")
    db.write_clips(
        conn,
        [
            Clip(clip_id="clip_0000", scene_index=0, t_start=0.0, t_end=30.0, shot_indices=[0]),
            Clip(clip_id="clip_0001", scene_index=1, t_start=30.0, t_end=60.0, shot_indices=[1]),
        ],
    )
    script = Script(
        lines=[
            ScriptLine(line_id="l0", text="第一句", clip_refs=["clip_0000"], est_spoken_seconds=4.0),
            ScriptLine(line_id="l1", text="第二句", clip_refs=["clip_0001"], est_spoken_seconds=6.0),
        ]
    )
    vo = VOManifest(
        voice_id="v", seed=1,
        lines=[
            VOLine(line_id="l0", text="第一句", file="/tmp/l0.wav", measured_duration_sec=4.2),
            VOLine(line_id="l1", text="第二句", file="/tmp/l1.wav", measured_duration_sec=6.1),
        ],
    )
    return conn, script, vo


def test_one_segment_per_line_with_measured_durations():
    conn, script, vo = _setup()
    edl = s10_edl.run_stage(script, vo, conn, fps=30, width=1920, height=1080)
    assert len(edl.segments) == 2
    # screen time == MEASURED voiceover duration (audio-driven)
    assert edl.segments[0].vo_duration == 4.2
    assert edl.segments[1].vo_duration == 6.1
    assert abs(edl.total_duration - 10.3) < 1e-6


def test_segment_window_anchored_to_clip():
    conn, script, vo = _setup()
    edl = s10_edl.run_stage(script, vo, conn, fps=30, width=1920, height=1080)
    assert edl.segments[1].src_in == 30.0  # clip_0001 starts at 30s


def test_normalize_cmd_carries_conform_and_ducking():
    conn, script, vo = _setup()
    edl = s10_edl.run_stage(script, vo, conn, fps=30, width=1920, height=1080)
    cmd = graph.normalize_segment_cmd(
        "/movie.mkv", edl.segments[0], Path("/tmp/seg.mp4"),
        width=1920, height=1080, fps=30, vcodec="h264_videotoolbox",
        pix_fmt="yuv420p", audio_rate=48000, ducking={"threshold": 0.03, "ratio": 8, "attack": 5, "release": 300},
    )
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "sidechaincompress" in fc           # original audio ducked under VO
    assert "tpad=stop_mode=clone" in fc         # footage freeze-padded to VO length
    assert "amix" in fc                          # ducked original + VO mixed
    # output capped at the voiceover duration
    assert cmd[cmd.index("-t") + 1] == "4.200"


def test_narration_with_score_bed_mixes_stem_no_ducking():
    conn, script, vo = _setup()
    edl = s10_edl.run_stage(conn=conn, script=script, vo=vo, fps=30, width=1920, height=1080)
    cmd = graph.normalize_segment_cmd(
        "/movie.mkv", edl.segments[0], Path("/tmp/seg.mp4"),
        width=1920, height=1080, fps=30, vcodec="libx264", pix_fmt="yuv420p",
        audio_rate=48000, ducking={}, score_stem="/stem.wav", bed_gain_db=-14.0,
    )
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert cmd.count("-i") == 3                 # movie (video) + score stem (bed) + voiceover
    assert "/stem.wav" in cmd
    assert "sidechaincompress" not in fc        # dialogue already gone — no ducking needed
    assert "volume=" in fc                       # bed attenuated to sit under the VO
    assert "normalize=0" in fc                   # amix keeps the VO at full level


def test_multi_clip_line_splits_into_montage_segments():
    conn, script, vo = _setup()
    script.lines[1].clip_refs = ["clip_0000", "clip_0001"]  # l1 now references 2 scenes
    edl = s10_edl.run_stage(script, vo, conn, fps=30, width=1920, height=1080)
    # l0 (1 scene) -> 1 segment; l1 (2 scenes) -> 2 montage segments
    assert len(edl.segments) == 3
    l1 = [s for s in edl.segments if s.line_id == "l1"]
    assert len(l1) == 2
    assert l1[0].vo_in == 0.0 and l1[1].vo_in > 0.0          # voiceover sliced, no overlap
    assert abs(sum(s.vo_duration for s in l1) - 6.1) < 1e-3  # slices sum to the line's VO
    assert abs(edl.total_duration - 10.3) < 1e-3             # total screen time conserved


def test_concat_uses_stream_copy():
    cmd = graph.concat_cmd(Path("/tmp/list.txt"), Path("/tmp/out.mp4"))
    assert "-c" in cmd and cmd[cmd.index("-c") + 1] == "copy"


def test_playback_line_grounds_window_to_quote_transcript_span():
    conn, script, vo = _setup()
    # turn l1 into a raw-playback moment featuring a quote inside clip_0001 (30-60s)
    script.lines[1].kind = "playback"
    script.lines[1].clip_refs = ["clip_0001"]
    script.lines[1].quote = "I am Michael Jackson"
    script.lines[1].text = "我是迈克尔·杰克逊"
    tr = Transcript(
        language="en", source="asr",
        segments=[TranscriptSegment(start=42.0, end=49.0, text="I am Michael Jackson and you are Tito")],
    )
    edl = s10_edl.run_stage(
        script, vo, conn, fps=30, width=1920, height=1080,
        transcript=tr, min_window=3, max_window=12,
    )
    pb = [s for s in edl.segments if s.kind == "playback"]
    assert len(pb) == 1
    seg = pb[0]
    assert seg.vo_file is None
    assert 41.5 <= seg.src_in <= 42.5 and 48.5 <= seg.src_out <= 49.5  # the quote's span
    assert seg.subtitle_text == "我是迈克尔·杰克逊"


def test_playback_segment_plays_original_audio_without_ducking():
    conn, script, vo = _setup()
    script.lines[1].kind = "playback"
    script.lines[1].clip_refs = ["clip_0001"]
    script.lines[1].quote = "hello"
    tr = Transcript(language="en", source="asr",
                    segments=[TranscriptSegment(start=40.0, end=45.0, text="hello there")])
    edl = s10_edl.run_stage(script, vo, conn, fps=30, width=1920, height=1080, transcript=tr)
    seg = next(s for s in edl.segments if s.kind == "playback")
    cmd = graph.normalize_segment_cmd(
        "/movie.mkv", seg, Path("/tmp/pb.mp4"),
        width=1920, height=1080, fps=30, vcodec="libx264",
        pix_fmt="yuv420p", audio_rate=48000, ducking={}, target_lufs=-16.0,
    )
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "sidechaincompress" not in fc   # no ducking — original audio plays full
    assert "amix" not in fc                # no voiceover to mix in
    assert "loudnorm" in fc                # just level-matched
    assert cmd.count("-i") == 1            # only the movie, no VO input


def test_reused_clip_advances_window_to_avoid_duplicate_footage():
    conn, script, vo = _setup()
    # both lines reference the same scene; the second showing should not replay the first window
    script.lines[0].clip_refs = ["clip_0001"]
    script.lines[1].clip_refs = ["clip_0001"]
    edl = s10_edl.run_stage(script, vo, conn, fps=30, width=1920, height=1080)
    s0, s1 = edl.segments[0], edl.segments[1]
    assert s1.src_in >= s0.src_out - 0.05  # advanced past the first window
