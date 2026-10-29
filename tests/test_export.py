"""Item 4: export options math, VFR detection, unique filenames, user presets, text exports with remapped timings, and REAL ffmpeg exports for every preset category
verified with ffprobe (duration, resolution, fps, CFR/VFR). Tests that need ffmpeg say SKIPPED when it is missing; that is not a pass."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from models.final.final_scorer import FinalScorer
from pipeline.step_06_llm_adjudication.llm_client import MockLLMClient
from service import export_media as xm
from service import export_options as xo
from service.app import Deps, create_app
from service.config import ServiceConfig
from service.events import read_log
from service.project_store import ProjectStore

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="SKIPPED: ffmpeg/ffprobe not found")


# ---------------------------------------------------------------- resolution and fps math (no ffmpeg)
@pytest.mark.parametrize("preset,w,h", [("2160p", 3840, 2160), ("1440p", 2560, 1440), ("1080p", 1920, 1080), ("720p", 1280, 720), ("480p", 854, 480)])
def test_height_presets_keep_the_source_aspect_and_are_even(preset, w, h):
    p = xo.plan_resolution(1920, 1080, {"mode": "preset", "preset": preset})
    assert (p.width, p.height, p.fit) == (w, h, "scale")


def test_source_and_odd_source_and_non_16_9_sources():
    assert xo.plan_resolution(1280, 720, None).width == 1280
    p = xo.plan_resolution(1279, 721, {"mode": "source"})
    assert (p.width, p.height) == (1280, 722) and p.notes                          # odd source -> even output, said so
    p = xo.plan_resolution(1440, 1080, {"mode": "preset", "preset": "720p"})      # 4:3
    assert (p.width, p.height) == (960, 720)
    assert xo.plan_resolution(1080, 1920, {"mode": "preset", "preset": "1080p"}).width == 608       # a vertical source keeps its shape (1080 * 1080/1920 = 607.5 -> 608)
    assert "Upscaling" in " ".join(xo.plan_resolution(1280, 720, {"mode": "preset", "preset": "2160p"}).notes)


def test_canvas_presets_and_fit_filters():
    v = xo.plan_resolution(1920, 1080, {"mode": "preset", "preset": "vertical_1080x1920"})
    assert (v.width, v.height, v.fit) == (1080, 1920, "crop")                       # "scaled to fill" like Filmcraft's Social Vertical
    assert "crop=1080:1920" in v.vf(1920, 1080)
    s = xo.plan_resolution(1920, 1080, {"mode": "preset", "preset": "square_1080x1080", "fit": "letterbox"})
    assert (s.width, s.height) == (1080, 1080) and "pad=1080:1080" in s.vf(1920, 1080)
    assert xo.plan_resolution(1920, 1080, {"mode": "preset", "preset": "square_1080x1080", "fit": "stretch"}).notes


def test_custom_size_validation_and_aspect_lock():
    for bad in ({"width": 1281, "height": 720}, {"width": 1280, "height": 721}, {"width": 8, "height": 720}, {"width": 9000, "height": 720}, {"width": "1280", "height": 720},
                {"width": 1280}, {}):
        with pytest.raises(xo.OptionError):
            xo.plan_resolution(1920, 1080, {"mode": "custom", **bad})
    c = xo.plan_resolution(1920, 1080, {"mode": "custom", "width": 1280, "height": 800})
    assert (c.width, c.height, c.fit) == (1280, 800, "letterbox")                   # no lock: exactly the size asked for, letterboxed by default
    assert xo.plan_resolution(1920, 1080, {"mode": "custom", "width": 1280, "height": 800, "fit": "crop"}).fit == "crop"
    l1 = xo.plan_resolution(1920, 1080, {"mode": "custom", "width": 1000, "lock_aspect": True})
    assert (l1.width, l1.height) == (1000, 562)                                      # 562.5 -> nearest even
    l2 = xo.plan_resolution(1920, 1080, {"mode": "custom", "height": 540, "lock_aspect": True})
    assert (l2.width, l2.height) == (960, 540)
    l3 = xo.plan_resolution(1920, 1080, {"mode": "custom", "width": 1000, "height": 1000, "lock_aspect": True})
    assert (l3.width, l3.height) == (1000, 562)                                      # a bounding box: the source shape is kept inside it
    with pytest.raises(xo.OptionError):
        xo.plan_resolution(1920, 1080, {"mode": "preset", "preset": "999p"})
    with pytest.raises(xo.OptionError):
        xo.plan_resolution(1920, 1080, {"mode": "custom", "width": 1280, "height": 720, "fit": "zoom"})


def test_fps_plans():
    assert xo.plan_fps("source", False, 29.97).mode == "source" and xo.plan_fps(None, False, 25).fps == 25
    v = xo.plan_fps("source", True, 30)
    assert v.mode == "source" and v.fps is None and v.fps_mode_arg == "passthrough" and "Variable frame rate is preserved" in v.notes[0]
    for f in xo.FPS_CHOICES + (23.976, 144):
        p = xo.plan_fps(f, False, 30)
        assert p.mode == "fixed" and p.filter == f"fps={float(f):g}" and p.fps_mode_arg == "cfr"
    assert "constant frame rate" in " ".join(xo.plan_fps(24, True, 30).notes) and "duplicated or dropped" in " ".join(xo.plan_fps(24, True, 30).notes)
    for bad in (0, -5, 0.5, 241, float("nan"), float("inf"), "30", True):
        with pytest.raises(xo.OptionError):
            xo.plan_fps(bad, False, 30)


def test_video_audio_option_validation():
    nv = {"h264": False, "h265": False}
    assert "-crf" in xo.plan_video({"quality": {"preset": "high"}}, nv).args and xo.plan_video(None, nv).quality == "CRF 23"
    assert xo.plan_video({"codec": "h265", "quality": {"preset": "high"}}, nv).quality == "CRF 23" and "hvc1" in xo.plan_video({"codec": "h265"}, nv).args
    assert xo.plan_video({"quality": {"bitrate_kbps": 4000}}, nv).args[:4] == ("-c:v", "libx264", "-preset", "medium") and "-b:v" in xo.plan_video({"quality": {"bitrate_kbps": 4000}}, nv).args
    for bad in ({"codec": "vp9"}, {"quality": {"crf": 99}}, {"quality": {"crf": True}}, {"quality": {"bitrate_kbps": 5}}, {"quality": {"preset": "ultra"}}, {"speed": "warp"}):
        with pytest.raises(xo.OptionError):
            xo.plan_video(bad, nv)
    with pytest.raises(xo.OptionError, match="NVENC"):
        xo.plan_video({"hardware": True}, nv)
    assert xo.plan_video({"hardware": True, "codec": "h265"}, {"h264": True, "h265": True}).encoder == "hevc_nvenc"
    assert xo.plan_audio(None) == xo.AudioPlan("aac", 192, False, 15.0)              # loudness normalisation OFF by default; fade = the pipeline's 15 ms
    assert xo.plan_audio({"normalize_loudness": True, "bitrate_kbps": 128}).loudnorm is True
    for bad in ({"bitrate_kbps": 100}, {"fade_ms": -1}, {"fade_ms": 500}):
        with pytest.raises(xo.OptionError):
            xo.plan_audio(bad)


def test_cut_math_and_time_remap():
    assert xm.keep_from_cuts([(1.0, 2.0), (4.0, 4.5)], 6.0) == [(0.0, 1.0), (2.0, 4.0), (4.5, 6.0)]
    assert xm.keep_from_cuts([(0, 1), (0.5, 2)], 6.0) == [(2.0, 6.0)]                # overlapping cuts merge
    assert xm.keep_from_cuts([], 6.0) == [(0.0, 6.0)] and xm.keep_from_cuts([(0, 6)], 6.0) == []
    assert xm.keep_from_cuts([(-3, 1), (5, 99)], 6.0) == [(1.0, 5.0)]                # clamped into the file
    for bad in ([(2, 1)], [(float("nan"), 1)]):
        with pytest.raises(xo.OptionError):
            xm.keep_from_cuts(bad, 6.0)
    keeps = [(0.0, 1.0), (2.0, 4.0), (4.5, 6.0)]
    assert xm.output_duration(keeps) == 4.5
    assert xm.remap_time(0.5, keeps) == 0.5 and xm.remap_time(2.5, keeps) == 1.5 and xm.remap_time(5.0, keeps) == 3.5 and xm.remap_time(1.5, keeps) is None
    with pytest.raises(xo.OptionError):
        xm.normalise_keeps([(0, 3), (2, 4)], 6.0)


def test_text_formats_remap_timings_onto_the_cut_timeline():
    units = [{"unit_id": "u1", "source_segment_id": 1, "start": 0.0, "end": 2.0}, {"unit_id": "u2", "source_segment_id": 1, "start": 4.0, "end": 6.0},
             {"unit_id": "u3", "source_segment_id": 2, "start": 2.5, "end": 3.5}]
    words = [{"word": "hello", "start": 0.0, "end": 0.9, "segment_id": 1}, {"word": "world", "start": 1.0, "end": 2.0, "segment_id": 1},
             {"word": "dropped", "start": 2.5, "end": 3.0, "segment_id": 2}, {"word": "again", "start": 3.0, "end": 3.5, "segment_id": 2},
             {"word": "second", "start": 4.0, "end": 5.0, "segment_id": 1}, {"word": "take", "start": 5.0, "end": 6.0, "segment_id": 1}]
    keeps = [(0.0, 2.0), (4.0, 6.0)]                                                   # u3 (2.5-3.5) is cut
    cues = xm.kept_cues(units, words, keeps)
    assert cues == [(0.0, 2.0, "hello world"), (2.0, 4.0, "second take")]            # the second sentence moved from 4-6 s to 2-4 s; the cut one is gone
    assert xm.to_srt(cues) == "1\n00:00:00,000 --> 00:00:02,000\nhello world\n\n2\n00:00:02,000 --> 00:00:04,000\nsecond take\n\n"
    assert xm.to_vtt(cues).startswith("WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nhello world")
    edl = xm.to_edl(keeps, "a.mp4", 25.0, "T")
    assert "001  AX       AA/V  C        00:00:00:00 00:00:02:00 00:00:00:00 00:00:02:00" in edl
    assert "002  AX       AA/V  C        00:00:04:00 00:00:06:00 00:00:02:00 00:00:04:00" in edl
    assert "29.970" in xm.to_edl(keeps, "a.mp4", 29.97, "T") and "approximate" in xm.to_edl(keeps, "a.mp4", 29.97, "T")
    cj = xm.cutlist_json("a.mp4", 6.0, keeps, 25.0)
    assert cj["cut"] == [{"start": 2.0, "end": 4.0}] and cj["output_duration_s"] == 4.0 and cj["keep"][1] == {"start": 4.0, "end": 6.0}


def test_user_presets_live_in_the_user_config_not_the_project(tmp_path, monkeypatch):
    monkeypatch.setenv("RETAKE_USER_DIR", str(tmp_path / "user"))
    assert xo.presets_path() == tmp_path / "user" / "export_presets.json" and set(xo.all_presets()) == set(xo.BUILTIN_PRESETS)
    xo.save_user_preset("My 720p", {"resolution": {"mode": "preset", "preset": "720p"}, "fps": 24, "video": {"quality": {"crf": 24}}})
    assert xo.all_presets()["My 720p"]["builtin"] is False and xo.presets_path().is_file()
    assert xo.apply_preset("My 720p", {"fps": 60})["fps"] == 60 and xo.apply_preset("My 720p", {})["fps"] == 24         # a request can override the preset
    for bad_name in ("YouTube 1080p", "youtube 1080p", "", "a" * 61, "x\\y"):
        with pytest.raises(xo.OptionError):
            xo.save_user_preset(bad_name, {"fps": 30})
    with pytest.raises(xo.OptionError):
        xo.save_user_preset("Bad", {"fps": 1000})
    with pytest.raises(xo.OptionError):
        xo.save_user_preset("Empty", {})
    assert xo.delete_user_preset("My 720p") and not xo.delete_user_preset("My 720p") and "My 720p" not in xo.all_presets()


# ---------------------------------------------------------------- real ffmpeg
def _ff(*args):
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    if not HAVE_FFMPEG:
        pytest.skip("SKIPPED: ffmpeg/ffprobe not found")
    d = tmp_path_factory.mktemp("clips")
    cfr = d / "cfr.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=26", "-f", "lavfi", "-i", "sine=frequency=440:d=26", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(cfr))
    vfr = d / "vfr.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=12", "-f", "lavfi", "-i", "sine=frequency=440:d=12", "-vf", "select='lt(mod(n,10),5)'", "-fps_mode", "vfr",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(vfr))
    silent = d / "silent.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=4", "-c:v", "libx264", "-preset", "ultrafast", str(silent))
    odd = d / "odd.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=s=642x362:r=25:d=3", "-vf", "format=yuv444p", "-c:v", "libx264", "-preset", "ultrafast", str(odd))
    return {"cfr": cfr, "vfr": vfr, "silent": silent, "odd": odd}


@needs_ffmpeg
def test_vfr_detection_and_probe(clips):
    assert xm.detect_vfr(clips["cfr"]) is False
    assert xm.detect_vfr(clips["vfr"]) is True
    a, b = xm.probe_video(clips["cfr"]), xm.probe_video(clips["vfr"])
    assert (a.width, a.height, a.has_audio, a.vfr) == (640, 360, True, False) and round(a.fps) == 30 and abs(a.duration - 26) < 0.1
    assert b.vfr is True and xm.probe_video(clips["silent"]).has_audio is False


@pytest.fixture
def env(tmp_path, monkeypatch, clips):
    monkeypatch.setenv("RETAKE_USER_DIR", str(tmp_path / "user"))
    cfg = ServiceConfig(data_dir=str(tmp_path / "sd"), llm_mock=True, keeper_provider="gpt-oss")
    app = create_app(cfg, Deps(scorer=FinalScorer(mock=True), llm_client=MockLLMClient()))
    return TestClient(app), tmp_path / "Proj", clips


def _export(client, proj, src, expect_status=202, wait=True, **body):
    r = client.post("/v1/export", json={"project": str(proj), "path": str(src), **body})
    if r.status_code != expect_status or not wait or r.status_code != 202:
        return r, None
    job = r.json()["job"]
    end = time.time() + 240
    while time.time() < end:
        b = client.get(f"/v1/jobs/{job}").json()
        if b["state"] != "running":
            return r, {"job": job, **b}
        time.sleep(0.05)
    raise AssertionError("export did not finish")


KEEP = [[0.0, 2.0], [4.0, 5.5]]                    # 3.5 s out of the clip


def _check(res, w, h, fps, mode="CFR", dur=3.5):
    v = res["result"]["verify"]
    assert (v["width"], v["height"]) == (w, h), v
    assert abs(v["duration_s"] - dur) < 0.2, v
    if fps is not None:
        assert abs(v["fps"] - fps) < 0.05, v
    assert v["frame_rate_mode"] == mode, v


@needs_ffmpeg
@pytest.mark.parametrize("preset,w,h,fps", [("Match Source", 640, 360, 30.0), ("YouTube 1080p", 1920, 1080, 30.0), ("YouTube 4K", 3840, 2160, 30.0),
                                             ("Shorts / Reels / TikTok", 1080, 1920, 30.0), ("Draft / Preview", 854, 480, 30.0)])
def test_real_export_of_every_builtin_preset(env, preset, w, h, fps):
    client, proj, clips = env
    _, body = _export(client, proj, clips["cfr"], preset=preset, keep_spans=KEEP)
    assert body["state"] == "done", body
    _check(body, w, h, fps)
    out = Path(body["result"]["path"])
    assert out.parent == proj / "data" / "exports" and out.is_file() and body["result"]["verify"]["has_audio"] is True
    assert any("Output frame rate: CFR" in n for n in body["result"]["notes"])


@needs_ffmpeg
def test_real_export_resolutions_fps_codecs_and_quality(env):
    client, proj, clips = env
    cases = [
        ({"resolution": {"mode": "preset", "preset": "720p"}}, 1280, 720, 30.0),
        ({"resolution": {"mode": "preset", "preset": "480p"}, "fps": 24}, 854, 480, 24.0),
        ({"resolution": {"mode": "preset", "preset": "square_1080x1080", "fit": "letterbox"}, "fps": 25}, 1080, 1080, 25.0),
        ({"resolution": {"mode": "custom", "width": 800, "height": 600, "fit": "crop"}, "fps": 50}, 800, 600, 50.0),
        ({"resolution": {"mode": "custom", "width": 500, "lock_aspect": True}, "fps": 60}, 500, 282, 60.0),
        ({"video": {"codec": "h265", "quality": {"preset": "draft"}, "speed": "ultrafast"}, "resolution": {"mode": "preset", "preset": "480p"}}, 854, 480, 30.0),
        ({"video": {"codec": "h264", "quality": {"bitrate_kbps": 800}, "speed": "veryfast"}}, 640, 360, 30.0),
        ({"fps": 23.976, "resolution": {"mode": "preset", "preset": "480p"}}, 854, 480, 23.976),
    ]
    for opts, w, h, fps in cases:
        _, body = _export(client, proj, clips["cfr"], keep_spans=KEEP, **opts)
        assert body["state"] == "done", (opts, body)
        _check(body, w, h, fps)


@needs_ffmpeg
def test_codec_really_is_what_was_asked(env):
    client, proj, clips = env
    for codec, name in (("h265", "hevc"), ("h264", "h264")):
        _, body = _export(client, proj, clips["cfr"], keep_spans=KEEP, video={"codec": codec, "speed": "ultrafast"}, resolution={"mode": "preset", "preset": "480p"})
        got = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", body["result"]["path"]],
                             capture_output=True, text=True).stdout.strip()
        assert got == name


@needs_ffmpeg
def test_vfr_source_is_preserved_or_made_cfr_and_the_output_says_which(env):
    client, proj, clips = env
    keep = [[0.0, 5.0]]
    _, kept = _export(client, proj, clips["vfr"], keep_spans=keep, fps="source")
    assert kept["state"] == "done", kept
    assert kept["result"]["verify"]["frame_rate_mode"] == "VFR" and any("VFR" in n for n in kept["result"]["notes"]), kept["result"]
    _, cfr = _export(client, proj, clips["vfr"], keep_spans=keep, fps=30)
    assert cfr["state"] == "done" and cfr["result"]["verify"]["frame_rate_mode"] == "CFR" and abs(cfr["result"]["verify"]["fps"] - 30) < 0.05
    assert any("constant frame rate" in n for n in cfr["result"]["plan"]["notes"]) and any("duplicated or dropped" in n for n in cfr["result"]["plan"]["notes"])


@needs_ffmpeg
def test_odd_source_and_silent_source(env):
    client, proj, clips = env
    _, body = _export(client, proj, clips["odd"], preset="Match Source", keep_spans=[[0.0, 2.0]])
    assert body["state"] == "done", body
    assert (body["result"]["verify"]["width"], body["result"]["verify"]["height"]) == (642, 362)
    _, sil = _export(client, proj, clips["silent"], preset="Draft / Preview", keep_spans=[[0.0, 2.0]])
    assert sil["state"] == "done" and sil["result"]["verify"]["has_audio"] is False


@needs_ffmpeg
def test_audio_only_exports(env):
    client, proj, clips = env
    for fmt, codec in (("mp3", "mp3"), ("wav", "pcm_s16le")):
        _, body = _export(client, proj, clips["cfr"], kind="audio", format=fmt, keep_spans=KEEP)
        assert body["state"] == "done", body
        out = Path(body["result"]["path"])
        assert out.suffix == f".{fmt}"
        d = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_name,codec_type", "-of", "json", str(out)],
                                      capture_output=True, text=True).stdout)
        assert d["streams"][0]["codec_name"] == codec and all(s["codec_type"] == "audio" for s in d["streams"]) and abs(float(d["format"]["duration"]) - 3.5) < 0.2
    none, _ = _export(client, proj, clips["silent"], kind="audio", format="mp3", keep_spans=[[0, 2]], expect_status=400)
    assert none.status_code == 400 and "no audio" in none.json()["error"]


@needs_ffmpeg
def test_loudness_normalisation_is_off_by_default_and_changes_the_filter_when_on(env, clips):
    info = xm.probe_video(clips["cfr"])
    for on in (False, True):
        script = clips["cfr"].parent / f"s{on}.txt"
        args, summary = xm.build_render_command(info, [(0.0, 2.0)], clips["cfr"].parent / "o.mp4", script, {"audio": {"normalize_loudness": on}}, {"h264": False, "h265": False})
        assert ("loudnorm" in script.read_text()) is on and summary["loudnorm"] is on


@needs_ffmpeg
def test_fade_at_cuts_uses_the_pipelines_setting(env, clips):
    info = xm.probe_video(clips["cfr"])
    script = clips["cfr"].parent / "fade.txt"
    xm.build_render_command(info, [(0.0, 2.0), (4.0, 5.0)], clips["cfr"].parent / "o.mp4", script, {}, {"h264": False, "h265": False})
    assert "afade=t=in:st=0:d=0.0150" in script.read_text()                                   # CutConfig.fade_ms = 15 ms, the same filter the pipeline editor builds
    xm.build_render_command(info, [(0.0, 2.0), (4.0, 5.0)], clips["cfr"].parent / "o.mp4", script, {"audio": {"fade_ms": 0}}, {"h264": False, "h265": False})
    assert "afade" not in script.read_text()


@needs_ffmpeg
def test_export_names_never_overwrite_earlier_exports(env):
    client, proj, clips = env
    store = ProjectStore.create(proj)
    pre = store.exports_dir / "cfr_Draft _ Preview.mp4"
    pre.write_bytes(b"an earlier export")
    names = []
    for _ in range(3):
        _, body = _export(client, proj, clips["cfr"], preset="Draft / Preview", keep_spans=[[0.0, 1.0]])
        assert body["state"] == "done"
        names.append(Path(body["result"]["path"]).name)
    assert len(set(names)) == 3 and pre.read_bytes() == b"an earlier export" and Path(names[0]).name not in {pre.name}


@needs_ffmpeg
def test_ffmpeg_errors_are_shown_verbatim_and_leave_no_empty_file(env, monkeypatch):
    client, proj, clips = env
    real = xm.build_render_command

    def broken(*a, **k):
        args, summary = real(*a, **k)
        return args[:-1] + ["-bogus_option", "1", args[-1]], summary

    monkeypatch.setattr(xm, "build_render_command", broken)
    _, body = _export(client, proj, clips["cfr"], preset="Draft / Preview", keep_spans=[[0.0, 1.0]])
    assert body["state"] == "failed" and "bogus_option" not in body["error"]               # the job error shown to the editor stays plain
    ev = client.get(f"/v1/jobs/{body['job']}/events").json()["events"]
    err = [e for e in ev if e["status"] == "error"]
    assert len(err) == 1 and err[0]["stage"] == "export_render" and "Unrecognized option 'bogus_option'" in err[0]["message"]        # ffmpeg's own words
    assert any(e["message"] == err[0]["message"] for e in read_log(proj / "data" / "progress.jsonl"))
    assert not list((proj / "data" / "exports").glob("*"))                                  # the reserved file of the failed job was removed
    assert not [p for p in (proj / "data" / "exports").glob("_filter_*")]


@needs_ffmpeg
def test_export_reports_progress_through_the_event_stream(env):
    client, proj, clips = env
    _, body = _export(client, proj, clips["cfr"], preset="YouTube 1080p", keep_spans=[[0.0, 8.0]], video={"speed": "veryfast"})
    ev = [e for e in client.get(f"/v1/jobs/{body['job']}/events").json()["events"] if e["stage"] == "export_render"]
    assert ev[0]["status"] == "started" and ev[-1]["status"] == "done" and ev[-1]["pct"] == 100.0
    pcts = [e["pct"] for e in ev if e["status"] == "progress"]
    assert pcts and pcts == sorted(pcts) and all(0 <= p <= 100 for p in pcts)


@needs_ffmpeg
def test_invalid_options_are_400s_before_any_work(env):
    client, proj, clips = env
    bad = [dict(resolution={"mode": "custom", "width": 1281, "height": 720}), dict(fps=0), dict(video={"codec": "vp9"}), dict(preset="Nope"),
           dict(video={"hardware": True, "codec": "h264"}, _nv=False), dict(kind="gif"), dict(kind="subtitles", format="ass"), dict(kind="audio", format="flac"),
           dict(keep_spans=[[3, 1]]), dict(cut_ranges=[[0, 100]], keep_spans=None)]
    for b in bad:
        nv = b.pop("_nv", None)
        if nv is False:
            import service.export_media as m
            orig = m.nvenc_support
            m.nvenc_support = lambda: {"h264": False, "h265": False}
        try:
            r, _ = _export(client, proj, clips["cfr"], expect_status=400, **({"keep_spans": KEEP} | b))
        finally:
            if nv is False:
                m.nvenc_support = orig
        assert r.status_code == 400 and r.json()["error"], b
    assert not (proj / "data" / "exports").exists() or not list((proj / "data" / "exports").glob("*"))
    assert client.post("/v1/export", json={"path": str(clips["cfr"])}).status_code == 400                    # no project: exports go into a project folder


@needs_ffmpeg
def test_cut_ranges_and_the_saved_cutlist_drive_the_cut(env):
    client, proj, clips = env
    _, a = _export(client, proj, clips["cfr"], preset="Draft / Preview", cut_ranges=[{"start": 1.0, "end": 24.0}])
    assert abs(a["result"]["verify"]["duration_s"] - 3.0) < 0.2                          # 26 s minus the 23 s removed
    ProjectStore.create(proj).save_result("cutlist", {"ranges": [{"start": 2.0, "end": 26.0}]})
    _, b = _export(client, proj, clips["cfr"], preset="Draft / Preview")                  # no cut given: the project's saved cut list
    assert abs(b["result"]["verify"]["duration_s"] - 2.0) < 0.2


@needs_ffmpeg
def test_text_exports_through_the_api(env):
    client, proj, clips = env
    store = ProjectStore.create(proj)
    units = [{"unit_id": "u1", "source_segment_id": 1, "start": 0.0, "end": 2.0}, {"unit_id": "u2", "source_segment_id": 1, "start": 4.0, "end": 6.0}]
    words = [{"word": "hello", "start": 0.0, "end": 1.0, "segment_id": 1}, {"word": "world", "start": 1.0, "end": 2.0, "segment_id": 1},
             {"word": "second", "start": 4.0, "end": 5.0, "segment_id": 1}, {"word": "take", "start": 5.0, "end": 6.0, "segment_id": 1}]
    store.put("units", units)
    store.put("words", words)
    store.put("transcript", {"segments": [{"id": 1, "start": 0.0, "end": 6.0, "text": "hello world second take"}]})
    keep = [[0.0, 2.0], [4.0, 6.0]]
    _, srt = _export(client, proj, clips["cfr"], kind="subtitles", format="srt", keep_spans=keep)
    assert Path(srt["result"]["path"]).read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:02,000\nhello world\n\n2\n00:00:02,000 --> 00:00:04,000\nsecond take\n\n"
    _, vtt = _export(client, proj, clips["cfr"], kind="subtitles", format="vtt", keep_spans=keep)
    assert Path(vtt["result"]["path"]).read_text(encoding="utf-8").startswith("WEBVTT")
    _, edl = _export(client, proj, clips["cfr"], kind="cutlist", format="edl", keep_spans=keep)
    assert "TITLE: cfr" in Path(edl["result"]["path"]).read_text(encoding="utf-8")
    _, cj = _export(client, proj, clips["cfr"], kind="cutlist", format="json", keep_spans=keep)
    assert json.loads(Path(cj["result"]["path"]).read_text(encoding="utf-8"))["output_duration_s"] == 4.0
    _, tj = _export(client, proj, clips["cfr"], kind="transcript", format="json")
    assert json.loads(Path(tj["result"]["path"]).read_text(encoding="utf-8"))["segments"][0]["text"].startswith("hello")
    _, tt = _export(client, proj, clips["cfr"], kind="transcript", format="txt")
    assert Path(tt["result"]["path"]).read_text(encoding="utf-8") == "[00:00:00] hello world second take\n"
    store2 = ProjectStore.create(proj.parent / "Empty")
    r, _ = _export(client, proj.parent / "Empty", clips["cfr"], kind="subtitles", format="srt", keep_spans=keep, wait=False)
    job = r.json()["job"]
    time.sleep(0.5)
    assert client.get(f"/v1/jobs/{job}").json()["state"] == "failed" and not list(store2.exports_dir.glob("*"))


@needs_ffmpeg
def test_options_endpoint_hides_hardware_unless_a_test_encode_works(env, monkeypatch):
    client, proj, clips = env
    monkeypatch.setattr(xm, "nvenc_support", lambda: {"h264": False, "h265": False})
    o = client.get("/v1/export/options").json()
    assert o["video"]["hardware"] is None and "YouTube 1080p" in o["presets"] and o["audio"]["normalize_loudness_default"] is False
    monkeypatch.setattr(xm, "nvenc_support", lambda: {"h264": True, "h265": False})
    assert client.get("/v1/export/options").json()["video"]["hardware"] == {"h264": True}
    assert {"2160p", "1440p", "1080p", "720p", "480p"} == set(o["resolutions"]["heights"]) and o["fps"]["choices"] == [24, 25, 30, 50, 60]


@needs_ffmpeg
def test_preset_endpoints_save_in_user_config(env, tmp_path):
    client, proj, clips = env
    r = client.put("/v1/export/presets/My Draft", json={"resolution": {"mode": "preset", "preset": "480p"}, "fps": 24, "video": {"quality": {"crf": 30}, "speed": "ultrafast"}})
    assert r.status_code == 200 and (tmp_path / "user" / "export_presets.json").is_file() and not (proj / "data").exists()
    assert "My Draft" in client.get("/v1/export/presets").json()
    _, body = _export(client, proj, clips["cfr"], preset="My Draft", keep_spans=KEEP)
    _check(body, 854, 480, 24.0)
    assert client.put("/v1/export/presets/YouTube 4K", json={"fps": 30}).status_code == 400
    assert client.delete("/v1/export/presets/YouTube 4K").status_code == 400
    assert client.delete("/v1/export/presets/My Draft").status_code == 204 and "My Draft" not in client.get("/v1/export/presets").json()


@needs_ffmpeg
def test_nvenc_export_only_if_ffmpeg_reports_it_and_the_test_encode_works(env):
    client, proj, clips = env
    hw = xm.nvenc_support()
    if not hw["h264"]:
        r, _ = _export(client, proj, clips["cfr"], expect_status=400, keep_spans=KEEP, video={"hardware": True})
        assert r.status_code == 400
        pytest.skip("SKIPPED: no usable NVENC on this machine (the 400 above is the checked behaviour)")
    _, body = _export(client, proj, clips["cfr"], keep_spans=KEEP, video={"hardware": True, "codec": "h264", "quality": {"preset": "standard"}},
                      resolution={"mode": "preset", "preset": "720p"})
    assert body["state"] == "done", body
    assert body["result"]["plan"]["encoder"] == "h264_nvenc" and body["result"]["plan"]["hardware"] is True
    _check(body, 1280, 720, 30.0)


@needs_ffmpeg
def test_options_carry_the_preset_order_because_json_objects_have_none(env, tmp_path):
    client, proj, clips = env
    client.put("/v1/export/presets/My Draft", json={"fps": 24})
    o = client.get("/v1/export/options").json()
    assert o["preset_order"] == ["Match Source", "YouTube 1080p", "YouTube 4K", "Shorts / Reels / TikTok", "Draft / Preview", "My Draft"]
    assert set(o["preset_order"]) == set(o["presets"])
