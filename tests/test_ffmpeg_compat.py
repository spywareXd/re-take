"""The filter-graph file option is chosen from the installed ffmpeg: `-/filter_complex` from 7.0 on, `-filter_complex_script` before."""
import subprocess

import pytest

from pipeline.step_08_video_edit import ffmpeg_compat as fc


def _fake_ffmpeg(monkeypatch, version_text, help_text=""):
    def run(cmd, **kw):
        out = help_text if "-h" in cmd else version_text
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
    monkeypatch.setattr(fc.subprocess, "run", run)
    fc.uses_slash_option.cache_clear()


@pytest.fixture(autouse=True)
def _reset():
    yield
    fc.uses_slash_option.cache_clear()


@pytest.mark.parametrize("version,expected", [("ffmpeg version 6.1.1 Copyright", False), ("ffmpeg version 7.0 Copyright", True),
                                              ("ffmpeg version 8.0.1-essentials_build Copyright", True), ("ffmpeg version 9.0.2-essentials_build Copyright", True),
                                              ("ffmpeg version n7.1 Copyright", True)])
def test_the_option_follows_the_major_version(monkeypatch, version, expected):
    _fake_ffmpeg(monkeypatch, version)
    assert fc.uses_slash_option() is expected
    assert fc.filter_graph_file_args("g.txt") == (["-/filter_complex", "g.txt"] if expected else ["-filter_complex_script", "g.txt"])


def test_a_development_build_is_asked_which_option_it_has(monkeypatch):
    _fake_ffmpeg(monkeypatch, "ffmpeg version N-120000-gabcdef Copyright", help_text="-filter_complex_script filename")
    assert fc.uses_slash_option() is False
    _fake_ffmpeg(monkeypatch, "ffmpeg version N-120000-gabcdef Copyright", help_text="-filter_complex")
    assert fc.uses_slash_option() is True


def test_without_ffmpeg_the_old_option_is_the_default(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("ffmpeg")
    monkeypatch.setattr(fc.subprocess, "run", boom)
    fc.uses_slash_option.cache_clear()
    assert fc.filter_graph_file_args("g.txt")[0] == "-filter_complex_script"
