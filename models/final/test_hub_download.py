"""The scorer fetches missing cross-encoder weights from the Hub once, and never when offline."""
import json
import sys
import types

from final_scorer import FinalScorer


def _dir(tmp_path):
    (tmp_path / "thresholds.json").write_text(json.dumps({"models": {"cross_encoder_weights": ["ce_seed0", "ce_seed1"]}}), encoding="utf-8")


def _fake_hub(monkeypatch, calls):
    mod = types.ModuleType("huggingface_hub")
    mod.snapshot_download = lambda repo, **kw: calls.append((repo, kw))
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)


def test_missing_weights_are_downloaded_once_from_the_published_repo(tmp_path, monkeypatch):
    _dir(tmp_path)
    calls = []
    _fake_hub(monkeypatch, calls)
    monkeypatch.delenv("RETAKE_OFFLINE", raising=False)
    s = FinalScorer.__new__(FinalScorer)
    s.model_dir = tmp_path
    s._fetch_weights_if_missing()
    assert calls == [("spywarex/retake-pair-scorer", {"local_dir": str(tmp_path), "allow_patterns": ["ce_seed0/*", "ce_seed1/*"]})]


def test_nothing_is_downloaded_when_offline_or_when_the_weights_are_there(tmp_path, monkeypatch):
    _dir(tmp_path)
    calls = []
    _fake_hub(monkeypatch, calls)
    s = FinalScorer.__new__(FinalScorer)
    s.model_dir = tmp_path
    monkeypatch.setenv("RETAKE_OFFLINE", "1")
    s._fetch_weights_if_missing()
    monkeypatch.delenv("RETAKE_OFFLINE")
    for d in ("ce_seed0", "ce_seed1"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "config.json").write_text("{}", encoding="utf-8")
    s._fetch_weights_if_missing()
    assert calls == []


def test_a_failed_download_only_logs_and_does_not_raise(tmp_path, monkeypatch, caplog):
    _dir(tmp_path)
    mod = types.ModuleType("huggingface_hub")
    def boom(*a, **k):
        raise OSError("no network")
    mod.snapshot_download = boom
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    monkeypatch.delenv("RETAKE_OFFLINE", raising=False)
    s = FinalScorer.__new__(FinalScorer)
    s.model_dir = tmp_path
    s._fetch_weights_if_missing()
    assert "could not download" in caplog.text
