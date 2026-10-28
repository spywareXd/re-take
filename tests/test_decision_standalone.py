"""Tests for the standalone decision-model loader (no GPU or weights needed except the slow smoke test, which skips when they are absent)."""
import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from pipeline.step_06_llm_adjudication import decision_standalone as ds  # noqa: E402


class FakeTokenizer:
    """One token per character: enough to check spans and truncation."""

    def __call__(self, text, add_special_tokens=False):
        class Out:
            input_ids = [ord(c) % 1000 for c in text]

        return Out()


def _questions():
    return {"keep_a": ds.normalize_question({"type": "noul", "instructions": "Keep a?"}), "keep_b": ds.normalize_question({"type": "noul", "instructions": "Keep b?"})}


def test_encode_record_spans_point_at_the_right_tokens():
    tok = FakeTokenizer()
    rec = ds.encode_record(tok, {"takes": ["x"]}, _questions())
    for q in rec.questions:
        a, b = q.question_span
        assert "".join(chr(t) if t < 128 else "?" for t in rec.input_ids[a:b]) == f"Keep {q.question_id[-1]}?"
        assert q.option_ids == ("true", "false") and len(q.option_spans) == 2


def test_encode_record_cuts_the_state_not_the_schema():
    tok = FakeTokenizer()
    short = ds.encode_record(tok, "s" * 10, _questions(), max_length=100000)
    cut = ds.encode_record(tok, "s" * 10000, _questions(), max_length=len(short.input_ids) + 5)
    assert len(cut.input_ids) == len(short.input_ids) + 5
    with pytest.raises(ValueError):
        ds.encode_record(tok, "s", _questions(), max_length=10)


def test_question_validation():
    with pytest.raises(ValueError):
        ds.normalize_question({"type": "choice"})
    with pytest.raises(ValueError):
        ds.normalize_question({"type": "noul", "criteria": {"maybe": "x"}})


def test_head_math_on_a_tiny_fixture():
    """Shapes, option order (true first) and determinism of the head on random weights; the real weights are covered by the parity run."""
    Head = ds._build_head_classes()
    torch.manual_seed(0)
    head = Head(hidden_size=16, width=8, routing_layers=1, layers=1, heads=2, feedforward=16).eval()
    rec = ds.encode_record(FakeTokenizer(), {"takes": ["x"]}, _questions())
    ids = torch.tensor([rec.input_ids])
    hidden = torch.randn(1, ids.shape[1], 16)
    emb = torch.randn(1000, 16)
    with torch.no_grad():
        first = head(hidden, ids, rec, emb)
        second = head(hidden, ids, rec, emb)
    assert len(first) == 2 and all(z.shape == (2,) for z in first)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert all(torch.isfinite(z).all() for z in first)


def test_score_converts_logits_to_p_yes(monkeypatch):
    m = ds.StandaloneDecisionModel(mode="4bit")
    m._lm = object()
    m._temperature = 1.0
    monkeypatch.setattr(m, "score_logits", lambda s, q: ({"keep_a": [0.0, 0.0], "keep_b": [100.0, 0.0]}, 5))
    p = m.score({}, {"keep_a": {}, "keep_b": {}})
    assert p["keep_a"] == pytest.approx(0.5) and p["keep_b"] == pytest.approx(1.0)


def test_bad_mode_and_missing_folder(tmp_path):
    with pytest.raises(ValueError):
        ds.StandaloneDecisionModel(mode="int3")
    m = ds.StandaloneDecisionModel(model_dir=tmp_path / "nope")
    ok, why = m.availability()
    assert not ok and "missing" in why
    with pytest.raises(ds.DecisionModelMissing):
        m.score({}, {"keep_a": {"type": "noul", "instructions": "x"}})


def test_unload_without_load_is_safe():
    m = ds.StandaloneDecisionModel()
    m.unload()
    assert not m.loaded


def test_module_does_not_import_unsloth():
    code = "import sys; import pipeline.step_06_llm_adjudication.decision_standalone as d; d._build_head_classes(); print(any(k == 'unsloth' or k.startswith('unsloth.') for k in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ds.REPO), check=True).stdout.strip()
    assert out == "False"
    source = ds.Path(ds.__file__).read_text(encoding="utf-8")
    assert "import unsloth" not in source and "from unsloth" not in source


@pytest.mark.skipif(not os.environ.get("STANDALONE_SLOW"), reason="slow: set STANDALONE_SLOW=1 (loads the 4-bit model on the GPU)")
def test_parity_smoke_against_reference_scores():
    """Needs the exported weights, a GPU and the (gitignored) reference scores; compares the first natural group with the original model's score."""
    priv = ds.REPO / "runs" / "standalone_export" / "private"
    m = ds.StandaloneDecisionModel(mode="4bit")
    ok, why = m.availability()
    if not ok or not (priv / "parity_set.jsonl").is_file() or not (priv / "reference_scores_run1.jsonl").is_file():
        pytest.skip(why if not ok else "no parity inputs")
    row = json.loads((priv / "parity_set.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ref = json.loads((priv / "reference_scores_run1.jsonl").read_text(encoding="utf-8").splitlines()[0])
    with m:
        got = m.score(row["state"], row["questions"])
    assert max(abs(a - b) for a, b in zip(got.values(), ref["p_yes"])) < 0.05
