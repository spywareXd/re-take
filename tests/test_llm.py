from pipeline.step_06_llm_adjudication.llm_client import endpoint_label, extract_usage

NONE_USAGE = {"prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None, "total_tokens": None}


def test_openai_format_usage():
    data = {"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160,
                                     "completion_tokens_details": {"reasoning_tokens": 25}}}
    assert extract_usage(data) == {"prompt_tokens": 120, "completion_tokens": 40, "reasoning_tokens": 25, "total_tokens": 160}


def test_openai_format_without_reasoning_details():
    data = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    assert extract_usage(data) == {"prompt_tokens": 10, "completion_tokens": 5, "reasoning_tokens": None, "total_tokens": 15}


def test_gemini_format_adds_thinking_to_completion():
    data = {"candidates": [], "usageMetadata": {"promptTokenCount": 200, "candidatesTokenCount": 30, "thoughtsTokenCount": 70, "totalTokenCount": 300}}
    assert extract_usage(data) == {"prompt_tokens": 200, "completion_tokens": 100, "reasoning_tokens": 70, "total_tokens": 300}


def test_gemini_format_without_thinking():
    data = {"usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 7, "totalTokenCount": 12}}
    assert extract_usage(data) == {"prompt_tokens": 5, "completion_tokens": 7, "reasoning_tokens": None, "total_tokens": 12}


def test_missing_usage_is_none_not_zero():
    assert extract_usage({"choices": []}) == NONE_USAGE
    assert extract_usage({"usage": {}}) == NONE_USAGE
    assert extract_usage({"usageMetadata": {}}) == NONE_USAGE
    assert extract_usage({"error": {"message": "x"}}) == NONE_USAGE
    assert extract_usage(None) == NONE_USAGE
    assert extract_usage("not a dict") == NONE_USAGE


def test_zero_reported_by_api_stays_zero():
    data = {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
    assert extract_usage(data)["prompt_tokens"] == 0


def test_non_integer_values_are_none():
    assert extract_usage({"usage": {"prompt_tokens": "12", "completion_tokens": True}})["prompt_tokens"] is None
    assert extract_usage({"usage": {"prompt_tokens": "12", "completion_tokens": True}})["completion_tokens"] is None


def test_endpoint_label():
    assert endpoint_label("https://api.groq.com/openai/v1/chat/completions") == "groq-openai-compat"
    assert endpoint_label("https://generativelanguage.googleapis.com/v1beta/models/x:generateContent?key=abc") == "gemini-generativelanguage"
    assert endpoint_label("https://api.openai.com/v1/chat/completions") == "openai"
    assert endpoint_label("http://localhost:11434") == "unknown"
