"""Checks the Claude request shape and response parsing against a fake HTTP transport (no API key needed)."""

import json

import anthropic
import httpx2 as httpx
import pytest

from core import llm
from core.schemas import JDProfile

JD = {"title": "Backend Engineer", "seniority": "mid", "required_skills": ["Python"],
      "preferred_skills": [], "min_years_experience": 2, "keywords": []}


def fake_client(captured, stop_reason="end_turn"):
    def handler(request: httpx.Request):
        captured.append({"url": str(request.url), "headers": dict(request.headers), "body": json.loads(request.content)})
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": captured[-1]["body"]["model"],
            "content": [{"type": "text", "text": json.dumps(JD)}],
            "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 10},
        })
    return anthropic.Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_haiku_structured_output(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "_client", fake_client(calls))
    out = llm.parse("claude-haiku-5-5", "sys", "jd text", JDProfile)
    assert out.required_skills == ["Python"]
    body = calls[0]["body"]
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["effort"] == "low"
    assert "fallbacks" not in body and "temperature" not in body


def test_sonnet_uses_refusal_fallbacks(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "_client", fake_client(calls))
    llm.parse("claude-sonnet-5-5", "sys", "jd text", JDProfile)
    assert calls[0]["body"]["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in calls[0]["headers"]["anthropic-beta"]


def test_refusal_raises(monkeypatch):
    monkeypatch.setattr(llm, "_client", fake_client([], stop_reason="refusal"))
    with pytest.raises(llm.LLMError):
        llm.parse("claude-haiku-5-5", "sys", "x", JDProfile)
