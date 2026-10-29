"""Checks the Gemini request shape and response parsing against a fake HTTP transport (no API key needed)."""

import json

import httpx
import pytest
from google import genai
from google.genai import types

from core import llm
from core.schemas import JDProfile

JD = {"title": "Backend Engineer", "seniority": "mid", "required_skills": ["Python"],
      "preferred_skills": [], "min_years_experience": 2, "keywords": []}


def fake_client(captured, finish="STOP", status=200):
    def handler(request: httpx.Request):
        captured.append({"url": str(request.url), "body": json.loads(request.content)})
        if status != 200:
            return httpx.Response(status, json={"error": {"code": status, "message": "boom", "status": "ERR"}})
        return httpx.Response(200, json={"candidates": [{
            "content": {"role": "model", "parts": [{"text": json.dumps(JD)}]},
            "finishReason": finish,
        }]})
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return genai.Client(api_key="test", http_options=types.HttpOptions(
        httpx_client=http, retry_options=types.HttpRetryOptions(attempts=1)))


def test_structured_output(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "_client", fake_client(calls))
    out = llm.parse("sys", "jd text", JDProfile)
    assert isinstance(out, JDProfile) and out.required_skills == ["Python"]
    body = calls[0]["body"]
    assert llm.MODEL in calls[0]["url"]
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert "required_skills" in json.dumps(body["generationConfig"])
    assert body["systemInstruction"]["parts"][0]["text"] == "sys"


def test_max_tokens_raises(monkeypatch):
    monkeypatch.setattr(llm, "_client", fake_client([], finish="MAX_TOKENS"))
    with pytest.raises(llm.LLMError, match="cut off"):
        llm.parse("sys", "x", JDProfile)


def test_safety_block_raises(monkeypatch):
    monkeypatch.setattr(llm, "_client", fake_client([], finish="SAFETY"))
    with pytest.raises(llm.LLMError):
        llm.parse("sys", "x", JDProfile)


def test_rate_limit_message(monkeypatch):
    monkeypatch.setattr(llm, "_client", fake_client([], status=429))
    with pytest.raises(llm.LLMError, match="quota"):
        llm.parse("sys", "x", JDProfile)


def test_missing_key(monkeypatch):
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(llm.LLMError, match="GEMINI_API_KEY"):
        llm.client()
