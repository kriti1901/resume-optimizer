"""Checks the LangChain chains against a fake Gemini HTTP transport (no API key needed)."""

import json

import httpx
import pytest
from langchain_google_genai import ChatGoogleGenerativeAI

from core import extractor, llm, rewriter
from core.schemas import JDProfile, RewriteResult

JD = {"title": "Backend Engineer", "seniority": "mid", "required_skills": ["Python"],
      "preferred_skills": [], "min_years_experience": 2, "keywords": []}
REWRITE = {"suggestions": [{"original": "a", "rewritten": "b", "reason": "c", "keywords_added": ["x"]}],
           "summary_tip": "t"}


def fake_llm(captured, payload=JD, finish="STOP", status=200):
    def handler(request: httpx.Request):
        captured.append({"url": str(request.url), "body": json.loads(request.content)})
        if status != 200:
            return httpx.Response(status, json={"error": {"code": status, "message": "boom", "status": "ERR"}})
        return httpx.Response(200, json={
            "candidates": [{"content": {"role": "model", "parts": [{"text": json.dumps(payload)}]}, "finishReason": finish}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
        })
    return ChatGoogleGenerativeAI(model=llm.MODEL, google_api_key="test", max_retries=0,
                                  client_args={"transport": httpx.MockTransport(handler)})


def test_extract_jd_chain(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "_llm", fake_llm(calls))
    out = extractor.extract_jd("We need Python")
    assert isinstance(out, JDProfile) and out.required_skills == ["Python"]
    body = calls[0]["body"]
    assert llm.MODEL in calls[0]["url"]
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert "required_skills" in json.dumps(body["generationConfig"])
    assert "We need Python" in body["contents"][0]["parts"][0]["text"]
    assert "hiring requirements" in body["systemInstruction"]["parts"][0]["text"]


def test_resume_chain_strips_pii(monkeypatch, resume_text):
    calls = []
    monkeypatch.setattr(llm, "_llm", fake_llm(calls))
    with pytest.raises(llm.LLMError):  # fake returns JD-shaped JSON, so parsing fails
        extractor.extract_resume(resume_text)
    sent = calls[0]["body"]["contents"][0]["parts"][0]["text"]
    assert "priya.sharma@example.com" not in sent and "[EMAIL]" in sent


def test_rewrite_chain(monkeypatch, resume_text, resume_profile, jd_profile):
    calls = []
    monkeypatch.setattr(llm, "_llm", fake_llm(calls, payload=REWRITE))
    out = rewriter.suggest_rewrites(resume_text, resume_profile, jd_profile, [])
    assert isinstance(out, RewriteResult) and out.summary_tip == "t"
    assert "Kubernetes" in calls[0]["body"]["contents"][0]["parts"][0]["text"]


def test_max_tokens_raises(monkeypatch):
    monkeypatch.setattr(llm, "_llm", fake_llm([], finish="MAX_TOKENS"))
    with pytest.raises(llm.LLMError, match="cut off"):
        extractor.extract_jd("x")


def test_safety_block_raises(monkeypatch):
    monkeypatch.setattr(llm, "_llm", fake_llm([], finish="SAFETY"))
    with pytest.raises(llm.LLMError):
        extractor.extract_jd("x")


def test_rate_limit_message(monkeypatch):
    monkeypatch.setattr(llm, "_llm", fake_llm([], status=429))
    with pytest.raises(llm.LLMError, match="quota"):
        extractor.extract_jd("x")


def test_missing_key(monkeypatch):
    monkeypatch.setattr(llm, "_llm", None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(llm.LLMError, match="GEMINI_API_KEY"):
        llm.get_llm()
