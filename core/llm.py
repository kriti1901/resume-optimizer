"""Thin wrapper around the Google Gen AI SDK for structured (pydantic) outputs from Gemini."""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")

_client: genai.Client | None = None


class LLMError(RuntimeError):
    pass


def client() -> genai.Client:
    global _client
    if _client is None:
        if not os.getenv("GEMINI_API_KEY"):
            raise LLMError("GEMINI_API_KEY is not set. Add it to your .env file.")
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def parse(system: str, user: str, schema, max_tokens: int = 16000):
    """Call Gemini and return a validated instance of `schema`.

    max_tokens is generous because Gemini Pro's thinking tokens count toward the output limit.
    """
    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=schema,
        max_output_tokens=max_tokens,
    )
    try:
        response = client().models.generate_content(model=MODEL, contents=user, config=config)
    except errors.ClientError as e:
        if e.code in (401, 403) or "API key" in str(e):
            raise LLMError("Invalid GEMINI_API_KEY.") from e
        if e.code == 429:
            raise LLMError("Gemini rate limit or free-tier quota reached. Wait a minute or check your quota.") from e
        if e.code == 404:
            raise LLMError(f"Model '{MODEL}' not found. Set GEMINI_MODEL in .env to a current model.") from e
        raise LLMError(f"Gemini API error ({e.code}): {e.message}") from e
    except errors.ServerError as e:
        raise LLMError(f"Gemini server error ({e.code}). Try again shortly.") from e

    feedback = response.prompt_feedback
    if feedback and feedback.block_reason:
        raise LLMError(f"Gemini blocked this request ({feedback.block_reason}).")
    finish = response.candidates[0].finish_reason if response.candidates else None
    if finish == types.FinishReason.MAX_TOKENS:
        raise LLMError("The response was cut off (max tokens). Try a shorter resume or JD.")
    if finish not in (None, types.FinishReason.STOP):
        raise LLMError(f"Gemini stopped early ({finish}).")
    if not isinstance(response.parsed, schema):
        raise LLMError("Gemini returned output that didn't match the expected format.")
    return response.parsed
