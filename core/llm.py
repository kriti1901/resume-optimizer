"""Thin wrapper around the Anthropic SDK for structured (pydantic) outputs."""

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "claude-haiku-5-5")
REWRITE_MODEL = os.getenv("REWRITE_MODEL", "claude-sonnet-5-5")

_client: anthropic.Anthropic | None = None


class LLMError(RuntimeError):
    pass


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def parse(model: str, system: str, user: str, schema, max_tokens: int = 8000, effort: str = "low"):
    """Call Claude and return a validated instance of `schema`.

    Sonnet 5.5 gets server-side refusal fallbacks; Haiku 5.5 has none, so it uses the plain endpoint.
    """
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=schema,
        output_config={"effort": effort},
    )
    try:
        if model.startswith("claude-sonnet-5-5"):
            response = client().beta.messages.parse(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            )
        else:
            response = client().messages.parse(**kwargs)
    except anthropic.AuthenticationError as e:
        raise LLMError("Invalid or missing ANTHROPIC_API_KEY.") from e
    except anthropic.RateLimitError as e:
        raise LLMError("Rate limited by the Anthropic API. Try again in a minute.") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"Anthropic API error ({e.status_code}): {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("Could not reach the Anthropic API.") from e

    if response.stop_reason == "refusal":
        raise LLMError("The model declined this request.")
    if response.stop_reason == "max_tokens":
        raise LLMError("The response was cut off (max_tokens). Try a shorter resume or JD.")
    if response.parsed_output is None:
        raise LLMError("The model returned output that didn't match the expected format.")
    return response.parsed_output
