"""LangChain setup: the Gemini chat model and a helper that builds structured-output chains."""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")

_llm: ChatGoogleGenerativeAI | None = None


class LLMError(RuntimeError):
    pass


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        if not os.getenv("GEMINI_API_KEY"):
            raise LLMError("GEMINI_API_KEY is not set. Add it to your .env file.")
        # Generous output limit because Gemini Pro's thinking tokens count toward it.
        _llm = ChatGoogleGenerativeAI(
            model=MODEL,
            google_api_key=os.environ["GEMINI_API_KEY"],
            max_output_tokens=32000,
            max_retries=2,
        )
    return _llm


def structured_chain(system: str, human: str, schema) -> Runnable:
    """prompt | Gemini with structured output. The chain returns {"raw", "parsed", "parsing_error"}."""
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    return prompt | get_llm().with_structured_output(schema, include_raw=True)


def _status_code(exc: BaseException) -> int | None:
    while exc is not None:
        code = getattr(exc, "code", None)
        if isinstance(code, int):
            return code
        exc = exc.__cause__
    return None


def run(chain: Runnable, inputs: dict, schema):
    """Invoke a structured chain and return a validated `schema` instance, or raise LLMError."""
    try:
        result = chain.invoke(inputs)
    except LLMError:
        raise
    except Exception as e:  # langchain-google-genai wraps SDK errors in its own exception types
        code = _status_code(e)
        if code in (401, 403) or "API key" in str(e):
            raise LLMError("Invalid GEMINI_API_KEY.") from e
        if code == 429:
            raise LLMError("Gemini rate limit or free-tier quota reached. Wait a minute or check your quota.") from e
        if code == 404:
            raise LLMError(f"Model '{MODEL}' not found. Set GEMINI_MODEL in .env to a current model.") from e
        raise LLMError(f"Gemini API error: {e}") from e

    finish = str(result["raw"].response_metadata.get("finish_reason", "STOP")).upper()
    if "MAX_TOKENS" in finish:
        raise LLMError("The response was cut off (max tokens). Try a shorter resume or JD.")
    if "STOP" not in finish:
        raise LLMError(f"Gemini stopped early ({finish}).")
    if result.get("parsing_error") or not isinstance(result.get("parsed"), schema):
        raise LLMError("Gemini returned output that didn't match the expected format.")
    return result["parsed"]
