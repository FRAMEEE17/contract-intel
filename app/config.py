"""Runtime wiring: turn .env / environment settings into concrete adapters.

Only this module imports concrete adapters; the application layer depends on the
Protocols in app.domain.ports, so swapping the LLM provider is a config change here.

Selection is via env (falls back to .env, same file evals/jury.py reads):

  common:
    LLM_PROVIDER      "local" (default) | "cloud" | "azure"
    LLM_MAX_TOKENS    output cap (default 2048)
    LLM_TIMEOUT       seconds (default 120)
  local / cloud (OpenAI-compatible):
    LLM_BASE_URL      endpoint (required for "cloud")
    LLM_MODEL         model id (required for "cloud")
    LLM_API_KEY       bearer token (required for "cloud")
    LLM_ENABLE_THINKING  "true"/"false" (default false: fast extraction)
  azure (Azure OpenAI):
    AZURE_OPENAI_ENDPOINT     https://<resource>.openai.azure.com   (required)
    AZURE_OPENAI_DEPLOYMENT   deployment name (used as `model`)      (required)
    AZURE_OPENAI_API_VERSION  default "2024-10-21" (use "2024-12-01-preview" for gpt-5)
    AZURE_OPENAI_API_KEY      optional; omit to use Managed Identity (Entra ID)
    AZURE_OPENAI_REASONING    optional override; auto-detected from o-series/gpt-5 names
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from app.domain.ports import LLMClient

if TYPE_CHECKING:
    from app.domain.ports import Chunker, DocParser, Embedder, Guardrail, PromptRegistry

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension (for the Azure Search vector field)

APP_DIR = Path(__file__).resolve().parent
ENV_PATH = APP_DIR.parent / ".env"

_DEFAULT_AZURE_API_VERSION = "2024-10-21"

# OpenAI-compatible presets. None -> must be supplied via env.
_OAI_PRESETS: dict[str, dict] = {
    "local": {
        "base_url": "http://localhost:8081/v1",   # mlx_lm.server --model ... --port 8081
        "model": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
        "api_key_env": None,                       # local server needs no key
    },
    "cloud": {
        "base_url": None,                          # from LLM_BASE_URL
        "model": None,                             # from LLM_MODEL
        "api_key_env": "LLM_API_KEY",
    },
}
_PROVIDERS = sorted(_OAI_PRESETS) + ["azure"]


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str                 # deployment name when provider == "azure"
    api_key: str               # "" allowed for azure -> Managed Identity
    timeout: float
    max_tokens: int
    base_url: str = ""
    enable_thinking: bool = False
    azure_endpoint: str = ""
    api_version: str = ""


def _require(value: str | None, name: str) -> str:
    if not value:
        raise RuntimeError(f"{name} is required for this provider (set it in {ENV_PATH} or the environment)")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_llm_settings() -> LLMSettings:
    load_dotenv(ENV_PATH)
    provider = os.environ.get("LLM_PROVIDER", "local").strip().lower()
    timeout = float(os.environ.get("LLM_TIMEOUT", "120"))
    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "2048"))

    if provider == "azure":
        return LLMSettings(
            provider="azure",
            model=_require(os.environ.get("AZURE_OPENAI_DEPLOYMENT"), "AZURE_OPENAI_DEPLOYMENT"),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),   # "" -> Managed Identity
            timeout=timeout,
            max_tokens=max_tokens,
            azure_endpoint=_require(os.environ.get("AZURE_OPENAI_ENDPOINT"), "AZURE_OPENAI_ENDPOINT"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", _DEFAULT_AZURE_API_VERSION),
        )

    if provider not in _OAI_PRESETS:
        raise RuntimeError(f"unknown LLM_PROVIDER={provider!r}; choose one of {_PROVIDERS}")
    preset = _OAI_PRESETS[provider]
    api_key_env = preset["api_key_env"]
    api_key = _require(os.environ.get(api_key_env), api_key_env) if api_key_env else "not-needed"
    return LLMSettings(
        provider=provider,
        model=_require(os.environ.get("LLM_MODEL", preset["model"]), "LLM_MODEL"),
        api_key=api_key,
        timeout=timeout,
        max_tokens=max_tokens,
        base_url=_require(os.environ.get("LLM_BASE_URL", preset["base_url"]), "LLM_BASE_URL"),
        enable_thinking=_env_bool("LLM_ENABLE_THINKING", False),
    )


_ENTRA_SCOPE = "https://cognitiveservices.azure.com/.default"


def _make_client(s: LLMSettings):
    """Composition root: construct the concrete SDK client. The ONLY place that knows a provider exists."""
    if s.provider == "azure":
        from openai import AzureOpenAI
        if s.api_key:  # local-dev fallback
            return AzureOpenAI(azure_endpoint=s.azure_endpoint, api_version=s.api_version,
                               api_key=s.api_key, timeout=s.timeout)
        try:  # preferred: Entra ID / Managed Identity, no secret in code
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        except ImportError as exc:
            raise RuntimeError(
                "Managed-Identity auth needs 'azure-identity' (pip install azure-identity), "
                "or set AZURE_OPENAI_API_KEY for local dev."
            ) from exc
        token_provider = get_bearer_token_provider(DefaultAzureCredential(), _ENTRA_SCOPE)
        return AzureOpenAI(azure_endpoint=s.azure_endpoint, api_version=s.api_version,
                           azure_ad_token_provider=token_provider, timeout=s.timeout)

    from openai import OpenAI
    return OpenAI(base_url=s.base_url, api_key=s.api_key, timeout=s.timeout)


def _is_reasoning_model(model: str) -> bool:
    """o-series / gpt-5 speak a different request dialect (max_completion_tokens, no
    temperature/seed). Detect by deployment name; AZURE_OPENAI_REASONING overrides."""
    override = os.environ.get("AZURE_OPENAI_REASONING")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}
    m = model.lower()
    return m.startswith(("o1", "o3", "o4")) or "gpt-5" in m


def build_llm(settings: LLMSettings | None = None) -> LLMClient:
    """Wire a provider-agnostic `LLMClient` (composition root). Callers pass the model/deployment per call."""
    s = settings or load_llm_settings()
    from app.adapters.llm.openai_chat import OpenAIChatLLM
    is_azure = s.provider == "azure"
    reasoning = is_azure and _is_reasoning_model(s.model)
    max_tokens = s.max_tokens
    if reasoning and "LLM_MAX_TOKENS" not in os.environ:
        max_tokens = max(max_tokens, 16384)  # reasoning tokens share the cap; small caps yield empty content
    return OpenAIChatLLM(
        _make_client(s),
        max_tokens=max_tokens,
        enable_thinking=s.enable_thinking,
        template_thinking=not is_azure,   # chat_template_kwargs: self-hosted OpenAI-compat only
        json_mode=is_azure,               # Azure supports response_format JSON mode
        reasoning=reasoning,
    )


def build_search():
    """SearchClient for the serving/storage path: in-memory (default) or Azure AI Search.

    The eval path builds its own in-memory index for reproducibility; this is the
    persistent store that backs the /contracts repository. Chosen via SEARCH_PROVIDER.
    """
    provider = os.environ.get("SEARCH_PROVIDER", "memory").strip().lower()
    if provider == "azure":
        from app.adapters.search.azure_ai_search import AzureAISearch
        return AzureAISearch.for_azure(
            endpoint=_require(os.environ.get("AZURE_SEARCH_ENDPOINT"), "AZURE_SEARCH_ENDPOINT"),
            api_key=_require(os.environ.get("AZURE_SEARCH_KEY"), "AZURE_SEARCH_KEY"),
            index_name=os.environ.get("AZURE_SEARCH_INDEX", "contracts"),
            vector_dim=EMBED_DIM,
        )
    from app.adapters.search.in_memory import InMemorySearch
    return InMemorySearch()


@dataclass(frozen=True)
class Pipeline:
    """Adapters the answer use-case runs on, wired for one provider."""

    llm: LLMClient
    embedder: Embedder
    chunker: Chunker
    parser: DocParser
    guardrail: Guardrail
    registry: PromptRegistry
    embed_model: str
    model: str


def build_pipeline(settings: LLMSettings | None = None) -> Pipeline:
    """Assemble the answer pipeline; the one place its adapters are chosen."""
    s = settings or load_llm_settings()
    from app.adapters.chunking.fixed_chunker import FixedChunker
    from app.adapters.documents.text_parser import TextDocParser
    from app.adapters.embeddings.sentence_transformer import SentenceTransformerEmbedder
    from app.adapters.guardrails.passthrough import PassthroughGuardrail
    from app.adapters.registry.file_registry import FilePromptRegistry

    return Pipeline(
        llm=build_llm(s),
        embedder=SentenceTransformerEmbedder(),
        chunker=FixedChunker(),
        parser=TextDocParser(),
        guardrail=PassthroughGuardrail(),
        registry=FilePromptRegistry(),
        embed_model=EMBED_MODEL,
        model=s.model,
    )
