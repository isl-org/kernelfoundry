"""Maps ``inference.provider``/``model``/``base_url`` config to the env vars the Copilot
CLI needs for BYOK (bring-your-own-key) model access.

Mirrors the pattern in :mod:`kernelfoundry.algorithm.inference_server`
(``register_inference_provider`` / ``_resolve_inference_target``), but for "how do I map a
provider onto Copilot CLI env vars" instead of "which class do I instantiate". Public
providers (``openai``, ``anthropic``) are registered here, internal providers can be
registered by an internal plugin module on import.
"""

from __future__ import annotations

import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)

_COPILOT_ENV_RESOLVERS: dict[str, Callable[..., dict[str, str] | None]] = {}


def register_copilot_env_resolver(provider: str, resolver: Callable[..., dict[str, str] | None]) -> None:
    """Register (or override) the Copilot CLI env resolver for an inference provider name."""
    _COPILOT_ENV_RESOLVERS[provider] = resolver


def resolve_copilot_env(provider: str | None, model: str, base_url: str | None = None) -> dict[str, str] | None:
    """Return Copilot CLI ``COPILOT_*`` env var overrides for *provider*/*model*, or ``None``.

    Returns ``None`` (rather than raising) when *provider* has no registered resolver, or
    when the resolver can't find the credentials it needs (e.g. a missing API key) --
    logging a warning either way. ``None`` means "leave Copilot's env untouched", so the
    CLI falls back to its own default model / whatever ``COPILOT_*`` vars the caller has
    already exported by hand.
    """
    if provider is None:
        return None
    resolver = _COPILOT_ENV_RESOLVERS.get(provider)
    if resolver is None:
        logger.warning(
            "No Copilot CLI env resolver registered for inference provider '%s'; the agentic "
            "workflow will use Copilot's default model unless COPILOT_* env vars are already "
            "set by hand. Registered providers: %s",
            provider,
            sorted(_COPILOT_ENV_RESOLVERS),
        )
        return None
    return resolver(model=model, base_url=base_url)


def _resolve_openai(model: str, base_url: str | None = None) -> dict[str, str] | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; skipping Copilot CLI provider override for 'openai'.")
        return None
    return {
        "COPILOT_MODEL": model,
        "COPILOT_PROVIDER_TYPE": "openai",
        "COPILOT_PROVIDER_BASE_URL": base_url or "https://api.openai.com/v1",
        "COPILOT_PROVIDER_API_KEY": api_key,
    }


def _resolve_anthropic(model: str, base_url: str | None = None) -> dict[str, str] | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set; skipping Copilot CLI provider override for 'anthropic'.")
        return None
    return {
        "COPILOT_MODEL": model,
        "COPILOT_PROVIDER_TYPE": "anthropic",
        "COPILOT_PROVIDER_BASE_URL": base_url or "https://api.anthropic.com",
        "COPILOT_PROVIDER_API_KEY": api_key,
    }


register_copilot_env_resolver("openai", _resolve_openai)
register_copilot_env_resolver("anthropic", _resolve_anthropic)
