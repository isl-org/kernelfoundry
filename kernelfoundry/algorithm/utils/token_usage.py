"""Utilities for normalizing and aggregating token usage across providers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)


class TokenUsage(NamedTuple):
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0

    def __add__(self, other: dict[str, Any] | TokenUsage) -> TokenUsage:
        """Add token usage from dict or another TokenUsage."""
        if isinstance(other, dict):
            new_input = self.input_tokens + int(other.get("input_tokens") or 0)
            new_output = self.output_tokens + int(other.get("output_tokens") or 0)
            new_cached = self.cached_input_tokens + int(other.get("cached_input_tokens") or 0)
        elif isinstance(other, TokenUsage):
            new_input = self.input_tokens + other.input_tokens
            new_output = self.output_tokens + other.output_tokens
            new_cached = self.cached_input_tokens + other.cached_input_tokens
        else:
            return NotImplemented
        return TokenUsage(input_tokens=new_input, output_tokens=new_output, cached_input_tokens=new_cached)

    def __radd__(self, other: dict[str, Any]) -> TokenUsage:
        """Support reverse addition (dict + TokenUsage)."""
        return self.__add__(other)


def zero_token_usage() -> TokenUsage:
    return TokenUsage(input_tokens=0, output_tokens=0, cached_input_tokens=0)


def parse_copilot_otel_usage(otel_path: str | Path) -> TokenUsage:
    """Parse Copilot OTEL JSONL usage and aggregate token counts over lines.
    Returns zero usage when the file is absent (i.e. copilot did not start)
    """
    path = Path(otel_path)
    usage_total = zero_token_usage()
    if not path.is_file():
        logger.warning(
            "No Copilot telemetry at %s, so token usage is unknown. This normally means the Copilot "
            "process did not start or exited early -- check its stdout log for the reason.",
            path,
        )
        return usage_total
    with path.open("r", encoding="utf-8", errors="replace") as infile:
        counter = 0
        for line in infile:
            counter += 1
            record = json.loads(line)
            line_usage = _extract_usage_from_otel_record(record)
            usage_total += line_usage
    return usage_total


def _extract_usage_from_otel_record(record: Any) -> TokenUsage:
    attrs = _collect_otel_attributes(record)
    # Copilot emits usage on both the per-request "chat" span *and* the parent
    # "invoke_agent" span, with the latter's fields being a rollup total across all
    # of its child chat calls. Counting both double-counts every session's usage, so
    # only the leaf "chat" spans (one per actual LLM request) are summed here.
    if attrs.get("gen_ai.operation.name") != "chat":
        return zero_token_usage()
    # Copilot's gen_ai.usage.input_tokens is inclusive of cache-read tokens (OpenAI-style
    # semantics), so subtract cache-read out to get the new/full-price input token count.
    all_input_tokens = int(attrs.get("gen_ai.usage.input_tokens") or 0)
    cached_input_tokens = int(attrs.get("gen_ai.usage.cache_read.input_tokens") or 0)
    new_input_tokens = max(all_input_tokens - cached_input_tokens, 0)
    return TokenUsage(
        input_tokens=new_input_tokens,
        output_tokens=int(attrs.get("gen_ai.usage.output_tokens") or 0),
        cached_input_tokens=cached_input_tokens,
    )


def _collect_otel_attributes(node: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}

    def walk(value: Any):
        if isinstance(value, dict):
            attr_list = value.get("attributes")
            if isinstance(attr_list, dict):
                if "gen_ai.usage.input_tokens" in attr_list or "gen_ai.usage.output_tokens" in attr_list:
                    attrs.update(attr_list)
            elif isinstance(attr_list, list):
                for attr in attr_list:
                    key = attr.get("key")
                    val = _extract_otel_attribute_value(attr.get("value"))
                    attrs[str(key)] = val
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(node)
    return attrs


def _extract_otel_attribute_value(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("intValue", "doubleValue", "stringValue", "boolValue"):
            if key in value:
                return value[key]
    return value
