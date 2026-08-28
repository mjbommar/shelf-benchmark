"""Common LLM backend interface with provider-specific implementations.

This abstracts provider-specific SDK calls behind a small protocol so the rest
of the codebase can switch providers (OpenAI, Anthropic, Gemini, OpenRouter,
xAI) without rewriting generation logic. Each backend supports per-request
generation plus a batch method; if a true batch API is unavailable at runtime,
the batch method falls back to per-item calls.

Two families live here:

* SDK-backed backends (:class:`OpenAIResponsesBackend`,
  :class:`AnthropicMessagesBackend`, :class:`GeminiBackend`) which import their
  vendor SDK lazily inside methods.
* HTTP backends built on :class:`OpenAIChatCompatibleBackend`, which speaks the
  OpenAI ``/chat/completions`` wire format over ``httpx`` (a hard dependency of
  this project, unlike the vendor SDKs). :class:`OpenRouterBackend` and
  :class:`XAIBackend` are thin subclasses that only change the base URL, the
  API-key environment variable, and the provider-specific reasoning knob.

Reasoning-token control is expressed once as a :class:`ReasoningConfig` and
translated per provider. Thinking tokens bill as output tokens, so the default
posture for corpus generation is ``ReasoningConfig.off()``.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

# --------------------------------------------------------------------------- #
# Shared request/response types
# --------------------------------------------------------------------------- #


@dataclass
class GenerationParams:
    """LLM sampling parameters."""

    temperature: float
    top_p: float
    max_output_tokens: int = 4096


@dataclass
class GenerationRequest:
    """A single generation request."""

    prompt: str
    system_prompt: str


#: Canonical effort levels, ordered from cheapest to most expensive. These are
#: the union of what OpenAI (``none``/``minimal``/``low``/``medium``/``high``/
#: ``xhigh``/``max``), OpenRouter (same set) and xAI (``low``/``medium``/
#: ``high``/``xhigh``, plus ``none`` on some models) accept. Support is
#: *model-dependent* on every provider: ``grok-4.6`` rejects ``none`` while
#: ``grok-4.3`` accepts it, and OpenRouter's ``/models`` payload reports the
#: accepted set per model under ``reasoning.supported_efforts``.
REASONING_EFFORTS: tuple[str, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

#: Heuristic effort -> Gemini ``thinking_budget`` mapping. Gemini has no effort
#: parameter, only a token budget (0 disables, -1 means "dynamic"), so this is
#: our own calibration rather than a documented equivalence.
_GEMINI_EFFORT_BUDGETS: dict[str, int] = {
    "none": 0,
    "minimal": 128,
    "low": 512,
    "medium": 2048,
    "high": 8192,
    "xhigh": -1,
    "max": -1,
}

#: Anthropic's ``output_config.effort`` accepts max/xhigh/high/medium/low only.
_ANTHROPIC_EFFORTS: dict[str, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}


@dataclass
class ReasoningConfig:
    """Provider-neutral request for how much the model should think.

    Thinking/reasoning tokens are billed as output tokens on every provider, so
    for bulk synthetic-document generation the intended posture is
    ``ReasoningConfig.off()``. Providers disagree on both the parameter name and
    the accepted values, so this object is translated per backend rather than
    passed through.

    Attributes:
        effort: One of :data:`REASONING_EFFORTS`, or ``None`` to leave the
            provider default alone.
        max_tokens: Explicit reasoning token budget, where the provider exposes
            one (Anthropic ``budget_tokens``, Gemini ``thinking_budget``,
            OpenRouter ``reasoning.max_tokens``).
        enabled: Explicit on/off switch. ``False`` is the most portable way to
            turn reasoning off on OpenRouter.
        exclude: Ask the provider to omit reasoning text from the response
            body. Reduces payload size; it does *not* reduce cost.
    """

    effort: str | None = None
    max_tokens: int | None = None
    enabled: bool | None = None
    exclude: bool = True

    @classmethod
    def off(cls) -> ReasoningConfig:
        """Request no reasoning at all (the cost-safe default)."""
        return cls(effort="none", max_tokens=0, enabled=False, exclude=True)

    @classmethod
    def minimal(cls) -> ReasoningConfig:
        """Request the smallest non-zero amount of reasoning."""
        return cls(effort="minimal", enabled=True, exclude=True)

    @property
    def is_off(self) -> bool:
        """True when this config asks for reasoning to be disabled."""
        return (
            self.enabled is False or self.effort == "none" or self.max_tokens == 0
        ) and self.enabled is not True

    def to_openrouter(self) -> dict[str, Any]:
        """Build the OpenRouter ``reasoning`` request object.

        Verified shapes: ``{"enabled": false, "exclude": true}`` drops
        ``usage.completion_tokens_details.reasoning_tokens`` to 0, and
        ``{"effort": ...}``/``{"max_tokens": ...}`` are mutually exclusive per
        OpenRouter's docs.
        """
        if self.is_off:
            return {"enabled": False, "exclude": self.exclude}
        out: dict[str, Any] = {}
        if self.effort:
            out["effort"] = self.effort
        elif self.max_tokens is not None and self.max_tokens > 0:
            out["max_tokens"] = self.max_tokens
        if self.enabled is not None:
            out["enabled"] = self.enabled
        out["exclude"] = self.exclude
        return out

    def to_openai_responses(self) -> dict[str, Any] | None:
        """Build the OpenAI Responses ``reasoning`` object, or None."""
        effort = self.effort or ("none" if self.is_off else None)
        if effort is None:
            return None
        return {"effort": effort}

    def to_xai_effort(self) -> str | None:
        """Build the xAI ``reasoning_effort`` value, or None."""
        if self.effort:
            return self.effort
        if self.is_off:
            return "none"
        return None

    def to_anthropic(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Build ``(thinking, output_config)`` for the Anthropic Messages API."""
        thinking: dict[str, Any] | None = None
        output_config: dict[str, Any] | None = None
        if self.is_off:
            thinking = {"type": "disabled"}
        elif self.max_tokens is not None and self.max_tokens > 0:
            thinking = {"type": "enabled", "budget_tokens": self.max_tokens}
        if self.effort:
            mapped = _ANTHROPIC_EFFORTS.get(self.effort)
            if mapped is not None:
                output_config = {"effort": mapped}
        return thinking, output_config

    def to_gemini_thinking_budget(self) -> int | None:
        """Build a Gemini ``thinking_budget``, or None to leave it unset."""
        if self.max_tokens is not None:
            return self.max_tokens
        if self.is_off:
            return 0
        if self.effort:
            return _GEMINI_EFFORT_BUDGETS.get(self.effort)
        return None


@dataclass
class GenerationResult:
    """Normalized generation result with optional token usage.

    The trailing fields are additive; existing positional callers of
    ``GenerationResult(text, input_tokens, output_tokens)`` are unaffected.

    Attributes:
        provider_served: The upstream provider that actually served the
            request. Only gateways report this. OpenRouter returns it as a
            top-level ``provider`` string (observed values: ``"DeepInfra"``,
            ``"Ambient"``, ``"OpenInference"``), and it matters because one
            model id can be routed to backends with different quantization.
        reasoning_tokens: Thinking tokens billed as output. Read from
            ``usage.completion_tokens_details.reasoning_tokens`` (chat
            completions), ``usage.output_tokens_details.reasoning_tokens``
            (OpenAI Responses) or ``usage_metadata.thoughts_token_count``
            (Gemini). Anthropic does not break these out, so it stays None.
        model_resolved: The exact model string the provider echoed back. xAI
            resolves aliases here (``grok-4.20-non-reasoning`` ->
            ``grok-4.20-0309-non-reasoning``).
    """

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    provider_served: str | None = None
    reasoning_tokens: int | None = None
    model_resolved: str | None = None


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for LLM backends used by the generator."""

    provider: str
    model: str

    def generate(
        self, request: GenerationRequest, params: GenerationParams
    ) -> GenerationResult:
        """Generate a single completion."""

    async def generate_async(
        self, request: GenerationRequest, params: GenerationParams
    ) -> GenerationResult:
        """Async version of generate."""

    def generate_batch(
        self, requests: list[GenerationRequest], params: GenerationParams
    ) -> list[GenerationResult]:
        """Generate a batch of completions."""


class _BaseBackend:
    """Default implementations for async and batch helpers."""

    provider: str
    model: str

    def generate_batch(
        self, requests: list[GenerationRequest], params: GenerationParams
    ) -> list[GenerationResult]:
        return [self.generate(request, params) for request in requests]

    async def generate_async(
        self, request: GenerationRequest, params: GenerationParams
    ) -> GenerationResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, request, params)


# --------------------------------------------------------------------------- #
# OpenAI Responses backend
# --------------------------------------------------------------------------- #


# Newer OpenAI models reject sampling parameters they no longer honour. The
# message names the parameter, so it can be dropped precisely instead of
# stripping every optional field.
_MAX_SAMPLING_RETRIES = 4

_UNSUPPORTED_PARAM_RE = re.compile(r"unsupported parameter:?\s*'?\"?([a-z_]+)", re.I)


def _drop_unsupported_sampling(exc: Exception, kwargs: dict) -> dict | None:
    """Return ``kwargs`` minus the parameter this error names, or None.

    None means the error was not an unsupported-parameter complaint about a key
    we actually sent, so the caller should re-raise rather than retry blindly.
    """
    match = _UNSUPPORTED_PARAM_RE.search(str(exc))
    if not match:
        return None
    name = match.group(1)
    if name not in kwargs:
        return None
    retry = dict(kwargs)
    retry.pop(name, None)
    return retry


class OpenAIResponsesBackend(_BaseBackend):
    """Backend for OpenAI Responses API."""

    provider = "openai"

    def __init__(
        self,
        model: str,
        *,
        service_tier: str | None = None,
        client=None,
        async_client=None,
        reasoning: ReasoningConfig | None = None,
    ):
        import openai

        self.model = model
        self._service_tier = service_tier
        self._reasoning = reasoning
        self._client: openai.OpenAI | None = client
        self._async_client: openai.AsyncOpenAI | None = async_client

    def _get_client(self):
        import openai

        if self._client is None:
            self._client = openai.OpenAI()
        return self._client

    def _get_async_client(self):
        import openai

        if self._async_client is None:
            self._async_client = openai.AsyncOpenAI()
        return self._async_client

    def _build_request_kwargs(
        self, request: GenerationRequest, params: GenerationParams
    ) -> dict:
        kwargs = {
            "model": self.model,
            "instructions": request.system_prompt,
            "input": request.prompt,
            "max_output_tokens": params.max_output_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
        }
        if self._service_tier:
            kwargs["service_tier"] = self._service_tier
        if self._reasoning is not None:
            reasoning = self._reasoning.to_openai_responses()
            if reasoning is not None:
                kwargs["reasoning"] = reasoning
        return kwargs

    def _to_result(self, response) -> GenerationResult:
        text = getattr(response, "output_text", "") or ""
        if not text.strip():
            raise ValueError("Received empty output from OpenAI")
        usage = getattr(response, "usage", None)
        details = getattr(usage, "output_tokens_details", None) if usage else None
        return GenerationResult(
            text=text,
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            provider_served=self.provider,
            reasoning_tokens=_extract_field(details, "reasoning_tokens"),
            model_resolved=getattr(response, "model", None),
        )

    def generate(
        self, request: GenerationRequest, params: GenerationParams
    ) -> GenerationResult:
        kwargs = self._build_request_kwargs(request, params)
        client = self._get_client()
        # Newer OpenAI models reject sampling parameters outright:
        #   400 "Unsupported parameter: 'temperature' is not supported with
        #   this model."
        # They reject them one at a time -- dropping `temperature` then surfaces
        # the same complaint about `top_p` -- so retry until the request is
        # accepted or the error stops naming a parameter we sent. Bounded by the
        # number of droppable keys.
        for _ in range(_MAX_SAMPLING_RETRIES):
            try:
                return self._to_result(
                    client.responses.create(**kwargs)  # type: ignore[attr-defined]
                )
            except Exception as exc:  # noqa: BLE001 - SDK error type varies
                dropped = _drop_unsupported_sampling(exc, kwargs)
                if dropped is None:
                    raise
                kwargs = dropped
        return self._to_result(
            client.responses.create(**kwargs)  # type: ignore[attr-defined]
        )

    async def generate_async(
        self, request: GenerationRequest, params: GenerationParams
    ) -> GenerationResult:
        response = await self._get_async_client().responses.create(  # type: ignore[attr-defined]
            **self._build_request_kwargs(request, params)
        )
        return self._to_result(response)


# --------------------------------------------------------------------------- #
# Anthropic Messages backend (with batch hook)
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=4)
def _messages_create_params(create_fn) -> frozenset[str]:
    """Parameter names accepted by this SDK's ``messages.create``."""
    try:
        return frozenset(inspect.signature(create_fn).parameters)
    except (TypeError, ValueError):
        # Un-introspectable callable: assume the older, permissive signature.
        return frozenset({"temperature"})


def _anthropic_accepts_temperature(client) -> bool:
    """Whether the installed anthropic SDK still takes ``temperature``."""
    return "temperature" in _messages_create_params(client.messages.create)


class AnthropicMessagesBackend(_BaseBackend):
    """Backend for Anthropic Claude Messages API.

    Note: Claude 4.x models (claude-opus-4-*, claude-sonnet-4-*) do not allow
    both temperature and top_p to be specified. This backend only uses
    temperature (clamped to 0.0-1.0 range) for compatibility.

    See: https://docs.anthropic.com/claude/reference/messages_post
    """

    provider = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        client=None,
        use_batch_api: bool = False,
        poll_seconds: float = 1.0,
        timeout_seconds: float = 300.0,
        reasoning: ReasoningConfig | None = None,
    ):
        import anthropic

        self.model = model
        self._client: anthropic.Anthropic | None = client
        self._use_batch_api = use_batch_api
        self._poll_seconds = poll_seconds
        self._timeout_seconds = timeout_seconds
        self._reasoning = reasoning

    def _get_client(self):
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def _clamp_temperature(self, temperature: float) -> float:
        """Clamp temperature to Anthropic's valid range (0.0-1.0)."""
        return max(0.0, min(1.0, temperature))

    def _reasoning_kwargs(self) -> dict:
        """Build ``thinking`` / ``output_config`` kwargs from the config.

        ``thinking={"type": "disabled"}`` is the documented off switch on models
        where thinking is on by default; ``output_config={"effort": ...}`` is
        the adaptive-thinking lever. Support is model-dependent (some models
        reject a disabled thinking block outright), so callers that hit a 400
        should retry without a reasoning config.
        """
        if self._reasoning is None:
            return {}
        thinking, output_config = self._reasoning.to_anthropic()
        kwargs: dict = {}
        if thinking is not None:
            kwargs["thinking"] = thinking
        if output_config is not None:
            kwargs["output_config"] = output_config
        return kwargs

    def generate(
        self, request: GenerationRequest, params: GenerationParams
    ) -> GenerationResult:
        client = self._get_client()
        # Note: Claude 4.x models don't allow both temperature and top_p, so
        # only temperature is used (clamped to 0-1).
        #
        # Newer SDKs drop it altogether: anthropic 1.1.0's messages.create has
        # no `temperature` parameter at all, and passing it raises TypeError,
        # which silently failed every Anthropic call. Probe the signature once
        # rather than pinning a version, so the backend works across SDKs.
        sampling: dict = {}
        if _anthropic_accepts_temperature(client):
            sampling["temperature"] = self._clamp_temperature(params.temperature)

        base = {
            "model": self.model,
            "max_tokens": params.max_output_tokens,
            "system": request.system_prompt,
            "messages": [{"role": "user", "content": request.prompt}],
            **sampling,
        }
        try:
            message = client.messages.create(**base, **self._reasoning_kwargs())
        except Exception as exc:  # noqa: BLE001 - SDK error type varies
            if not _looks_like_reasoning_rejection(str(exc)):
                raise
            # Some models refuse to have thinking switched off at all:
            #   400 '"thinking.type.disabled" is not supported by this model'
            # claude-fable-5 failed all 500 holdout documents this way. Retry
            # with the provider default rather than lose the run.
            message = client.messages.create(**base)
        usage = getattr(message, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage else None
        output_tokens = getattr(usage, "output_tokens", None) if usage else None
        return GenerationResult(
            text=_anthropic_content_to_text(message),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_served=self.provider,
            # Anthropic folds thinking tokens into output_tokens and does not
            # report them separately, so this intentionally stays None.
            reasoning_tokens=None,
            model_resolved=getattr(message, "model", None),
        )

    def generate_batch(
        self, requests: list[GenerationRequest], params: GenerationParams
    ) -> list[GenerationResult]:
        if not self._use_batch_api:
            return super().generate_batch(requests, params)

        client = self._get_client()
        if not hasattr(client, "beta") or not hasattr(client.beta, "messages"):
            # Fallback if batch API is unavailable in the installed SDK.
            return super().generate_batch(requests, params)

        # Build batch requests with custom_id and params structure
        # See: https://docs.anthropic.com/en/docs/build-with-claude/message-batches
        batch_requests = []
        reasoning_kwargs = self._reasoning_kwargs()
        for i, req in enumerate(requests):
            batch_requests.append(
                {
                    "custom_id": f"req-{i}",
                    "params": {
                        "model": self.model,
                        "max_tokens": params.max_output_tokens,
                        "temperature": self._clamp_temperature(params.temperature),
                        "system": req.system_prompt,
                        "messages": [{"role": "user", "content": req.prompt}],
                        **reasoning_kwargs,
                    },
                }
            )

        batch = client.beta.messages.batches.create(requests=batch_requests)

        deadline = time.monotonic() + self._timeout_seconds
        while True:
            status = client.beta.messages.batches.retrieve(batch.id)
            # API uses "ended" when complete, not "succeeded"
            if status.processing_status in {"ended"}:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Anthropic batch did not complete before timeout")
            time.sleep(self._poll_seconds)

        # SDK exposes results via an iterator, ordered by custom_id
        results = list(client.beta.messages.batches.results(batch.id))

        # Sort by custom_id to maintain order
        results.sort(key=lambda x: int(x.custom_id.split("-")[1]))

        outputs: list[GenerationResult] = []
        for item in results:
            result = getattr(item, "result", None)
            if result is None or getattr(result, "type", None) == "errored":
                error = getattr(result, "error", None) if result else None
                raise RuntimeError(f"Anthropic batch item failed: {error}")

            message = getattr(result, "message", None)
            usage = getattr(message, "usage", None) if message else None
            input_tokens = getattr(usage, "input_tokens", None) if usage else None
            output_tokens = getattr(usage, "output_tokens", None) if usage else None
            outputs.append(
                GenerationResult(
                    text=_anthropic_content_to_text(message),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    provider_served=self.provider,
                    reasoning_tokens=None,
                    model_resolved=getattr(message, "model", None),
                )
            )
        return outputs


def _anthropic_content_to_text(message) -> str:
    """Extract text from an Anthropic messages.create response."""
    content = getattr(message, "content", None) or []
    parts = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    combined = "\n".join(parts)
    if not combined.strip():
        raise ValueError("Received empty output from Anthropic")
    return combined


# --------------------------------------------------------------------------- #
# Google Gemini backend (google-genai)
# --------------------------------------------------------------------------- #


# Gemini's own "reasoning cannot be disabled" signal. Distinct wording from
# OpenRouter's, so it needs its own matcher.
_GEMINI_THINKING_REQUIRED_MARKERS: tuple[str, ...] = (
    "only works in thinking mode",
    "budget 0 is invalid",
    "thinking is required",
    "thinking_budget must be",
)


def _gemini_requires_thinking(exc: Exception) -> bool:
    """Does this Gemini error mean the model refuses thinking_budget=0?"""
    text = str(exc).lower()
    return any(marker in text for marker in _GEMINI_THINKING_REQUIRED_MARKERS)


class GeminiBackend(_BaseBackend):
    """Backend for Google Gemini via google-genai.

    Uses the modern google-genai SDK with GenerateContentConfig.
    See: https://googleapis.github.io/python-genai/

    Gemini 2.5+ models use "extended thinking" which consumes output tokens.
    To ensure enough tokens for actual content, you can either:
    - Set thinking_budget=0 to disable thinking (flash models only)
    - Set token_multiplier>1 to boost max_output_tokens (works for all models)

    See: https://github.com/googleapis/python-genai/issues/811

    The batch API uses async job submission with polling. Target turnaround
    is 24 hours but often completes faster. For latency-sensitive workloads,
    use sequential generation with concurrency instead.
    See: https://ai.google.dev/gemini-api/docs/batch-api
    """

    provider = "gemini"

    def __init__(
        self,
        model: str,
        *,
        client=None,
        use_batch_api: bool = False,
        poll_seconds: float = 30.0,
        timeout_seconds: float = 3600.0,  # 1 hour default
        thinking_budget: int | None = None,  # None=auto, 0=disable, >0=set budget
        token_multiplier: float = 1.0,  # Multiply max_output_tokens (for thinking overhead)
        reasoning: ReasoningConfig | None = None,
    ):
        from google import genai
        from google.genai import types

        self.model = model
        self._use_batch_api = use_batch_api
        self._poll_seconds = poll_seconds
        self._timeout_seconds = timeout_seconds
        self._reasoning = reasoning
        # An explicit thinking_budget always wins; otherwise derive one from the
        # provider-neutral reasoning config so callers can use one knob.
        if thinking_budget is None and reasoning is not None:
            thinking_budget = reasoning.to_gemini_thinking_budget()
        self._thinking_budget = thinking_budget
        self._token_multiplier = token_multiplier
        self._types = types

        if client is not None:
            self._client = client
        else:
            # Try GOOGLE_API_KEY first (common), then GEMINI_API_KEY
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get(
                "GEMINI_API_KEY"
            )
            if not api_key:
                raise ValueError(
                    "Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY "
                    "in environment or env.json"
                )
            self._client = genai.Client(api_key=api_key)

    def _is_thinking_model(self) -> bool:
        """Check if model uses extended thinking (Gemini 2.5+)."""
        return "2.5" in self.model or "3" in self.model

    def _build_config(self, request: GenerationRequest, params: GenerationParams):
        """Build a GenerateContentConfig with all parameters.

        Applies token_multiplier to max_output_tokens and optionally sets
        thinking_budget for models that support it.

        For Gemini 2.5+ models with thinking enabled, we auto-apply a 4x
        multiplier if none was specified, to ensure enough tokens for both
        thinking and output.

        See: https://github.com/googleapis/python-genai/issues/811
        """
        # Apply token multiplier (useful for thinking overhead)
        # Auto-apply 4x for thinking models if no multiplier specified,
        # with a minimum floor of 8192 to ensure room for variable thinking
        multiplier = self._token_multiplier
        if (
            multiplier == 1.0
            and self._is_thinking_model()
            and self._thinking_budget != 0
        ):
            multiplier = 4.0  # Default buffer for thinking overhead

        max_tokens = int(params.max_output_tokens * multiplier)

        # Ensure minimum for thinking models (thinking can consume 1000-4000+ tokens)
        if self._is_thinking_model() and self._thinking_budget != 0:
            max_tokens = max(max_tokens, 8192)

        config_kwargs = {
            "system_instruction": request.system_prompt,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_output_tokens": max_tokens,
        }

        # Apply thinking budget if configured and model supports it
        if self._thinking_budget is not None and self._is_thinking_model():
            config_kwargs["thinking_config"] = self._types.ThinkingConfig(
                thinking_budget=self._thinking_budget,
            )

        return self._types.GenerateContentConfig(**config_kwargs)

    def generate(
        self, request: GenerationRequest, params: GenerationParams
    ) -> GenerationResult:
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=request.prompt,
                config=self._build_config(request, params),
            )
        except Exception as exc:  # noqa: BLE001 - provider SDK error type varies
            if not _gemini_requires_thinking(exc):
                raise
            # Gemini pro models reject thinking_budget=0 outright:
            #   400 INVALID_ARGUMENT "Budget 0 is invalid. This model only works
            #   in thinking mode."
            # Silently fatal at scale -- gemini-3.1-pro-preview returned 1,500
            # errors and zero documents under a reasoning-off run. Retry with
            # the provider default, matching the OpenRouter mandatory-reasoning
            # path.
            previous = self._thinking_budget
            self._thinking_budget = None
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=request.prompt,
                    config=self._build_config(request, params),
                )
            except Exception:
                self._thinking_budget = previous
                raise
        text = _gemini_response_text(response)
        if not text.strip():
            # Build diagnostic info for empty responses
            diag = _gemini_empty_response_diagnostic(response)
            raise ValueError(f"Received empty output from Gemini: {diag}")
        return self._to_result(response, text)

    def _to_result(self, response, text: str) -> GenerationResult:
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        output_tokens = (
            getattr(usage, "candidates_token_count", None) if usage else None
        )
        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_served=self.provider,
            reasoning_tokens=(
                getattr(usage, "thoughts_token_count", None) if usage else None
            ),
            model_resolved=getattr(response, "model_version", None),
        )

    def generate_batch(
        self, requests: list[GenerationRequest], params: GenerationParams
    ) -> list[GenerationResult]:
        """Generate batch using Gemini's async batch API.

        Submits an inline batch job, polls for completion, and retrieves results.
        Falls back to sequential if batch API is unavailable or fails.
        """
        if not self._use_batch_api:
            return super().generate_batch(requests, params)

        try:
            return self._generate_batch_impl(requests, params)
        except Exception:
            # Fall back to sequential on any batch API error
            return super().generate_batch(requests, params)

    def _generate_batch_impl(
        self, requests: list[GenerationRequest], params: GenerationParams
    ) -> list[GenerationResult]:
        """Internal batch implementation using client.batches API."""
        # Build inline requests in Gemini batch format
        # Auto-apply 4x for thinking models if no multiplier specified
        multiplier = self._token_multiplier
        if (
            multiplier == 1.0
            and self._is_thinking_model()
            and self._thinking_budget != 0
        ):
            multiplier = 4.0

        max_tokens = int(params.max_output_tokens * multiplier)

        # Ensure minimum for thinking models
        if self._is_thinking_model() and self._thinking_budget != 0:
            max_tokens = max(max_tokens, 8192)

        inline_requests = []
        for req in requests:
            generation_config = {
                "temperature": params.temperature,
                "topP": params.top_p,
                "maxOutputTokens": max_tokens,
            }

            # Apply thinking budget if configured
            if self._thinking_budget is not None and self._is_thinking_model():
                generation_config["thinkingConfig"] = {
                    "thinkingBudget": self._thinking_budget
                }

            inline_requests.append(
                {
                    "contents": [{"parts": [{"text": req.prompt}], "role": "user"}],
                    "systemInstruction": {"parts": [{"text": req.system_prompt}]},
                    "generationConfig": generation_config,
                }
            )

        # Submit batch job
        batch_job = self._client.batches.create(
            model=self.model,
            src=inline_requests,
            config={"display_name": f"shelf-batch-{len(requests)}-requests"},
        )

        # Poll for completion
        completed_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }

        deadline = time.monotonic() + self._timeout_seconds
        job_name = batch_job.name

        while True:
            batch_job = self._client.batches.get(name=job_name)
            state_name = getattr(batch_job.state, "name", str(batch_job.state))
            if state_name in completed_states:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Gemini batch did not complete before timeout")
            time.sleep(self._poll_seconds)

        state_name = getattr(batch_job.state, "name", str(batch_job.state))
        if state_name != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Gemini batch failed with state={state_name}")

        # Extract results from inlined responses
        outputs: list[GenerationResult] = []
        inlined_responses = getattr(
            getattr(batch_job, "dest", None), "inlined_responses", []
        )

        for inline_response in inlined_responses or []:
            response = getattr(inline_response, "response", None)
            error = getattr(inline_response, "error", None)

            if error:
                raise RuntimeError(f"Gemini batch item failed: {error}")

            text = _gemini_response_text(response) if response else ""
            outputs.append(self._to_result(response, text))

        if len(outputs) != len(requests):
            raise ValueError(
                f"Gemini batch returned {len(outputs)} results for {len(requests)} requests"
            )

        return outputs


def _gemini_empty_response_diagnostic(response) -> str:
    """Build diagnostic string for empty Gemini responses."""
    parts = []

    # Check usage metadata
    usage = getattr(response, "usage_metadata", None)
    if usage:
        thoughts = getattr(usage, "thoughts_token_count", 0) or 0
        output = getattr(usage, "candidates_token_count", 0) or 0
        prompt = getattr(usage, "prompt_token_count", 0) or 0
        parts.append(f"tokens(prompt={prompt}, thoughts={thoughts}, output={output})")

    # Check prompt feedback
    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback:
        block_reason = getattr(prompt_feedback, "block_reason", None)
        if block_reason:
            parts.append(f"prompt_blocked={block_reason}")

    # Check candidates
    candidates = getattr(response, "candidates", []) or []
    if not candidates:
        parts.append("no_candidates")
    else:
        for i, cand in enumerate(candidates):
            finish_reason = getattr(cand, "finish_reason", None)
            if finish_reason:
                reason_name = getattr(finish_reason, "name", str(finish_reason))
                parts.append(f"candidate[{i}].finish_reason={reason_name}")

            # Check safety ratings
            safety_ratings = getattr(cand, "safety_ratings", []) or []
            blocked = [
                getattr(r, "category", "?")
                for r in safety_ratings
                if getattr(r, "blocked", False)
            ]
            if blocked:
                parts.append(f"candidate[{i}].blocked_categories={blocked}")

    return "; ".join(parts) if parts else "unknown"


def _gemini_response_text(response) -> str:
    """Extract text from a google-genai response.

    Also checks for common issues like safety blocks or token exhaustion.
    """
    # Check for prompt-level feedback (safety blocks)
    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback:
        block_reason = getattr(prompt_feedback, "block_reason", None)
        if block_reason and str(block_reason) != "BLOCK_REASON_UNSPECIFIED":
            raise ValueError(f"Gemini blocked prompt: {block_reason}")

    # Try direct .text accessor first
    if hasattr(response, "text"):
        text = response.text
        if text:
            return text

    # Fallback to candidates
    candidates = getattr(response, "candidates", []) or []
    if not candidates:
        # Check if thinking exhausted tokens
        usage = getattr(response, "usage_metadata", None)
        if usage:
            thoughts = getattr(usage, "thoughts_token_count", 0) or 0
            output = getattr(usage, "candidates_token_count", 0) or 0
            if thoughts > 0 and output == 0:
                raise ValueError(
                    f"Gemini thinking consumed all tokens ({thoughts} thought tokens, "
                    f"0 output tokens). Try increasing max_output_tokens or "
                    f"disabling thinking with thinking_budget=0."
                )
        return ""

    for cand in candidates:
        # Check finish reason for issues
        finish_reason = getattr(cand, "finish_reason", None)
        if finish_reason:
            reason_name = getattr(finish_reason, "name", str(finish_reason))
            if reason_name == "SAFETY":
                safety_ratings = getattr(cand, "safety_ratings", [])
                blocked = [
                    r for r in (safety_ratings or []) if getattr(r, "blocked", False)
                ]
                raise ValueError(f"Gemini blocked for safety: {blocked}")

        content = getattr(cand, "content", None)
        if content and hasattr(content, "parts") and content.parts:
            parts = []
            for part in content.parts:
                text = getattr(part, "text", None)
                if text:
                    parts.append(text)
            if parts:
                return "\n".join(parts)
    return ""


# --------------------------------------------------------------------------- #
# OpenAI-chat-compatible HTTP backends (OpenRouter, xAI)
# --------------------------------------------------------------------------- #


class ReasoningUnsupportedError(RuntimeError):
    """Raised when a provider rejects the reasoning parameter for a model.

    Effort support is per-model, not per-provider: ``grok-4.6`` returns
    ``"This model does not support `reasoning_effort` value `none`."`` while
    ``grok-4.3`` accepts ``none``. Backends catch this and retry once without
    the reasoning parameter rather than failing a long generation run.
    """


class BackendHTTPError(RuntimeError):
    """A non-2xx response from an HTTP backend."""

    def __init__(self, status_code: int, body: str, url: str):
        super().__init__(f"HTTP {status_code} from {url}: {body[:500]}")
        self.status_code = status_code
        self.body = body
        self.url = url


_REASONING_PARAM_MARKERS: tuple[str, ...] = (
    "reasoning_effort",
    "reasoningeffort",
    "reasoning.effort",
    "reasoning_tokens",
    "parameter reasoning",
    "`reasoning`",
    '"reasoning"',
    "thinking",
)

_REASONING_REJECTION_MARKERS: tuple[str, ...] = (
    "not support",
    "unsupported",
    "unrecognized",
    "invalid",
    "unknown",
)


# The opposite failure: the model REQUIRES reasoning and refuses to have it
# switched off. Observed verbatim from OpenRouter:
#
#   {"error":{"message":"Reasoning is mandatory for this endpoint and cannot be
#    disabled.","code":400}}
#
# This is not a "parameter unsupported" message and matched none of the markers
# above, so no retry fired. It silently destroyed three full model runs --
# x-ai/grok-4.6, google/gemini-3.1-pro-preview and google/gemini-3.7-flash all
# returned 1,500 errors and zero documents under --reasoning-effort none.
_REASONING_MANDATORY_MARKERS: tuple[str, ...] = (
    "thinking.type.disabled",
    "reasoning is mandatory",
    "cannot be disabled",
    "must be enabled",
    "required for this model",
    "reasoning cannot be turned off",
)


def _looks_like_reasoning_mandatory(body: str) -> bool:
    """Does this 400 body say reasoning cannot be switched off?"""
    lowered = body.lower()
    return any(marker in lowered for marker in _REASONING_MANDATORY_MARKERS)


def _looks_like_reasoning_rejection(body: str) -> bool:
    """Heuristic: does this 400 body complain about the reasoning parameter?

    Covers both directions -- the model rejecting the parameter, and the model
    insisting reasoning stay on. Both are fixed the same way: drop our
    reasoning directive and retry with the provider's default.
    """
    lowered = body.lower()
    if _looks_like_reasoning_mandatory(lowered):
        return True
    if not any(marker in lowered for marker in _REASONING_PARAM_MARKERS):
        return False
    return any(marker in lowered for marker in _REASONING_REJECTION_MARKERS)


def _extract_field(container, key: str):
    """Read ``key`` from a mapping or an attribute-style object."""
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(key)
    return getattr(container, key, None)


def _parse_chat_completion(
    data: dict, *, fallback_provider: str | None = None
) -> GenerationResult:
    """Normalize an OpenAI-style ``chat.completion`` body.

    Field locations confirmed against live responses:

    * text: ``choices[0].message.content``
    * provider (OpenRouter only): top-level ``provider``
    * usage: ``usage.prompt_tokens`` / ``usage.completion_tokens``
    * reasoning: ``usage.completion_tokens_details.reasoning_tokens``
    * resolved model: top-level ``model``
    """
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"Response contained no choices: {str(data)[:300]}")
    message = choices[0].get("message") or {}
    text = message.get("content") or ""
    if not text.strip():
        finish = choices[0].get("finish_reason")
        raise ValueError(
            f"Received empty output (finish_reason={finish}, usage={data.get('usage')})"
        )

    usage = data.get("usage") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return GenerationResult(
        text=text,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        provider_served=data.get("provider") or fallback_provider,
        reasoning_tokens=completion_details.get("reasoning_tokens"),
        model_resolved=data.get("model"),
    )


class OpenAIChatCompatibleBackend(_BaseBackend):
    """Base backend for providers speaking the OpenAI ``/chat/completions`` API.

    Implemented over ``httpx`` rather than the ``openai`` SDK: ``httpx`` is a
    declared dependency of this project and the vendor SDKs are not, so this
    keeps the new backends importable without adding dependencies.

    Subclasses override ``provider``, ``default_base_url``, ``api_key_env_vars``
    and ``_apply_reasoning`` (each provider spells the reasoning knob
    differently).
    """

    provider = "openai_chat"
    default_base_url: str = "https://api.openai.com/v1"
    api_key_env_vars: tuple[str, ...] = ("OPENAI_API_KEY",)

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        client=None,
        async_client=None,
        reasoning: ReasoningConfig | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout_seconds: float = 600.0,
        retry_without_reasoning: bool = True,
    ):
        self.model = model
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self._api_key = api_key or self._api_key_from_env()
        self._client = client
        self._async_client = async_client
        self._reasoning = reasoning
        self._extra_headers = dict(extra_headers or {})
        self._extra_body = dict(extra_body or {})
        self._timeout_seconds = timeout_seconds
        self._retry_without_reasoning = retry_without_reasoning

        if self._api_key is None and client is None and async_client is None:
            raise ValueError(
                f"{type(self).__name__} requires an API key. Set one of "
                f"{', '.join(self.api_key_env_vars)} or pass api_key=..."
            )

    @classmethod
    def _api_key_from_env(cls) -> str | None:
        for name in cls.api_key_env_vars:
            value = os.environ.get(name)
            if value:
                return value
        return None

    # -- client plumbing --------------------------------------------------- #

    def _get_client(self):
        import httpx

        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout_seconds)
        return self._client

    def _get_async_client(self):
        import httpx

        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self._timeout_seconds)
        return self._async_client

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self._extra_headers)
        return headers

    @staticmethod
    def _decode(response, url: str) -> dict:
        status = getattr(response, "status_code", 200)
        if status >= 400:
            body = getattr(response, "text", "") or ""
            if status == 400 and _looks_like_reasoning_rejection(body):
                raise ReasoningUnsupportedError(body)
            raise BackendHTTPError(status, body, url)
        return response.json()

    def _post(self, url: str, payload: dict) -> dict:
        response = self._get_client().post(url, json=payload, headers=self._headers())
        return self._decode(response, url)

    async def _post_async(self, url: str, payload: dict) -> dict:
        response = await self._get_async_client().post(
            url, json=payload, headers=self._headers()
        )
        return self._decode(response, url)

    def _get(self, url: str) -> dict:
        response = self._get_client().get(url, headers=self._headers())
        return self._decode(response, url)

    # -- payload construction ---------------------------------------------- #

    def _apply_reasoning(self, payload: dict[str, Any]) -> None:
        """Attach the provider-specific reasoning knob. Default: OpenAI style."""
        if self._reasoning is None:
            return
        reasoning = self._reasoning.to_openai_responses()
        if reasoning is not None:
            payload["reasoning_effort"] = reasoning["effort"]

    def _apply_routing(self, payload: dict[str, Any]) -> None:
        """Hook for gateway-specific routing preferences. Default: no-op."""

    def build_payload(
        self,
        request: GenerationRequest,
        params: GenerationParams,
        *,
        include_reasoning: bool = True,
        include_model: bool = True,
    ) -> dict[str, Any]:
        """Build a ``/chat/completions`` request body."""
        payload: dict[str, Any] = {}
        if include_model:
            payload["model"] = self.model
        payload["messages"] = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.prompt},
        ]
        payload["max_tokens"] = params.max_output_tokens
        payload["temperature"] = params.temperature
        payload["top_p"] = params.top_p
        if include_reasoning:
            self._apply_reasoning(payload)
        self._apply_routing(payload)
        payload.update(self._extra_body)
        return payload

    # -- generation --------------------------------------------------------- #

    @property
    def _completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def generate(
        self, request: GenerationRequest, params: GenerationParams
    ) -> GenerationResult:
        try:
            data = self._post(
                self._completions_url, self.build_payload(request, params)
            )
        except ReasoningUnsupportedError:
            if not self._retry_without_reasoning:
                raise
            data = self._post(
                self._completions_url,
                self.build_payload(request, params, include_reasoning=False),
            )
        return _parse_chat_completion(data, fallback_provider=self.provider)

    async def generate_async(
        self, request: GenerationRequest, params: GenerationParams
    ) -> GenerationResult:
        try:
            data = await self._post_async(
                self._completions_url, self.build_payload(request, params)
            )
        except ReasoningUnsupportedError:
            if not self._retry_without_reasoning:
                raise
            data = await self._post_async(
                self._completions_url,
                self.build_payload(request, params, include_reasoning=False),
            )
        return _parse_chat_completion(data, fallback_provider=self.provider)


@dataclass
class ProviderRouting:
    """OpenRouter provider-routing preferences.

    OpenRouter load-balances a single model id across upstream providers that
    may differ in quantization, so pinning matters for corpus reproducibility.
    Field names match OpenRouter's ``provider`` request object.
    """

    order: list[str] | None = None
    only: list[str] | None = None
    ignore: list[str] | None = None
    quantizations: list[str] | None = None
    sort: str | None = None
    allow_fallbacks: bool | None = None
    require_parameters: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def pin(cls, provider_slug: str) -> ProviderRouting:
        """Pin every request to one upstream provider, with no fallbacks."""
        return cls(order=[provider_slug], allow_fallbacks=False)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.order:
            out["order"] = list(self.order)
        if self.only:
            out["only"] = list(self.only)
        if self.ignore:
            out["ignore"] = list(self.ignore)
        if self.quantizations:
            out["quantizations"] = list(self.quantizations)
        if self.sort:
            out["sort"] = self.sort
        if self.allow_fallbacks is not None:
            out["allow_fallbacks"] = self.allow_fallbacks
        if self.require_parameters is not None:
            out["require_parameters"] = self.require_parameters
        out.update(self.extra)
        return out


class OpenRouterBackend(OpenAIChatCompatibleBackend):
    """Backend for OpenRouter, the gateway that unlocks open-weight families.

    Two OpenRouter-specific behaviours matter for SHELF:

    1. **Serving provider capture.** OpenRouter routes one model id to several
       upstream backends whose quantization can differ, and reports the one it
       used as a top-level ``provider`` string. That lands in
       :attr:`GenerationResult.provider_served`. Use ``routing=`` to pin.

    2. **Batch API.** ``:batch`` model ids are billed at ~50% but are *only*
       reachable through ``POST /api/beta/batches`` -- a synchronous call to
       ``google/gemini-3.7-flash:batch`` returns
       ``"This model is only available through the Batch API."``. So the suffix
       alone is not sufficient; :meth:`generate_batch` implements the batch
       endpoint (submit inline ``requests``, poll, read inline ``results``).
    """

    provider = "openrouter"
    default_base_url = "https://openrouter.ai/api/v1"
    api_key_env_vars = ("OPENROUTER_API_KEY",)
    batch_model_suffix = ":batch"
    #: Statuses at which polling stops.
    terminal_batch_statuses = frozenset({"completed", "failed", "expired", "cancelled"})

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        batch_base_url: str | None = None,
        api_key: str | None = None,
        client=None,
        async_client=None,
        reasoning: ReasoningConfig | None = None,
        routing: ProviderRouting | None = None,
        app_url: str | None = None,
        app_title: str | None = None,
        use_batch_api: bool = False,
        poll_seconds: float = 30.0,
        timeout_seconds: float = 600.0,
        batch_timeout_seconds: float = 86_400.0,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        retry_without_reasoning: bool = True,
    ):
        headers = dict(extra_headers or {})
        # OpenRouter's optional app-attribution headers.
        if app_url:
            headers.setdefault("HTTP-Referer", app_url)
        if app_title:
            headers.setdefault("X-Title", app_title)

        super().__init__(
            model,
            base_url=base_url,
            api_key=api_key,
            client=client,
            async_client=async_client,
            reasoning=reasoning,
            extra_headers=headers,
            extra_body=extra_body,
            timeout_seconds=timeout_seconds,
            retry_without_reasoning=retry_without_reasoning,
        )
        self.routing = routing
        # A ":batch" model id can only be used through the batch endpoint, so
        # selecting one implies batch mode.
        self._use_batch_api = use_batch_api or self.is_batch_model(model)
        self._poll_seconds = poll_seconds
        self._batch_timeout_seconds = batch_timeout_seconds
        self._batch_base_url = (
            batch_base_url or self.base_url.replace("/api/v1", "/api/beta")
        ).rstrip("/")

    # -- model id helpers --------------------------------------------------- #

    @classmethod
    def is_batch_model(cls, model: str) -> bool:
        """True if ``model`` is a batch-only ``:batch`` variant slug."""
        return model.endswith(cls.batch_model_suffix)

    @classmethod
    def batch_model_id(cls, model: str) -> str:
        """Return the ``:batch`` variant of ``model`` (idempotent).

        Not every model has a batch variant -- check ``GET /api/v1/models`` for
        the id before using this.
        """
        if cls.is_batch_model(model):
            return model
        return f"{model}{cls.batch_model_suffix}"

    @property
    def uses_batch_api(self) -> bool:
        return self._use_batch_api

    # -- request shaping ---------------------------------------------------- #

    def _apply_reasoning(self, payload: dict[str, Any]) -> None:
        if self._reasoning is None:
            return
        payload["reasoning"] = self._reasoning.to_openrouter()

    def _apply_routing(self, payload: dict[str, Any]) -> None:
        if self.routing is None:
            return
        routing = self.routing.to_dict()
        if routing:
            payload["provider"] = routing

    # -- batch -------------------------------------------------------------- #

    def generate_batch(
        self, requests: list[GenerationRequest], params: GenerationParams
    ) -> list[GenerationResult]:
        if not self._use_batch_api:
            return super().generate_batch(requests, params)
        if not requests:
            return []
        return self._generate_batch_impl(requests, params)

    def _build_batch_body(
        self, requests: list[GenerationRequest], params: GenerationParams
    ) -> dict[str, Any]:
        """Build the ``POST /api/beta/batches`` body.

        Key order matters: OpenRouter stream-parses the body and returns 400 if
        ``requests`` is serialized before ``endpoint`` and ``model``.
        """
        body: dict[str, Any] = {
            "endpoint": "/v1/chat/completions",
            "model": self.model,
        }
        body["requests"] = [
            {
                "custom_id": f"req-{i}",
                # The per-request body inherits the batch-level model.
                "body": self.build_payload(req, params, include_model=False),
            }
            for i, req in enumerate(requests)
        ]
        return body

    def _generate_batch_impl(
        self, requests: list[GenerationRequest], params: GenerationParams
    ) -> list[GenerationResult]:
        batches_url = f"{self._batch_base_url}/batches"
        batch = self._post(batches_url, self._build_batch_body(requests, params))
        batch_id = batch.get("id")
        if not batch_id:
            raise RuntimeError(f"OpenRouter batch submission returned no id: {batch}")

        deadline = time.monotonic() + self._batch_timeout_seconds
        status = batch.get("status")
        while status not in self.terminal_batch_statuses:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"OpenRouter batch {batch_id} did not complete before timeout"
                )
            time.sleep(self._poll_seconds)
            batch = self._get(f"{batches_url}/{batch_id}")
            status = batch.get("status")

        if status != "completed":
            raise RuntimeError(
                f"OpenRouter batch {batch_id} finished with status={status}: "
                f"{batch.get('error')}"
            )

        results = batch.get("results") or []
        by_custom_id: dict[str, dict] = {}
        for item in results:
            custom_id = item.get("custom_id")
            if custom_id is not None:
                by_custom_id[custom_id] = item

        outputs: list[GenerationResult] = []
        for i in range(len(requests)):
            custom_id = f"req-{i}"
            item = by_custom_id.get(custom_id)
            if item is None:
                raise ValueError(
                    f"OpenRouter batch {batch_id} missing result for {custom_id}"
                )
            error = item.get("error")
            if error:
                raise RuntimeError(f"OpenRouter batch item {custom_id} failed: {error}")
            response = item.get("response") or {}
            body = response.get("body") or {}
            outputs.append(
                _parse_chat_completion(body, fallback_provider=self.provider)
            )
        return outputs


class XAIBackend(OpenAIChatCompatibleBackend):
    """Backend for xAI / Grok (``https://api.x.ai/v1``).

    xAI is OpenAI-chat-compatible and reports no serving provider, so
    :attr:`GenerationResult.provider_served` falls back to ``"xai"``. It does
    resolve model aliases, which is why :attr:`GenerationResult.model_resolved`
    is worth recording (``grok-4.20-non-reasoning`` ->
    ``grok-4.20-0309-non-reasoning``).

    Reasoning is controlled with the top-level ``reasoning_effort`` string.
    Accepted values are per-model: ``grok-4.3`` accepts ``"none"`` (and reports
    0 reasoning tokens), ``grok-4.6`` rejects it with a 400, and non-reasoning
    models reject the parameter entirely. With ``retry_without_reasoning=True``
    (the default) such a 400 is retried once with the parameter dropped.

    xAI has a batch API, but it uses a create/add/monitor flow unlike the
    OpenAI batch shape; it is not implemented here, so ``generate_batch``
    falls back to sequential calls.
    """

    provider = "xai"
    default_base_url = "https://api.x.ai/v1"
    api_key_env_vars = ("XAI_API_KEY",)

    def _apply_reasoning(self, payload: dict[str, Any]) -> None:
        if self._reasoning is None:
            return
        effort = self._reasoning.to_xai_effort()
        if effort is not None:
            payload["reasoning_effort"] = effort
