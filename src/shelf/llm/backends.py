"""Common LLM backend interface with provider-specific implementations.

This abstracts provider-specific SDK calls behind a small protocol so the rest
of the codebase can switch providers (OpenAI, Anthropic, Gemini) without
rewriting generation logic. Each backend supports per-request generation plus a
batch method; if a true batch API is unavailable at runtime, the batch method
falls back to per-item calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import asyncio
import time


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


@dataclass
class GenerationResult:
    """Normalized generation result with optional token usage."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


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
    ):
        import openai

        self.model = model
        self._service_tier = service_tier
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
        return kwargs

    def generate(
        self, request: GenerationRequest, params: GenerationParams
    ) -> GenerationResult:
        response = self._get_client().responses.create(  # type: ignore[attr-defined]
            **self._build_request_kwargs(request, params)
        )
        text = getattr(response, "output_text", "") or ""
        if not text.strip():
            raise ValueError("Received empty output from OpenAI")
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage else None
        output_tokens = getattr(usage, "output_tokens", None) if usage else None
        return GenerationResult(
            text=text, input_tokens=input_tokens, output_tokens=output_tokens
        )

    async def generate_async(
        self, request: GenerationRequest, params: GenerationParams
    ) -> GenerationResult:
        response = await self._get_async_client().responses.create(  # type: ignore[attr-defined]
            **self._build_request_kwargs(request, params)
        )
        text = getattr(response, "output_text", "") or ""
        if not text.strip():
            raise ValueError("Received empty output from OpenAI")
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage else None
        output_tokens = getattr(usage, "output_tokens", None) if usage else None
        return GenerationResult(
            text=text, input_tokens=input_tokens, output_tokens=output_tokens
        )


# --------------------------------------------------------------------------- #
# Anthropic Messages backend (with batch hook)
# --------------------------------------------------------------------------- #


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
    ):
        import anthropic

        self.model = model
        self._client: anthropic.Anthropic | None = client
        self._use_batch_api = use_batch_api
        self._poll_seconds = poll_seconds
        self._timeout_seconds = timeout_seconds

    def _get_client(self):
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def _clamp_temperature(self, temperature: float) -> float:
        """Clamp temperature to Anthropic's valid range (0.0-1.0)."""
        return max(0.0, min(1.0, temperature))

    def generate(
        self, request: GenerationRequest, params: GenerationParams
    ) -> GenerationResult:
        client = self._get_client()
        # Note: Claude 4.x models don't allow both temperature and top_p.
        # We only use temperature (clamped to 0-1 range).
        message = client.messages.create(
            model=self.model,
            max_tokens=params.max_output_tokens,
            temperature=self._clamp_temperature(params.temperature),
            system=request.system_prompt,
            messages=[{"role": "user", "content": request.prompt}],
        )
        usage = getattr(message, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage else None
        output_tokens = getattr(usage, "output_tokens", None) if usage else None
        return GenerationResult(
            text=_anthropic_content_to_text(message),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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
    ):
        import os

        from google import genai
        from google.genai import types

        self.model = model
        self._use_batch_api = use_batch_api
        self._poll_seconds = poll_seconds
        self._timeout_seconds = timeout_seconds
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
        response = self._client.models.generate_content(
            model=self.model,
            contents=request.prompt,
            config=self._build_config(request, params),
        )
        text = _gemini_response_text(response)
        if not text.strip():
            # Build diagnostic info for empty responses
            diag = _gemini_empty_response_diagnostic(response)
            raise ValueError(f"Received empty output from Gemini: {diag}")
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        output_tokens = (
            getattr(usage, "candidates_token_count", None) if usage else None
        )
        return GenerationResult(
            text=text, input_tokens=input_tokens, output_tokens=output_tokens
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
            usage = getattr(response, "usage_metadata", None) if response else None
            input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
            output_tokens = (
                getattr(usage, "candidates_token_count", None) if usage else None
            )
            outputs.append(
                GenerationResult(
                    text=text, input_tokens=input_tokens, output_tokens=output_tokens
                )
            )

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
