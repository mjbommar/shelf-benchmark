"""Unit tests for shelf.llm.backends.

Covers:
- GenerationResult backward compatibility and the new provenance fields
- ReasoningConfig translation to each provider's parameter shape
- OpenAI-style chat-completion response parsing, using response bodies
  captured from live OpenRouter and xAI calls
- OpenRouterBackend: payload construction, provider capture, provider pinning,
  ``:batch`` model handling and the /api/beta/batches flow
- XAIBackend: reasoning_effort handling and the retry-without-reasoning path

No network access: every backend is driven with an injected fake HTTP client.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any

import pytest
from shelf.llm.backends import (
    REASONING_EFFORTS,
    BackendHTTPError,
    GenerationParams,
    GenerationRequest,
    GenerationResult,
    LLMBackend,
    OpenAIChatCompatibleBackend,
    OpenRouterBackend,
    ProviderRouting,
    ReasoningConfig,
    ReasoningUnsupportedError,
    XAIBackend,
    _looks_like_reasoning_rejection,
    _parse_chat_completion,
)

# --------------------------------------------------------------------------- #
# Fixtures: real response bodies captured from live API calls (2026-08-26)
# --------------------------------------------------------------------------- #

# POST https://openrouter.ai/api/v1/chat/completions, meta-llama/llama-4-maverick
OPENROUTER_RESPONSE: dict[str, Any] = {
    "id": "gen-1787793050-SZBKS9JndUfYTSbvfYYk",
    "object": "chat.completion",
    "created": 1787793050,
    "model": "meta-llama/llama-4-maverick",
    "provider": "DeepInfra",
    "system_fingerprint": None,
    "service_tier": None,
    "choices": [
        {
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop",
            "native_finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Hi.",
                "refusal": None,
                "reasoning": None,
            },
        }
    ],
    "usage": {
        "prompt_tokens": 23,
        "completion_tokens": 3,
        "total_tokens": 26,
        "cost": 7e-06,
        "is_byok": False,
        "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0, "image_tokens": 0},
    },
}

# Same endpoint, deepseek/deepseek-v4-flash-0731 with reasoning left enabled.
OPENROUTER_REASONING_RESPONSE: dict[str, Any] = {
    "id": "gen-1787793067-yihi14Pit04wZGttNRK9",
    "object": "chat.completion",
    "created": 1787793067,
    "model": "deepseek/deepseek-v4-flash-0731",
    "provider": "Ambient",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "391", "reasoning": "..."},
        }
    ],
    "usage": {
        "prompt_tokens": 17,
        "completion_tokens": 71,
        "total_tokens": 88,
        "cost": 1.414e-05,
        "completion_tokens_details": {"reasoning_tokens": 42},
    },
}

# POST https://api.x.ai/v1/chat/completions, model alias grok-4.20-non-reasoning
XAI_RESPONSE: dict[str, Any] = {
    "id": "e3a32716-f49d-968f-b0b7-963eaf4b1975",
    "object": "chat.completion",
    "created": 1787793080,
    "model": "grok-4.20-0309-non-reasoning",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hi.", "refusal": None},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 194,
        "completion_tokens": 2,
        "total_tokens": 196,
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
    "system_fingerprint": "fp_e309ea75b5",
    "service_tier": "default",
}

# Live 400 bodies observed from api.x.ai.
XAI_UNSUPPORTED_PARAM_BODY = (
    '{"code":"invalid-argument","error":"Model grok-4.20-0309-non-reasoning '
    'does not support parameter reasoningEffort."}'
)
XAI_UNSUPPORTED_VALUE_BODY = (
    '{"code":"invalid-argument","error":"This model does not support '
    '`reasoning_effort` value `none`."}'
)
# Live 404 body from a sync call to a ":batch" model id.
OPENROUTER_BATCH_ONLY_BODY = (
    '{"error":{"message":"This model is only available through the Batch API. '
    'Use the /api/beta/batches endpoint instead.","code":404}}'
)


# --------------------------------------------------------------------------- #
# Fake HTTP clients
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, payload=None, *, status_code: int = 200, text: str = ""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class FakeHTTPClient:
    """Minimal stand-in for httpx.Client recording posts and gets."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.posts: list[dict] = []
        self.gets: list[dict] = []

    def _next(self):
        if not self._responses:
            raise AssertionError("FakeHTTPClient ran out of queued responses")
        return self._responses.pop(0)

    def post(self, url, json=None, headers=None):
        self.posts.append({"url": url, "json": json, "headers": headers})
        return self._next()

    def get(self, url, headers=None):
        self.gets.append({"url": url, "headers": headers})
        return self._next()


class FakeAsyncHTTPClient(FakeHTTPClient):
    async def post(self, url, json=None, headers=None):
        return super().post(url, json=json, headers=headers)


PARAMS = GenerationParams(temperature=0.8, top_p=0.95, max_output_tokens=1024)
REQUEST = GenerationRequest(prompt="Write a lecture.", system_prompt="You are terse.")


# --------------------------------------------------------------------------- #
# GenerationResult
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestGenerationResult:
    def test_positional_construction_is_backward_compatible(self):
        result = GenerationResult("body", 10, 20)
        assert result.text == "body"
        assert result.input_tokens == 10
        assert result.output_tokens == 20

    def test_new_fields_default_to_none(self):
        result = GenerationResult("body")
        assert result.provider_served is None
        assert result.reasoning_tokens is None
        assert result.model_resolved is None

    def test_new_fields_are_settable(self):
        result = GenerationResult(
            text="body",
            provider_served="DeepInfra",
            reasoning_tokens=42,
            model_resolved="deepseek/deepseek-v4-flash-0731",
        )
        assert result.provider_served == "DeepInfra"
        assert result.reasoning_tokens == 42
        assert result.model_resolved == "deepseek/deepseek-v4-flash-0731"


# --------------------------------------------------------------------------- #
# ReasoningConfig
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestReasoningConfig:
    def test_default_config_is_not_off(self):
        assert ReasoningConfig().is_off is False

    def test_off_is_off(self):
        assert ReasoningConfig.off().is_off is True

    def test_minimal_is_not_off(self):
        assert ReasoningConfig.minimal().is_off is False

    def test_zero_budget_reads_as_off(self):
        assert ReasoningConfig(max_tokens=0).is_off is True

    def test_openrouter_off_uses_verified_shape(self):
        # Confirmed live: {"enabled": false, "exclude": true} yields
        # usage.completion_tokens_details.reasoning_tokens == 0.
        assert ReasoningConfig.off().to_openrouter() == {
            "enabled": False,
            "exclude": True,
        }

    def test_openrouter_effort_and_max_tokens_are_mutually_exclusive(self):
        payload = ReasoningConfig(effort="low", max_tokens=2000).to_openrouter()
        assert payload["effort"] == "low"
        assert "max_tokens" not in payload

    def test_openrouter_uses_max_tokens_when_no_effort(self):
        payload = ReasoningConfig(max_tokens=2000).to_openrouter()
        assert payload["max_tokens"] == 2000
        assert "effort" not in payload

    def test_openai_responses_shape(self):
        assert ReasoningConfig.off().to_openai_responses() == {"effort": "none"}
        assert ReasoningConfig.minimal().to_openai_responses() == {"effort": "minimal"}
        assert ReasoningConfig().to_openai_responses() is None

    def test_xai_effort(self):
        assert ReasoningConfig.off().to_xai_effort() == "none"
        assert ReasoningConfig(effort="low").to_xai_effort() == "low"
        assert ReasoningConfig().to_xai_effort() is None

    def test_anthropic_off_disables_thinking(self):
        thinking, output_config = ReasoningConfig.off().to_anthropic()
        assert thinking == {"type": "disabled"}
        # "none" has no Anthropic effort equivalent.
        assert output_config is None

    def test_anthropic_budget_enables_thinking(self):
        thinking, output_config = ReasoningConfig(max_tokens=2000).to_anthropic()
        assert thinking == {"type": "enabled", "budget_tokens": 2000}
        assert output_config is None

    def test_anthropic_effort_maps_to_output_config(self):
        _, output_config = ReasoningConfig(effort="medium").to_anthropic()
        assert output_config == {"effort": "medium"}

    def test_anthropic_minimal_maps_to_low(self):
        _, output_config = ReasoningConfig.minimal().to_anthropic()
        assert output_config == {"effort": "low"}

    def test_gemini_budget(self):
        assert ReasoningConfig.off().to_gemini_thinking_budget() == 0
        assert ReasoningConfig(max_tokens=512).to_gemini_thinking_budget() == 512
        assert ReasoningConfig(effort="high").to_gemini_thinking_budget() == 8192
        assert ReasoningConfig().to_gemini_thinking_budget() is None

    def test_all_canonical_efforts_translate(self):
        for effort in REASONING_EFFORTS:
            config = ReasoningConfig(effort=effort)
            assert config.to_openai_responses() == {"effort": effort}
            assert config.to_xai_effort() == effort
            assert isinstance(config.to_openrouter(), dict)


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestParseChatCompletion:
    def test_openrouter_body_yields_provider_and_usage(self):
        result = _parse_chat_completion(OPENROUTER_RESPONSE)
        assert result.text == "Hi."
        assert result.input_tokens == 23
        assert result.output_tokens == 3
        # OpenRouter reports the serving provider at the top level.
        assert result.provider_served == "DeepInfra"
        assert result.reasoning_tokens == 0
        assert result.model_resolved == "meta-llama/llama-4-maverick"

    def test_openrouter_reasoning_tokens_are_captured(self):
        result = _parse_chat_completion(OPENROUTER_REASONING_RESPONSE)
        assert result.provider_served == "Ambient"
        assert result.reasoning_tokens == 42
        assert result.output_tokens == 71

    def test_xai_body_has_no_provider_and_uses_fallback(self):
        result = _parse_chat_completion(XAI_RESPONSE, fallback_provider="xai")
        assert result.provider_served == "xai"
        # xAI resolves aliases, so model_resolved differs from the request.
        assert result.model_resolved == "grok-4.20-0309-non-reasoning"
        assert result.reasoning_tokens == 0

    def test_missing_choices_raises(self):
        with pytest.raises(ValueError, match="no choices"):
            _parse_chat_completion({"choices": []})

    def test_empty_content_raises_with_diagnostics(self):
        body = copy.deepcopy(OPENROUTER_RESPONSE)
        body["choices"][0]["message"]["content"] = "   "
        body["choices"][0]["finish_reason"] = "length"
        with pytest.raises(ValueError, match="finish_reason=length"):
            _parse_chat_completion(body)

    def test_missing_usage_is_tolerated(self):
        body = copy.deepcopy(OPENROUTER_RESPONSE)
        del body["usage"]
        result = _parse_chat_completion(body)
        assert result.input_tokens is None
        assert result.reasoning_tokens is None


@pytest.mark.unit
class TestReasoningRejectionDetection:
    def test_detects_live_xai_unsupported_parameter_body(self):
        assert _looks_like_reasoning_rejection(XAI_UNSUPPORTED_PARAM_BODY)

    def test_detects_live_xai_unsupported_value_body(self):
        assert _looks_like_reasoning_rejection(XAI_UNSUPPORTED_VALUE_BODY)

    def test_does_not_fire_on_unrelated_errors(self):
        assert not _looks_like_reasoning_rejection(OPENROUTER_BATCH_ONLY_BODY)
        assert not _looks_like_reasoning_rejection('{"error":"rate limit exceeded"}')


# --------------------------------------------------------------------------- #
# Shared HTTP backend behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestOpenAIChatCompatibleBackend:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="XAI_API_KEY"):
            XAIBackend("grok-4.6")

    def test_api_key_is_read_from_env(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "env-key")
        backend = XAIBackend("grok-4.6")
        assert backend._headers()["Authorization"] == "Bearer env-key"

    def test_openrouter_key_env_var(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            OpenRouterBackend("meta-llama/llama-4-maverick")

    def test_injected_client_does_not_require_key(self):
        backend = XAIBackend("grok-4.6", api_key=None, client=FakeHTTPClient([]))
        assert backend is not None

    def test_backends_satisfy_the_llm_backend_protocol(self):
        client = FakeHTTPClient([])
        assert isinstance(
            OpenRouterBackend("m", api_key="k", client=client), LLMBackend
        )
        assert isinstance(XAIBackend("m", api_key="k", client=client), LLMBackend)

    def test_base_urls(self):
        assert OpenRouterBackend.default_base_url == "https://openrouter.ai/api/v1"
        assert XAIBackend.default_base_url == "https://api.x.ai/v1"

    def test_base_url_override_strips_trailing_slash(self):
        backend = XAIBackend(
            "m", api_key="k", client=FakeHTTPClient([]), base_url="http://local/v1/"
        )
        assert backend._completions_url == "http://local/v1/chat/completions"

    def test_payload_carries_sampling_params_and_messages(self):
        backend = XAIBackend("grok-4.6", api_key="k", client=FakeHTTPClient([]))
        payload = backend.build_payload(REQUEST, PARAMS)
        assert payload["model"] == "grok-4.6"
        assert payload["messages"] == [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Write a lecture."},
        ]
        assert payload["max_tokens"] == 1024
        assert payload["temperature"] == 0.8
        assert payload["top_p"] == 0.95

    def test_extra_body_is_merged(self):
        backend = XAIBackend(
            "grok-4.6",
            api_key="k",
            client=FakeHTTPClient([]),
            extra_body={"seed": 7},
        )
        assert backend.build_payload(REQUEST, PARAMS)["seed"] == 7

    def test_http_error_raises_backend_http_error(self):
        client = FakeHTTPClient(
            [FakeResponse(status_code=500, text="upstream exploded")]
        )
        backend = XAIBackend("grok-4.6", api_key="k", client=client)
        with pytest.raises(BackendHTTPError) as excinfo:
            backend.generate(REQUEST, PARAMS)
        assert excinfo.value.status_code == 500

    def test_batch_only_model_error_is_not_swallowed(self):
        client = FakeHTTPClient(
            [FakeResponse(status_code=404, text=OPENROUTER_BATCH_ONLY_BODY)]
        )
        backend = OpenRouterBackend(
            "google/gemini-3.7-flash", api_key="k", client=client
        )
        with pytest.raises(BackendHTTPError):
            backend.generate(REQUEST, PARAMS)

    def test_default_batch_falls_back_to_sequential(self):
        client = FakeHTTPClient([FakeResponse(XAI_RESPONSE) for _ in range(3)])
        backend = XAIBackend("grok-4.6", api_key="k", client=client)
        results = backend.generate_batch([REQUEST] * 3, PARAMS)
        assert len(results) == 3
        assert len(client.posts) == 3

    def test_generate_async_uses_the_async_client(self):
        client = FakeAsyncHTTPClient([FakeResponse(XAI_RESPONSE)])
        backend = XAIBackend("grok-4.6", api_key="k", async_client=client)
        result = asyncio.run(backend.generate_async(REQUEST, PARAMS))
        assert result.text == "Hi."
        assert result.provider_served == "xai"


# --------------------------------------------------------------------------- #
# XAIBackend
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestXAIBackend:
    def test_provider_name(self):
        assert XAIBackend.provider == "xai"

    def test_reasoning_effort_is_a_top_level_string(self):
        backend = XAIBackend(
            "grok-4.6",
            api_key="k",
            client=FakeHTTPClient([]),
            reasoning=ReasoningConfig(effort="low"),
        )
        payload = backend.build_payload(REQUEST, PARAMS)
        assert payload["reasoning_effort"] == "low"
        assert "reasoning" not in payload

    def test_no_reasoning_config_means_no_parameter(self):
        backend = XAIBackend("grok-4.6", api_key="k", client=FakeHTTPClient([]))
        assert "reasoning_effort" not in backend.build_payload(REQUEST, PARAMS)

    def test_generate_records_resolved_model(self):
        client = FakeHTTPClient([FakeResponse(XAI_RESPONSE)])
        backend = XAIBackend("grok-4.20-non-reasoning", api_key="k", client=client)
        result = backend.generate(REQUEST, PARAMS)
        assert client.posts[0]["url"] == "https://api.x.ai/v1/chat/completions"
        assert result.model_resolved == "grok-4.20-0309-non-reasoning"
        assert result.provider_served == "xai"

    def test_unsupported_reasoning_parameter_is_retried_without_it(self):
        client = FakeHTTPClient(
            [
                FakeResponse(status_code=400, text=XAI_UNSUPPORTED_PARAM_BODY),
                FakeResponse(XAI_RESPONSE),
            ]
        )
        backend = XAIBackend(
            "grok-4.20-0309-non-reasoning",
            api_key="k",
            client=client,
            reasoning=ReasoningConfig.off(),
        )
        result = backend.generate(REQUEST, PARAMS)
        assert result.text == "Hi."
        assert len(client.posts) == 2
        assert client.posts[0]["json"]["reasoning_effort"] == "none"
        assert "reasoning_effort" not in client.posts[1]["json"]

    def test_unsupported_reasoning_value_is_retried_without_it(self):
        client = FakeHTTPClient(
            [
                FakeResponse(status_code=400, text=XAI_UNSUPPORTED_VALUE_BODY),
                FakeResponse(XAI_RESPONSE),
            ]
        )
        backend = XAIBackend(
            "grok-4.6", api_key="k", client=client, reasoning=ReasoningConfig.off()
        )
        assert backend.generate(REQUEST, PARAMS).text == "Hi."

    def test_retry_can_be_disabled(self):
        client = FakeHTTPClient(
            [FakeResponse(status_code=400, text=XAI_UNSUPPORTED_VALUE_BODY)]
        )
        backend = XAIBackend(
            "grok-4.6",
            api_key="k",
            client=client,
            reasoning=ReasoningConfig.off(),
            retry_without_reasoning=False,
        )
        with pytest.raises(ReasoningUnsupportedError):
            backend.generate(REQUEST, PARAMS)


# --------------------------------------------------------------------------- #
# OpenRouterBackend
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestOpenRouterBackend:
    def test_provider_name(self):
        assert OpenRouterBackend.provider == "openrouter"

    def test_reasoning_is_a_nested_object(self):
        backend = OpenRouterBackend(
            "deepseek/deepseek-v4-flash-0731",
            api_key="k",
            client=FakeHTTPClient([]),
            reasoning=ReasoningConfig.off(),
        )
        payload = backend.build_payload(REQUEST, PARAMS)
        assert payload["reasoning"] == {"enabled": False, "exclude": True}
        assert "reasoning_effort" not in payload

    def test_provider_routing_is_serialized(self):
        backend = OpenRouterBackend(
            "deepseek/deepseek-v4-flash-0731",
            api_key="k",
            client=FakeHTTPClient([]),
            routing=ProviderRouting.pin("deepinfra"),
        )
        payload = backend.build_payload(REQUEST, PARAMS)
        assert payload["provider"] == {
            "order": ["deepinfra"],
            "allow_fallbacks": False,
        }

    def test_provider_routing_full_shape(self):
        routing = ProviderRouting(
            only=["deepinfra"],
            ignore=["novita"],
            quantizations=["fp8"],
            sort="price",
            require_parameters=True,
            extra={"zdr": True},
        )
        assert routing.to_dict() == {
            "only": ["deepinfra"],
            "ignore": ["novita"],
            "quantizations": ["fp8"],
            "sort": "price",
            "require_parameters": True,
            "zdr": True,
        }

    def test_no_routing_means_no_provider_key(self):
        backend = OpenRouterBackend("m", api_key="k", client=FakeHTTPClient([]))
        assert "provider" not in backend.build_payload(REQUEST, PARAMS)

    def test_generate_captures_the_serving_provider(self):
        client = FakeHTTPClient([FakeResponse(OPENROUTER_RESPONSE)])
        backend = OpenRouterBackend(
            "meta-llama/llama-4-maverick", api_key="k", client=client
        )
        result = backend.generate(REQUEST, PARAMS)
        assert client.posts[0]["url"] == "https://openrouter.ai/api/v1/chat/completions"
        assert result.provider_served == "DeepInfra"
        assert result.model_resolved == "meta-llama/llama-4-maverick"
        assert result.reasoning_tokens == 0

    def test_app_attribution_headers(self):
        backend = OpenRouterBackend(
            "m",
            api_key="k",
            client=FakeHTTPClient([]),
            app_url="https://example.org/shelf",
            app_title="SHELF",
        )
        headers = backend._headers()
        assert headers["HTTP-Referer"] == "https://example.org/shelf"
        assert headers["X-Title"] == "SHELF"

    # -- batch ------------------------------------------------------------- #

    def test_batch_model_id_is_idempotent(self):
        assert (
            OpenRouterBackend.batch_model_id("anthropic/claude-opus-5")
            == "anthropic/claude-opus-5:batch"
        )
        assert (
            OpenRouterBackend.batch_model_id("anthropic/claude-opus-5:batch")
            == "anthropic/claude-opus-5:batch"
        )

    def test_is_batch_model(self):
        assert OpenRouterBackend.is_batch_model("openai/gpt-5.6-luna:batch")
        assert not OpenRouterBackend.is_batch_model("openai/gpt-5.6-luna")

    def test_batch_model_id_implies_batch_mode(self):
        backend = OpenRouterBackend(
            "anthropic/claude-opus-5:batch", api_key="k", client=FakeHTTPClient([])
        )
        assert backend.uses_batch_api is True

    def test_plain_model_does_not_imply_batch_mode(self):
        backend = OpenRouterBackend(
            "anthropic/claude-opus-5", api_key="k", client=FakeHTTPClient([])
        )
        assert backend.uses_batch_api is False

    def test_batch_body_orders_endpoint_and_model_before_requests(self):
        backend = OpenRouterBackend(
            "anthropic/claude-opus-5:batch", api_key="k", client=FakeHTTPClient([])
        )
        body = backend._build_batch_body([REQUEST, REQUEST], PARAMS)
        # OpenRouter stream-parses the body and 400s if requests comes first.
        assert list(body.keys()) == ["endpoint", "model", "requests"]
        assert body["endpoint"] == "/v1/chat/completions"
        assert body["model"] == "anthropic/claude-opus-5:batch"
        assert [r["custom_id"] for r in body["requests"]] == ["req-0", "req-1"]
        # Per-request bodies inherit the batch-level model.
        assert "model" not in body["requests"][0]["body"]
        assert body["requests"][0]["body"]["max_tokens"] == 1024

    def test_batch_flow_submits_polls_and_orders_results(self):
        submitted = {"id": "batch_123", "status": "validating"}
        in_progress = {"id": "batch_123", "status": "in_progress"}
        completed = {
            "id": "batch_123",
            "status": "completed",
            "results": [
                {
                    "custom_id": "req-1",
                    "response": {
                        "status_code": 200,
                        "body": OPENROUTER_REASONING_RESPONSE,
                    },
                    "error": None,
                },
                {
                    "custom_id": "req-0",
                    "response": {"status_code": 200, "body": OPENROUTER_RESPONSE},
                    "error": None,
                },
            ],
        }
        client = FakeHTTPClient(
            [
                FakeResponse(submitted),
                FakeResponse(in_progress),
                FakeResponse(completed),
            ]
        )
        backend = OpenRouterBackend(
            "anthropic/claude-opus-5:batch",
            api_key="k",
            client=client,
            poll_seconds=0.0,
        )
        results = backend.generate_batch([REQUEST, REQUEST], PARAMS)

        assert client.posts[0]["url"] == "https://openrouter.ai/api/beta/batches"
        assert client.gets[0]["url"] == (
            "https://openrouter.ai/api/beta/batches/batch_123"
        )
        # Results come back out of order and must be realigned by custom_id.
        assert [r.text for r in results] == ["Hi.", "391"]
        assert [r.provider_served for r in results] == ["DeepInfra", "Ambient"]

    def test_batch_failure_status_raises(self):
        client = FakeHTTPClient(
            [FakeResponse({"id": "b1", "status": "failed", "error": "nope"})]
        )
        backend = OpenRouterBackend(
            "anthropic/claude-opus-5:batch",
            api_key="k",
            client=client,
            poll_seconds=0.0,
        )
        with pytest.raises(RuntimeError, match="status=failed"):
            backend.generate_batch([REQUEST], PARAMS)

    def test_batch_item_error_raises(self):
        client = FakeHTTPClient(
            [
                FakeResponse(
                    {
                        "id": "b1",
                        "status": "completed",
                        "results": [
                            {
                                "custom_id": "req-0",
                                "response": None,
                                "error": {"message": "boom"},
                            }
                        ],
                    }
                )
            ]
        )
        backend = OpenRouterBackend(
            "anthropic/claude-opus-5:batch",
            api_key="k",
            client=client,
            poll_seconds=0.0,
        )
        with pytest.raises(RuntimeError, match="req-0 failed"):
            backend.generate_batch([REQUEST], PARAMS)

    def test_batch_missing_result_raises(self):
        client = FakeHTTPClient(
            [FakeResponse({"id": "b1", "status": "completed", "results": []})]
        )
        backend = OpenRouterBackend(
            "anthropic/claude-opus-5:batch",
            api_key="k",
            client=client,
            poll_seconds=0.0,
        )
        with pytest.raises(ValueError, match="missing result for req-0"):
            backend.generate_batch([REQUEST], PARAMS)

    def test_batch_submission_without_id_raises(self):
        client = FakeHTTPClient([FakeResponse({"status": "validating"})])
        backend = OpenRouterBackend(
            "anthropic/claude-opus-5:batch",
            api_key="k",
            client=client,
            poll_seconds=0.0,
        )
        with pytest.raises(RuntimeError, match="no id"):
            backend.generate_batch([REQUEST], PARAMS)

    def test_empty_batch_returns_empty(self):
        backend = OpenRouterBackend(
            "anthropic/claude-opus-5:batch", api_key="k", client=FakeHTTPClient([])
        )
        assert backend.generate_batch([], PARAMS) == []

    def test_batch_timeout(self):
        client = FakeHTTPClient(
            [
                FakeResponse({"id": "b1", "status": "validating"}),
                FakeResponse({"id": "b1", "status": "in_progress"}),
            ]
        )
        backend = OpenRouterBackend(
            "anthropic/claude-opus-5:batch",
            api_key="k",
            client=client,
            poll_seconds=0.0,
            batch_timeout_seconds=-1.0,
        )
        with pytest.raises(TimeoutError):
            backend.generate_batch([REQUEST], PARAMS)

    def test_custom_batch_base_url_is_derived_from_base_url(self):
        backend = OpenRouterBackend(
            "m",
            api_key="k",
            client=FakeHTTPClient([]),
            base_url="https://proxy.internal/api/v1",
        )
        assert backend._batch_base_url == "https://proxy.internal/api/beta"


# --------------------------------------------------------------------------- #
# Subclass contract
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestSubclassContract:
    def test_openrouter_and_xai_share_the_chat_compatible_base(self):
        assert issubclass(OpenRouterBackend, OpenAIChatCompatibleBackend)
        assert issubclass(XAIBackend, OpenAIChatCompatibleBackend)

    def test_each_subclass_declares_its_key_env_var(self):
        assert OpenRouterBackend.api_key_env_vars == ("OPENROUTER_API_KEY",)
        assert XAIBackend.api_key_env_vars == ("XAI_API_KEY",)
