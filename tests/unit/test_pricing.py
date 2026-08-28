"""Unit tests for shelf.llm.pricing.

No network: every test builds a :class:`PricingTable` from an in-memory payload
shaped exactly like a real ``https://openrouter.ai/api/v1/models`` response
(shape confirmed against the live endpoint on 2026-08-26).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from shelf.llm.pricing import (
    DEFAULT_TTL_SECONDS,
    VERIFIED_NATIVE_PRICES,
    ModelPrice,
    PriceOverride,
    PricingFetchError,
    PricingTable,
    PricingUnavailableError,
    UnknownModelError,
    batch_variant,
    load_overrides,
    load_snapshot,
    model_candidates,
    split_batch_suffix,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _entry(
    model_id: str,
    prompt: str,
    completion: str,
    *,
    context_length: int = 200_000,
    extra_pricing: dict | None = None,
) -> dict:
    """Build one catalogue entry in the real OpenRouter shape."""
    pricing = {"prompt": prompt, "completion": completion}
    if extra_pricing:
        pricing.update(extra_pricing)
    return {
        "id": model_id,
        "canonical_slug": model_id.replace(":", "-"),
        "name": model_id,
        "created": 1787773060,
        "description": "fixture",
        "context_length": context_length,
        "architecture": {"modality": "text->text"},
        "pricing": pricing,
        "top_provider": {"context_length": context_length},
        "supported_parameters": ["temperature", "top_p"],
    }


@pytest.fixture
def payload() -> dict:
    """A miniature but structurally faithful OpenRouter payload."""
    return {
        "data": [
            _entry(
                "anthropic/claude-opus-5",
                "0.000005",
                "0.000025",
                extra_pricing={
                    "input_cache_read": "0.0000005",
                    "input_cache_write": "0.00000625",
                },
            ),
            _entry("anthropic/claude-opus-5:batch", "0.0000025", "0.0000125"),
            _entry("anthropic/claude-sonnet-4.5", "0.000003", "0.000015"),
            _entry("openai/gpt-5.2", "0.00000175", "0.000014"),
            _entry(
                "openai/gpt-5.6-luna",
                "0.0000002",
                "0.0000012",
                context_length=400_000,
                extra_pricing={
                    "overrides": [
                        {
                            "min_prompt_tokens": 272000,
                            "prompt": "0.0000004",
                            "completion": "0.0000018",
                        }
                    ]
                },
            ),
            _entry(
                "google/gemini-3.1-pro-preview",
                "0.000002",
                "0.000012",
                extra_pricing={"internal_reasoning": "0.000012"},
            ),
            _entry("x-ai/grok-4.6", "0.000002", "0.000006"),
            _entry("meta-llama/llama-4-maverick", "0.00000015", "0.0000006"),
        ],
        "total_count": 8,
    }


@pytest.fixture
def table(payload: dict) -> PricingTable:
    """Table built from the fixture payload."""
    return PricingTable.from_payload(payload, fetched_at="2026-08-26T12:00:00+00:00")


# --------------------------------------------------------------------------- #
# Id helpers
# --------------------------------------------------------------------------- #


class TestIdHelpers:
    """Tests for :func:`split_batch_suffix`, :func:`batch_variant`."""

    def test_split_batch_suffix(self):
        assert split_batch_suffix("anthropic/claude-opus-5:batch") == (
            "anthropic/claude-opus-5",
            True,
        )

    def test_split_no_suffix(self):
        assert split_batch_suffix("openai/gpt-5.2") == ("openai/gpt-5.2", False)

    def test_split_ignores_other_suffixes(self):
        assert split_batch_suffix("meta/model:free") == ("meta/model:free", False)

    def test_batch_variant_is_idempotent(self):
        once = batch_variant("openai/gpt-5.2")
        assert once == "openai/gpt-5.2:batch"
        assert batch_variant(once) == once

    def test_candidates_include_native_prefixes(self):
        candidates = model_candidates("gpt-5.2")
        assert "openai/gpt-5.2" in candidates
        assert "gpt-5.2" in candidates

    def test_provider_hint_ranks_first(self):
        candidates = model_candidates("claude-opus-5", provider="anthropic")
        prefixed = [c for c in candidates if "/" in c]
        assert prefixed[0] == "anthropic/claude-opus-5"

    def test_candidates_strip_date_suffix(self):
        assert "anthropic/claude-sonnet-4.5" in model_candidates(
            "claude-sonnet-4-5-20250929"
        )

    def test_candidates_dot_dashed_versions_mid_name(self):
        """``claude-3-5-haiku`` and ``claude-sonnet-4-5`` both need dotting."""
        assert "anthropic/claude-3.5-haiku" in model_candidates(
            "claude-3-5-haiku-20241022", provider="anthropic"
        )
        assert "anthropic/claude-sonnet-4.5" in model_candidates(
            "claude-sonnet-4-5", provider="anthropic"
        )

    def test_candidates_leave_non_version_dashes_alone(self):
        assert "openai/gpt-4-turbo" in model_candidates("gpt-4-turbo")

    def test_candidates_handle_preview_suffix(self):
        assert "google/gemini-3.1-pro-preview" in model_candidates(
            "gemini-3.1-pro", provider="google"
        )

    def test_candidates_are_deduplicated(self):
        candidates = model_candidates("gpt-5.2")
        assert len(candidates) == len(set(candidates))

    def test_empty_model_has_no_candidates(self):
        assert model_candidates("  ") == []


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #


class TestPriceOf:
    """Tests for model resolution."""

    def test_exact_openrouter_id(self, table: PricingTable):
        price = table.price_of("anthropic/claude-opus-5")
        assert price.input_per_1m == pytest.approx(5.0)
        assert price.output_per_1m == pytest.approx(25.0)
        assert price.source == "openrouter"
        assert price.is_estimated is False

    def test_published_batch_id_used_directly(self, table: PricingTable):
        price = table.price_of("anthropic/claude-opus-5:batch")
        assert price.resolved_id == "anthropic/claude-opus-5:batch"
        assert price.is_batch is True
        assert price.input_per_1m == pytest.approx(2.5)
        assert price.is_estimated is False

    def test_unpublished_batch_id_falls_back_to_discount(self, table: PricingTable):
        """gpt-5.2 has no ':batch' entry, so the discount is applied and flagged."""
        price = table.price_of("openai/gpt-5.2:batch")
        assert price.resolved_id == "openai/gpt-5.2"
        assert price.is_batch is True
        assert price.input_per_1m == pytest.approx(0.875)
        assert price.output_per_1m == pytest.approx(7.0)
        assert price.source == "openrouter+batch_discount"
        assert price.is_estimated is True

    def test_native_id_without_prefix(self, table: PricingTable):
        price = table.price_of("gpt-5.2")
        assert price.resolved_id == "openai/gpt-5.2"

    def test_native_anthropic_id_with_date(self, table: PricingTable):
        price = table.price_of("claude-sonnet-4-5-20250929", provider="anthropic")
        assert price.resolved_id == "anthropic/claude-sonnet-4.5"
        assert price.input_per_1m == pytest.approx(3.0)

    def test_native_google_id_gets_preview_suffix(self, table: PricingTable):
        price = table.price_of("gemini-3.1-pro", provider="google")
        assert price.resolved_id == "google/gemini-3.1-pro-preview"

    def test_native_xai_id(self, table: PricingTable):
        price = table.price_of("grok-4.6", provider="xai")
        assert price.resolved_id == "x-ai/grok-4.6"

    def test_unknown_model_raises(self, table: PricingTable):
        with pytest.raises(UnknownModelError) as excinfo:
            table.price_of("acme/does-not-exist")
        assert excinfo.value.model == "acme/does-not-exist"

    def test_get_returns_none_for_unknown(self, table: PricingTable):
        assert table.get("acme/does-not-exist") is None

    def test_contains(self, table: PricingTable):
        assert "openai/gpt-5.2" in table
        assert "acme/nope" not in table

    def test_result_is_cached(self, table: PricingTable):
        first = table.price_of("openai/gpt-5.2")
        second = table.price_of("openai/gpt-5.2")
        assert first is second

    def test_context_length_captured(self, table: PricingTable):
        assert table.price_of("openai/gpt-5.6-luna").context_length == 400_000

    def test_reasoning_rate_captured(self, table: PricingTable):
        price = table.price_of("google/gemini-3.1-pro-preview")
        assert price.reasoning_per_token == pytest.approx(0.000012)

    def test_cache_rates_captured(self, table: PricingTable):
        price = table.price_of("anthropic/claude-opus-5")
        assert price.cache_read_per_token == pytest.approx(0.0000005)
        assert price.cache_write_per_token == pytest.approx(0.00000625)


class TestOverrides:
    """Manual overrides take precedence over the catalogue."""

    def test_override_wins(self, payload: dict):
        table = PricingTable.from_payload(
            payload,
            overrides={"openai/gpt-5.2": PriceOverride(99.0, 199.0, note="manual")},
        )
        price = table.price_of("openai/gpt-5.2")
        assert price.input_per_1m == pytest.approx(99.0)
        assert price.source == "override"

    def test_override_reachable_via_native_id(self, payload: dict):
        table = PricingTable.from_payload(
            payload,
            overrides={"openai/gpt-5.2": PriceOverride(99.0, 199.0)},
        )
        assert table.price_of("gpt-5.2").source == "override"

    def test_batch_override(self, payload: dict):
        table = PricingTable.from_payload(
            payload,
            overrides={"openai/gpt-5.2:batch": PriceOverride(0.5, 4.0)},
        )
        price = table.price_of("openai/gpt-5.2:batch")
        assert price.is_batch is True
        assert price.input_per_1m == pytest.approx(0.5)

    def test_load_overrides_from_json(self, tmp_path: Path):
        path = tmp_path / "overrides.json"
        path.write_text(
            json.dumps(
                {
                    "openai/gpt-5.6-sol": {
                        "input_per_1m": 4.0,
                        "output_per_1m": 20.0,
                        "note": "openai page",
                    }
                }
            )
        )
        overrides = load_overrides(path)
        assert overrides["openai/gpt-5.6-sol"].output_per_1m == pytest.approx(20.0)
        assert overrides["openai/gpt-5.6-sol"].note == "openai page"

    def test_load_overrides_rejects_bad_shape(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"m": {"input_per_1m": 1.0}}))
        with pytest.raises(ValueError):
            load_overrides(path)


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #


class TestEstimateCost:
    """Tests for cost arithmetic."""

    def test_simple_cost(self, table: PricingTable):
        # 1000 in @ $5/1M + 2000 out @ $25/1M
        cost = table.estimate_cost("anthropic/claude-opus-5", 1000, 2000)
        assert cost == pytest.approx(0.005 + 0.05)

    def test_batch_is_half(self, table: PricingTable):
        standard = table.estimate_cost("anthropic/claude-opus-5", 1000, 2000)
        batched = table.estimate_cost("anthropic/claude-opus-5:batch", 1000, 2000)
        assert batched == pytest.approx(standard / 2)

    def test_zero_tokens_is_free(self, table: PricingTable):
        assert table.estimate_cost("openai/gpt-5.2", 0, 0) == 0.0

    def test_reasoning_defaults_to_excluded(self, table: PricingTable):
        with_default = table.estimate_cost("openai/gpt-5.2", 100, 100)
        assert with_default == pytest.approx(100 * 1.75e-6 + 100 * 1.4e-5, rel=1e-9)

    def test_reasoning_billed_at_reasoning_rate(self, table: PricingTable):
        price = table.price_of("google/gemini-3.1-pro-preview")
        assert price.cost(0, 0, 1000) == pytest.approx(1000 * 0.000012)

    def test_reasoning_falls_back_to_output_rate(self, table: PricingTable):
        price = table.price_of("openai/gpt-5.2")
        assert price.reasoning_per_token is None
        assert price.cost(0, 0, 1000) == pytest.approx(1000 * 1.4e-5)

    def test_long_context_tier_applies(self, table: PricingTable):
        """gpt-5.6-luna doubles above 272k prompt tokens (confirmed live)."""
        price = table.price_of("openai/gpt-5.6-luna")
        short = price.cost(1000, 1000)
        long = price.cost(300_000, 1000)
        assert price.rates_for(1000) == pytest.approx((2e-7, 1.2e-6))
        assert price.rates_for(300_000) == pytest.approx((4e-7, 1.8e-6))
        assert long > short

    def test_negative_tokens_rejected(self, table: PricingTable):
        with pytest.raises(ValueError):
            table.estimate_cost("openai/gpt-5.2", -1, 0)

    def test_phase1_style_budget_projection(self, table: PricingTable):
        """1,500 docs at ~600 in / ~2,400 out on batched Opus stays under $50."""
        per_doc = table.estimate_cost("anthropic/claude-opus-5:batch", 600, 2400)
        assert per_doc * 1500 < 50.0

    def test_invalid_batch_discount_rejected(self, payload: dict):
        with pytest.raises(ValueError):
            PricingTable.from_payload(payload, batch_discount=0.0)


# --------------------------------------------------------------------------- #
# Snapshot / cache
# --------------------------------------------------------------------------- #


class TestSnapshotAndCache:
    """Cache, TTL, and offline behaviour. No network is ever touched."""

    def _write_cache(self, path: Path, payload: dict, fetched_at: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "source_url": "https://openrouter.ai/api/v1/models",
                    "fetched_at": fetched_at,
                    "data": payload["data"],
                }
            )
        )

    def test_fresh_cache_is_used_without_network(self, tmp_path: Path, payload: dict):
        from datetime import UTC, datetime

        cache = tmp_path / "models.json"
        self._write_cache(cache, payload, datetime.now(UTC).isoformat())
        snapshot = load_snapshot(cache_path=cache, allow_network=False)
        assert snapshot.from_cache is True
        assert snapshot.is_stale is False
        assert snapshot.model_count == len(payload["data"])

    def test_stale_cache_still_usable_offline(self, tmp_path: Path, payload: dict):
        cache = tmp_path / "models.json"
        self._write_cache(cache, payload, "2020-01-01T00:00:00+00:00")
        snapshot = load_snapshot(cache_path=cache, allow_network=False)
        assert snapshot.is_stale is True
        assert snapshot.model_count == len(payload["data"])

    def test_no_cache_offline_raises(self, tmp_path: Path):
        with pytest.raises(PricingUnavailableError):
            load_snapshot(cache_path=tmp_path / "missing.json", allow_network=False)

    def test_network_failure_without_cache_raises(self, tmp_path: Path, monkeypatch):
        def boom(*_args, **_kwargs):
            raise PricingFetchError("no route to host")

        monkeypatch.setattr("shelf.llm.pricing.fetch_openrouter_models", boom)
        with pytest.raises(PricingUnavailableError):
            load_snapshot(cache_path=tmp_path / "missing.json", allow_network=True)

    def test_network_failure_falls_back_to_stale_cache(
        self, tmp_path: Path, payload: dict, monkeypatch
    ):
        cache = tmp_path / "models.json"
        self._write_cache(cache, payload, "2020-01-01T00:00:00+00:00")

        def boom(*_args, **_kwargs):
            raise PricingFetchError("no route to host")

        monkeypatch.setattr("shelf.llm.pricing.fetch_openrouter_models", boom)
        snapshot = load_snapshot(cache_path=cache, allow_network=True)
        assert snapshot.from_cache is True
        assert snapshot.model_count == len(payload["data"])

    def test_stale_cache_triggers_refetch_and_rewrite(
        self, tmp_path: Path, payload: dict, monkeypatch
    ):
        cache = tmp_path / "models.json"
        self._write_cache(cache, payload, "2020-01-01T00:00:00+00:00")
        monkeypatch.setattr(
            "shelf.llm.pricing.fetch_openrouter_models",
            lambda *_a, **_k: payload["data"][:2],
        )
        snapshot = load_snapshot(cache_path=cache, allow_network=True)
        assert snapshot.from_cache is False
        assert snapshot.model_count == 2
        # Cache was rewritten with the fresh data.
        assert len(json.loads(cache.read_text())["data"]) == 2

    def test_corrupt_cache_is_ignored(self, tmp_path: Path):
        cache = tmp_path / "models.json"
        cache.write_text("{not json")
        with pytest.raises(PricingUnavailableError):
            load_snapshot(cache_path=cache, allow_network=False)

    def test_wrong_schema_cache_is_ignored(self, tmp_path: Path, payload: dict):
        cache = tmp_path / "models.json"
        cache.write_text(json.dumps({"schema": 99, "data": payload["data"]}))
        with pytest.raises(PricingUnavailableError):
            load_snapshot(cache_path=cache, allow_network=False)

    def test_force_refresh_bypasses_fresh_cache(
        self, tmp_path: Path, payload: dict, monkeypatch
    ):
        from datetime import UTC, datetime

        cache = tmp_path / "models.json"
        self._write_cache(cache, payload, datetime.now(UTC).isoformat())
        called: list[bool] = []

        def fake(*_a, **_k):
            called.append(True)
            return payload["data"]

        monkeypatch.setattr("shelf.llm.pricing.fetch_openrouter_models", fake)
        load_snapshot(cache_path=cache, force_refresh=True)
        assert called == [True]

    def test_manifest_records_fetch_timestamp(self, table: PricingTable):
        manifest = table.manifest_entry(["anthropic/claude-opus-5"])
        assert manifest["fetched_at"] == "2026-08-26T12:00:00+00:00"
        assert manifest["source_url"].startswith("https://openrouter.ai")
        assert manifest["model_count"] == 8
        assert manifest["ttl_seconds"] == DEFAULT_TTL_SECONDS
        assert manifest["models"]["anthropic/claude-opus-5"]["input_per_1m"] == 5.0

    def test_manifest_is_json_serializable(self, table: PricingTable):
        json.dumps(table.manifest_entry(["openai/gpt-5.2"]))


class TestPayloadValidation:
    """Malformed catalogue payloads are rejected, not silently priced at zero."""

    def test_missing_data_key(self):
        with pytest.raises(PricingFetchError):
            PricingTable.from_payload({"models": []})

    def test_entry_without_pricing_is_unpriceable(self):
        table = PricingTable.from_payload({"data": [{"id": "acme/x"}]})
        assert table.get("acme/x") is None

    def test_iter_prices_skips_unpriceable(self, payload: dict):
        payload["data"].append({"id": "acme/x"})
        table = PricingTable.from_payload(payload)
        priced = list(table.iter_prices())
        assert all(isinstance(p, ModelPrice) for p in priced)
        assert "acme/x" not in {p.resolved_id for p in priced}

    def test_search(self, table: PricingTable):
        assert table.search("opus") == [
            "anthropic/claude-opus-5",
            "anthropic/claude-opus-5:batch",
        ]


# --------------------------------------------------------------------------- #
# Native cross-check
# --------------------------------------------------------------------------- #


class TestCrosscheck:
    """The catalogue vs. published provider prices."""

    def test_no_discrepancy_when_prices_agree(self, table: PricingTable):
        reference = {
            "anthropic/claude-opus-5": PriceOverride(5.0, 25.0, note="anthropic"),
            "openai/gpt-5.2": PriceOverride(1.75, 14.0, note="openai"),
        }
        assert table.crosscheck(reference) == []

    def test_discrepancy_reports_both_numbers(self, table: PricingTable):
        reference = {"openai/gpt-5.2": PriceOverride(3.5, 28.0, note="openai page")}
        (found,) = table.crosscheck(reference)
        assert found.catalogue_input_per_1m == pytest.approx(1.75)
        assert found.reference_input_per_1m == pytest.approx(3.5)
        assert found.input_ratio == pytest.approx(0.5)
        assert found.output_ratio == pytest.approx(0.5)
        assert found.note == "openai page"

    def test_models_absent_from_catalogue_are_skipped(self, table: PricingTable):
        reference = {"acme/nope": PriceOverride(1.0, 2.0)}
        assert table.crosscheck(reference) == []

    def test_default_reference_is_the_verified_map(self, table: PricingTable):
        # The fixture agrees with the verified map for claude-opus-5, and the
        # remaining verified models are simply absent from the fixture.
        assert table.crosscheck() == []

    def test_verified_map_covers_phase1_roster_families(self):
        for model in (
            "anthropic/claude-opus-5:batch",
            "openai/gpt-5.6-sol:batch",
            "google/gemini-3.1-pro-preview:batch",
            "x-ai/grok-4.6",
        ):
            assert model in VERIFIED_NATIVE_PRICES

    def test_discrepancy_dict_is_serializable(self, table: PricingTable):
        reference = {"openai/gpt-5.2": PriceOverride(3.5, 28.0)}
        (found,) = table.crosscheck(reference)
        json.dumps(found.to_dict())
