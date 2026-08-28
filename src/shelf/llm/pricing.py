"""Live model pricing for SHELF generation runs.

The hardcoded price table that used to live in ``scripts/generate_documents.py``
goes stale the moment a provider ships a new model. This module replaces it with
a fetch from the public OpenRouter model catalogue
(``https://openrouter.ai/api/v1/models``, no auth required), cached to disk with
a TTL so a run is reproducible, offline-capable, and recordable in a manifest.

Design notes
------------
* **Single source of truth.** OpenRouter publishes per-token USD prices for the
  OpenAI / Anthropic / Google / xAI / open-weight families, so it doubles as the
  price source for models we call *natively* rather than through OpenRouter.
  :func:`price_of` accepts native ids (``claude-opus-4-5-20250929``,
  ``gemini-3.1-pro-preview``, ``gpt-5.2``, ``grok-4.6``) and resolves them onto
  OpenRouter slugs. A manual override map wins over the fetched table.
* **Batch variants.** OpenRouter lists 50%-discount batch tiers as separate ids
  with a ``:batch`` suffix (e.g. ``anthropic/claude-opus-5:batch``). Those are
  looked up directly. If a ``:batch`` id is *not* listed, the base price is
  multiplied by :data:`DEFAULT_BATCH_DISCOUNT` and the resulting
  :class:`ModelPrice` is flagged ``source="openrouter+batch_discount"`` so an
  estimate is never mistaken for a published number.
* **Fail loud.** If the network is unavailable and there is no cache at all,
  :class:`PricingUnavailableError` is raised. We never silently fall back to
  stale hardcoded numbers.

Example
-------
    >>> table = PricingTable.load()                        # doctest: +SKIP
    >>> table.estimate_cost("anthropic/claude-opus-5:batch", 1200, 2400)
    0.0345
    >>> table.manifest_entry()                             # doctest: +SKIP
    {'source_url': ..., 'fetched_at': ..., 'model_count': 417, ...}
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

#: Where the fetched catalogue is cached. Override per run if you need to pin.
DEFAULT_CACHE_PATH = Path("data/cache/openrouter_models.json")

#: 24 hours.
DEFAULT_TTL_SECONDS = 24 * 60 * 60

DEFAULT_TIMEOUT_SECONDS = 30.0

#: Fallback multiplier when a ``:batch`` id is requested but not published.
DEFAULT_BATCH_DISCOUNT = 0.5

CACHE_SCHEMA_VERSION = 1

BATCH_SUFFIX = "batch"

#: Native provider name -> OpenRouter author prefix. Used to resolve a model id
#: that was given without the ``author/`` component (i.e. called natively).
NATIVE_PROVIDER_PREFIXES: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "gemini": "google",
    "vertex": "google",
    "xai": "x-ai",
    "x-ai": "x-ai",
    "grok": "x-ai",
    "openrouter": "",
}

#: Prefix search order when the caller gives no provider hint. Ordered by how
#: distinctive the bare model names are, so ``gpt-*`` hits ``openai`` first.
_PREFIX_SEARCH_ORDER: tuple[str, ...] = ("openai", "anthropic", "google", "x-ai")

_DATE_SUFFIX_RE = re.compile(r"-(?:\d{8}|\d{4}-\d{2}-\d{2}|latest)$")
#: ``claude-sonnet-4-5`` -> ``claude-sonnet-4.5``, ``claude-3-5-haiku`` ->
#: ``claude-3.5-haiku``. Native ids spell version numbers with dashes; the
#: OpenRouter catalogue uses dots.
_DASHED_VERSION_RE = re.compile(r"-(\d+)-(\d+)(?=-|$)")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class PricingError(RuntimeError):
    """Base class for pricing failures."""


class PricingFetchError(PricingError):
    """The OpenRouter catalogue could not be fetched or parsed."""


class PricingUnavailableError(PricingError):
    """No live catalogue and no cache — refuse to guess."""


class UnknownModelError(PricingError):
    """A model id could not be resolved against the catalogue or overrides."""

    def __init__(self, model: str, tried: Iterable[str] = ()) -> None:
        self.model = model
        self.tried = tuple(tried)
        detail = f" (tried: {', '.join(self.tried)})" if self.tried else ""
        super().__init__(f"No pricing found for model {model!r}{detail}")


# --------------------------------------------------------------------------- #
# Value types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PriceTier:
    """A long-context pricing tier.

    OpenRouter expresses these as ``pricing.overrides`` entries keyed by
    ``min_prompt_tokens``; the tier with the largest threshold at or below the
    actual prompt size wins.
    """

    min_prompt_tokens: int
    input_per_token: float
    output_per_token: float


@dataclass(frozen=True, slots=True)
class PriceOverride:
    """Manual price entry, expressed in the units humans read on pricing pages."""

    input_per_1m: float
    output_per_1m: float
    context_length: int | None = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """Resolved prices for one model, in USD per token."""

    model: str
    """The id the caller asked for."""

    resolved_id: str
    """The catalogue id (or override key) the price came from."""

    input_per_token: float
    output_per_token: float
    is_batch: bool = False
    source: str = "openrouter"
    fetched_at: str = ""
    context_length: int | None = None
    reasoning_per_token: float | None = None
    cache_read_per_token: float | None = None
    cache_write_per_token: float | None = None
    tiers: tuple[PriceTier, ...] = ()

    @property
    def input_per_1m(self) -> float:
        """Input price in USD per 1M tokens."""
        return self.input_per_token * 1_000_000

    @property
    def output_per_1m(self) -> float:
        """Output price in USD per 1M tokens."""
        return self.output_per_token * 1_000_000

    @property
    def is_estimated(self) -> bool:
        """True when the price was derived rather than published verbatim."""
        return "discount" in self.source

    def rates_for(self, input_tokens: int) -> tuple[float, float]:
        """Return ``(input_per_token, output_per_token)`` for a prompt size."""
        rate_in, rate_out = self.input_per_token, self.output_per_token
        best = -1
        for tier in self.tiers:
            if input_tokens >= tier.min_prompt_tokens > best:
                best = tier.min_prompt_tokens
                rate_in, rate_out = tier.input_per_token, tier.output_per_token
        return rate_in, rate_out

    def cost(
        self,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
    ) -> float:
        """Cost in USD for one request.

        Args:
            input_tokens: Billed prompt tokens.
            output_tokens: Billed completion tokens.
            reasoning_tokens: Thinking tokens **only if the provider reports
                them separately from ``output_tokens``**. Most providers already
                fold reasoning into the completion count; passing them again
                would double-bill, so this defaults to 0. Billed at
                ``reasoning_per_token`` when the catalogue lists one, otherwise
                at the completion rate.

        Returns:
            Estimated USD cost, never negative.
        """
        if input_tokens < 0 or output_tokens < 0 or reasoning_tokens < 0:
            raise ValueError("token counts must be non-negative")
        rate_in, rate_out = self.rates_for(input_tokens)
        total = input_tokens * rate_in + output_tokens * rate_out
        if reasoning_tokens:
            rate_reasoning = (
                self.reasoning_per_token
                if self.reasoning_per_token is not None
                else rate_out
            )
            total += reasoning_tokens * rate_reasoning
        return total

    def to_dict(self) -> dict[str, Any]:
        """Serialize for a run manifest."""
        return {
            "model": self.model,
            "resolved_id": self.resolved_id,
            "input_per_1m": self.input_per_1m,
            "output_per_1m": self.output_per_1m,
            "is_batch": self.is_batch,
            "is_estimated": self.is_estimated,
            "source": self.source,
            "fetched_at": self.fetched_at,
            "context_length": self.context_length,
        }


@dataclass(frozen=True, slots=True)
class PricingSnapshot:
    """An immutable point-in-time copy of the OpenRouter catalogue."""

    fetched_at: str
    source_url: str
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    from_cache: bool = False
    age_seconds: float = 0.0
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    cache_path: str | None = None

    @property
    def model_count(self) -> int:
        """Number of catalogue entries."""
        return len(self.entries)

    @property
    def is_stale(self) -> bool:
        """True when the snapshot is older than its TTL."""
        return self.age_seconds > self.ttl_seconds

    def to_manifest(self) -> dict[str, Any]:
        """Provenance block for a run manifest."""
        return {
            "source_url": self.source_url,
            "fetched_at": self.fetched_at,
            "model_count": self.model_count,
            "from_cache": self.from_cache,
            "age_seconds": round(self.age_seconds, 3),
            "ttl_seconds": self.ttl_seconds,
            "is_stale": self.is_stale,
            "cache_path": self.cache_path,
        }


# --------------------------------------------------------------------------- #
# Fetch / cache
# --------------------------------------------------------------------------- #


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _parse_iso(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def fetch_openrouter_models(
    url: str = OPENROUTER_MODELS_URL,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Fetch the raw OpenRouter model catalogue.

    Args:
        url: Catalogue endpoint. No API key is required.
        timeout: Per-request timeout in seconds.

    Returns:
        The raw ``data`` list, one dict per model.

    Raises:
        PricingFetchError: On any transport, status, or shape problem.
    """
    import httpx

    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed error
        raise PricingFetchError(f"Failed to fetch pricing from {url}: {exc}") from exc

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data:
        raise PricingFetchError(f"Unexpected payload shape from {url}: missing 'data'")
    return [entry for entry in data if isinstance(entry, dict) and entry.get("id")]


def _write_cache(
    path: Path, source_url: str, fetched_at: str, data: list[dict]
) -> None:
    """Atomically write the catalogue cache (tmp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "source_url": source_url,
        "fetched_at": fetched_at,
        "data": data,
    }
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(tmp, path)


def _read_cache(path: Path) -> tuple[str, str, list[dict]] | None:
    """Return ``(fetched_at, source_url, data)`` or None if unusable."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable pricing cache %s: %s", path, exc)
        return None
    if not isinstance(payload, dict) or payload.get("schema") != CACHE_SCHEMA_VERSION:
        logger.warning("Ignoring pricing cache %s: unsupported schema", path)
        return None
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    fetched_at = str(payload.get("fetched_at") or "")
    source_url = str(payload.get("source_url") or OPENROUTER_MODELS_URL)
    return fetched_at, source_url, [e for e in data if isinstance(e, dict)]


def _snapshot_from(
    fetched_at: str,
    source_url: str,
    data: list[dict],
    *,
    from_cache: bool,
    ttl_seconds: int,
    cache_path: Path | None,
) -> PricingSnapshot:
    parsed = _parse_iso(fetched_at)
    age = (
        max(0.0, (datetime.now(UTC) - parsed).total_seconds())
        if parsed is not None
        else float("inf")
    )
    return PricingSnapshot(
        fetched_at=fetched_at,
        source_url=source_url,
        entries={str(entry["id"]): entry for entry in data if entry.get("id")},
        from_cache=from_cache,
        age_seconds=age,
        ttl_seconds=ttl_seconds,
        cache_path=str(cache_path) if cache_path is not None else None,
    )


def load_snapshot(
    *,
    cache_path: Path | str | None = DEFAULT_CACHE_PATH,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    force_refresh: bool = False,
    allow_network: bool = True,
    url: str = OPENROUTER_MODELS_URL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> PricingSnapshot:
    """Load the catalogue from cache, refreshing over the network when stale.

    Args:
        cache_path: Cache file. ``None`` disables caching entirely (always fetch).
        ttl_seconds: Cache lifetime. A cache younger than this is used as-is.
        force_refresh: Ignore a fresh cache and re-fetch.
        allow_network: If False, never touch the network (offline / CI mode).
        url: Catalogue endpoint.
        timeout: Per-request timeout.

    Returns:
        A :class:`PricingSnapshot`.

    Raises:
        PricingUnavailableError: No usable cache and no way to fetch one.
    """
    path = Path(cache_path) if cache_path is not None else None
    cached = _read_cache(path) if path is not None else None

    if cached is not None and not force_refresh:
        snapshot = _snapshot_from(
            *cached,
            from_cache=True,
            ttl_seconds=ttl_seconds,
            cache_path=path,
        )
        if not snapshot.is_stale:
            return snapshot

    if allow_network:
        try:
            data = fetch_openrouter_models(url, timeout=timeout)
        except PricingFetchError as exc:
            if cached is None:
                raise PricingUnavailableError(
                    f"Cannot fetch pricing from {url} and no cache is available"
                    f" at {path}. Refusing to run with unknown prices. ({exc})"
                ) from exc
            logger.warning(
                "Pricing fetch failed (%s); falling back to stale cache at %s",
                exc,
                path,
            )
            return _snapshot_from(
                *cached, from_cache=True, ttl_seconds=ttl_seconds, cache_path=path
            )
        fetched_at = _utc_now_iso()
        if path is not None:
            try:
                _write_cache(path, url, fetched_at, data)
            except OSError as exc:
                logger.warning("Could not write pricing cache %s: %s", path, exc)
        return _snapshot_from(
            fetched_at,
            url,
            data,
            from_cache=False,
            ttl_seconds=ttl_seconds,
            cache_path=path,
        )

    if cached is not None:
        return _snapshot_from(
            *cached, from_cache=True, ttl_seconds=ttl_seconds, cache_path=path
        )

    raise PricingUnavailableError(
        f"Network access is disabled and no pricing cache exists at {path}."
        " Refusing to run with unknown prices."
    )


# --------------------------------------------------------------------------- #
# Model id resolution
# --------------------------------------------------------------------------- #


def split_batch_suffix(model: str) -> tuple[str, bool]:
    """Split ``"anthropic/claude-opus-5:batch"`` into ``("anthropic/claude-opus-5", True)``."""
    base, sep, suffix = model.rpartition(":")
    if sep and suffix == BATCH_SUFFIX:
        return base, True
    return model, False


def batch_variant(model: str) -> str:
    """Return the ``:batch`` form of a model id (idempotent)."""
    base, _ = split_batch_suffix(model)
    return f"{base}:{BATCH_SUFFIX}"


def _name_variants(name: str) -> list[str]:
    """Plausible spellings of a bare model name, most specific first.

    Handles the two mismatches between native ids and OpenRouter slugs:
    trailing snapshot dates (``claude-opus-4-5-20250929``) and dashed version
    numbers (``claude-opus-4-5`` vs ``claude-opus-4.5``), plus the optional
    ``-preview`` suffix Google uses.
    """
    seen: list[str] = []

    def add(candidate: str) -> None:
        if candidate and candidate not in seen:
            seen.append(candidate)

    add(name)
    stripped = _DATE_SUFFIX_RE.sub("", name)
    add(stripped)

    for base in list(seen):
        add(_DASHED_VERSION_RE.sub(r"-\1.\2", base))

    for base in list(seen):
        if base.endswith("-preview"):
            add(base[: -len("-preview")])
        else:
            add(f"{base}-preview")

    return seen


def model_candidates(model: str, provider: str | None = None) -> list[str]:
    """Enumerate catalogue ids to try for a (possibly native) model id.

    Args:
        model: Model id, with or without an ``author/`` prefix and with or
            without a ``:batch`` suffix (the suffix is ignored here).
        provider: Optional native provider hint (``"anthropic"``, ``"openai"``,
            ``"google"``/``"gemini"``, ``"xai"``). Narrows the prefix search.

    Returns:
        Ordered, de-duplicated candidate ids.
    """
    base, _ = split_batch_suffix(model.strip())
    if not base:
        return []

    if "/" in base:
        author, _, name = base.partition("/")
        return [f"{author}/{variant}" for variant in _name_variants(name)]

    if provider:
        hint = NATIVE_PROVIDER_PREFIXES.get(provider.strip().lower())
        prefixes = (
            (hint, *(p for p in _PREFIX_SEARCH_ORDER if p != hint))
            if hint
            else _PREFIX_SEARCH_ORDER
        )
    else:
        prefixes = _PREFIX_SEARCH_ORDER

    candidates: list[str] = []
    for variant in _name_variants(base):
        candidates.append(variant)
        for prefix in prefixes:
            if prefix:
                candidates.append(f"{prefix}/{variant}")
    # Preserve order, drop duplicates.
    return list(dict.fromkeys(candidates))


# --------------------------------------------------------------------------- #
# Pricing table
# --------------------------------------------------------------------------- #


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _close(left: float, right: float, tolerance: float) -> bool:
    """Relative comparison that tolerates float noise and tiny prices."""
    if left == right:
        return True
    scale = max(abs(left), abs(right))
    return abs(left - right) <= tolerance * scale


def _tiers_from(pricing: Mapping[str, Any]) -> tuple[PriceTier, ...]:
    raw = pricing.get("overrides")
    if not isinstance(raw, list):
        return ()
    tiers: list[PriceTier] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        threshold = item.get("min_prompt_tokens")
        rate_in = _as_float(item.get("prompt"))
        rate_out = _as_float(item.get("completion"))
        if threshold is None or rate_in is None or rate_out is None:
            continue
        tiers.append(PriceTier(int(threshold), rate_in, rate_out))
    return tuple(sorted(tiers, key=lambda tier: tier.min_prompt_tokens))


def _price_from_entry(
    entry: Mapping[str, Any],
    *,
    requested: str,
    resolved_id: str,
    fetched_at: str,
    is_batch: bool,
    discount: float = 1.0,
) -> ModelPrice | None:
    pricing = entry.get("pricing")
    if not isinstance(pricing, Mapping):
        return None
    rate_in = _as_float(pricing.get("prompt"))
    rate_out = _as_float(pricing.get("completion"))
    if rate_in is None or rate_out is None:
        return None

    reasoning = _as_float(pricing.get("internal_reasoning"))
    cache_read = _as_float(pricing.get("input_cache_read"))
    cache_write = _as_float(pricing.get("input_cache_write"))
    tiers = _tiers_from(pricing)
    if discount != 1.0:
        tiers = tuple(
            PriceTier(
                tier.min_prompt_tokens,
                tier.input_per_token * discount,
                tier.output_per_token * discount,
            )
            for tier in tiers
        )

    return ModelPrice(
        model=requested,
        resolved_id=resolved_id,
        input_per_token=rate_in * discount,
        output_per_token=rate_out * discount,
        is_batch=is_batch,
        source="openrouter" if discount == 1.0 else "openrouter+batch_discount",
        fetched_at=fetched_at,
        context_length=(
            int(entry["context_length"])
            if isinstance(entry.get("context_length"), (int, float))
            else None
        ),
        reasoning_per_token=None if reasoning is None else reasoning * discount,
        cache_read_per_token=None if cache_read is None else cache_read * discount,
        cache_write_per_token=None if cache_write is None else cache_write * discount,
        tiers=tiers,
    )


class PricingTable:
    """Resolves model ids to prices against a :class:`PricingSnapshot`.

    Thread-safe for reads; the resolution cache is guarded by a lock.
    """

    def __init__(
        self,
        snapshot: PricingSnapshot,
        *,
        overrides: Mapping[str, PriceOverride] | None = None,
        batch_discount: float = DEFAULT_BATCH_DISCOUNT,
    ) -> None:
        """Build a table.

        Args:
            snapshot: The catalogue snapshot to price against.
            overrides: Manual prices, keyed by model id (``:batch`` suffix
                allowed). These take precedence over the catalogue.
            batch_discount: Multiplier applied when a ``:batch`` id is requested
                but not published by OpenRouter.
        """
        if not 0.0 < batch_discount <= 1.0:
            raise ValueError("batch_discount must be in (0, 1]")
        self.snapshot = snapshot
        self.overrides: dict[str, PriceOverride] = dict(overrides or {})
        self.batch_discount = batch_discount
        self._cache: dict[tuple[str, str | None], ModelPrice] = {}
        self._lock = threading.Lock()

    # -- construction ------------------------------------------------------ #

    @classmethod
    def load(
        cls,
        *,
        cache_path: Path | str | None = DEFAULT_CACHE_PATH,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        force_refresh: bool = False,
        allow_network: bool = True,
        url: str = OPENROUTER_MODELS_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        overrides: Mapping[str, PriceOverride] | None = None,
        batch_discount: float = DEFAULT_BATCH_DISCOUNT,
    ) -> PricingTable:
        """Load a snapshot (cache or network) and wrap it in a table."""
        snapshot = load_snapshot(
            cache_path=cache_path,
            ttl_seconds=ttl_seconds,
            force_refresh=force_refresh,
            allow_network=allow_network,
            url=url,
            timeout=timeout,
        )
        return cls(snapshot, overrides=overrides, batch_discount=batch_discount)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any] | list[dict[str, Any]],
        *,
        fetched_at: str | None = None,
        source_url: str = OPENROUTER_MODELS_URL,
        overrides: Mapping[str, PriceOverride] | None = None,
        batch_discount: float = DEFAULT_BATCH_DISCOUNT,
    ) -> PricingTable:
        """Build a table directly from an OpenRouter payload (tests, fixtures)."""
        data: Any = payload if isinstance(payload, list) else payload.get("data")
        if not isinstance(data, list):
            raise PricingFetchError("payload must contain a 'data' list")
        snapshot = _snapshot_from(
            fetched_at or _utc_now_iso(),
            source_url,
            [entry for entry in data if isinstance(entry, dict)],
            from_cache=False,
            ttl_seconds=DEFAULT_TTL_SECONDS,
            cache_path=None,
        )
        return cls(snapshot, overrides=overrides, batch_discount=batch_discount)

    # -- lookup ------------------------------------------------------------ #

    @property
    def fetched_at(self) -> str:
        """ISO-8601 timestamp of the underlying snapshot."""
        return self.snapshot.fetched_at

    @property
    def model_count(self) -> int:
        """Number of models in the snapshot."""
        return self.snapshot.model_count

    def __contains__(self, model: object) -> bool:
        return isinstance(model, str) and self.get(model) is not None

    def _override_price(self, key: str, requested: str) -> ModelPrice | None:
        override = self.overrides.get(key)
        if override is None:
            return None
        _, is_batch = split_batch_suffix(key)
        return ModelPrice(
            model=requested,
            resolved_id=key,
            input_per_token=override.input_per_1m / 1_000_000,
            output_per_token=override.output_per_1m / 1_000_000,
            is_batch=is_batch,
            source="override",
            fetched_at=self.snapshot.fetched_at,
            context_length=override.context_length,
        )

    def _resolve(self, model: str, provider: str | None) -> ModelPrice | None:
        requested = model.strip()
        if not requested:
            return None
        _, want_batch = split_batch_suffix(requested)

        # 1. Manual overrides, exact key first.
        price = self._override_price(requested, requested)
        if price is not None:
            return price

        entries = self.snapshot.entries
        fetched_at = self.snapshot.fetched_at

        # 2. Exact catalogue hit on the id as given (covers published ':batch').
        entry = entries.get(requested)
        if entry is not None:
            price = _price_from_entry(
                entry,
                requested=requested,
                resolved_id=requested,
                fetched_at=fetched_at,
                is_batch=want_batch,
            )
            if price is not None:
                return price

        # 3. Normalized candidates (native ids, dates, dotted versions).
        for candidate in model_candidates(requested, provider):
            if want_batch:
                batch_id = f"{candidate}:{BATCH_SUFFIX}"
                override = self._override_price(batch_id, requested)
                if override is not None:
                    return override
                entry = entries.get(batch_id)
                if entry is not None:
                    price = _price_from_entry(
                        entry,
                        requested=requested,
                        resolved_id=batch_id,
                        fetched_at=fetched_at,
                        is_batch=True,
                    )
                    if price is not None:
                        return price

            override = self._override_price(candidate, requested)
            if override is not None:
                return override

            entry = entries.get(candidate)
            if entry is None:
                continue
            price = _price_from_entry(
                entry,
                requested=requested,
                resolved_id=candidate,
                fetched_at=fetched_at,
                is_batch=want_batch,
                discount=self.batch_discount if want_batch else 1.0,
            )
            if price is not None:
                return price

        return None

    def price_of(self, model: str, provider: str | None = None) -> ModelPrice:
        """Resolve prices for a model id.

        Args:
            model: OpenRouter slug (``anthropic/claude-opus-5``), an optional
                ``:batch`` variant, or a native id (``claude-opus-4-5-20250929``).
            provider: Optional native provider hint to disambiguate bare names.

        Returns:
            A :class:`ModelPrice`.

        Raises:
            UnknownModelError: If nothing in the catalogue or overrides matches.
        """
        key = (model.strip(), provider)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        price = self._resolve(model, provider)
        if price is None:
            raise UnknownModelError(model, model_candidates(model, provider)[:8])

        with self._lock:
            self._cache[key] = price
        return price

    def get(self, model: str, provider: str | None = None) -> ModelPrice | None:
        """Like :meth:`price_of` but returns None instead of raising."""
        try:
            return self.price_of(model, provider)
        except UnknownModelError:
            return None

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
        *,
        provider: str | None = None,
    ) -> float:
        """Estimate the USD cost of one request.

        Raises:
            UnknownModelError: If the model cannot be priced.
        """
        return self.price_of(model, provider).cost(
            input_tokens, output_tokens, reasoning_tokens
        )

    # -- cross-check ------------------------------------------------------- #

    def crosscheck(
        self,
        reference: Mapping[str, PriceOverride] | None = None,
        *,
        tolerance: float = 0.005,
    ) -> list[PriceDiscrepancy]:
        """Compare catalogue prices against published provider prices.

        OpenRouter and the provider's own pricing page do not always agree — as
        of 2026-08-26 they differ by 2x on ``openai/gpt-5.6-sol`` and
        ``google/gemini-3.7-flash`` while matching exactly on every other
        model in the v0.4 roster. Both numbers are reported so the discrepancy
        is visible before money is committed, rather than one being silently
        preferred.

        Args:
            reference: Published prices to compare against. Defaults to
                :data:`VERIFIED_NATIVE_PRICES`.
            tolerance: Relative difference below which prices count as equal.

        Returns:
            One :class:`PriceDiscrepancy` per disagreeing model, sorted by id.
            Models absent from the catalogue are skipped.
        """
        table = VERIFIED_NATIVE_PRICES if reference is None else reference
        found: list[PriceDiscrepancy] = []
        for model, expected in table.items():
            entry = self.snapshot.entries.get(model)
            if entry is None:
                continue
            price = _price_from_entry(
                entry,
                requested=model,
                resolved_id=model,
                fetched_at=self.snapshot.fetched_at,
                is_batch=split_batch_suffix(model)[1],
            )
            if price is None:
                continue
            if _close(price.input_per_1m, expected.input_per_1m, tolerance) and _close(
                price.output_per_1m, expected.output_per_1m, tolerance
            ):
                continue
            found.append(
                PriceDiscrepancy(
                    model=model,
                    catalogue_input_per_1m=price.input_per_1m,
                    catalogue_output_per_1m=price.output_per_1m,
                    reference_input_per_1m=expected.input_per_1m,
                    reference_output_per_1m=expected.output_per_1m,
                    note=expected.note,
                )
            )
        return sorted(found, key=lambda item: item.model)

    # -- reporting --------------------------------------------------------- #

    def manifest_entry(self, models: Iterable[str] = ()) -> dict[str, Any]:
        """Provenance block for a run manifest, optionally with per-model prices."""
        manifest = self.snapshot.to_manifest()
        manifest["batch_discount"] = self.batch_discount
        manifest["overrides"] = sorted(self.overrides)
        manifest["discrepancies"] = [item.to_dict() for item in self.crosscheck()]
        priced: dict[str, Any] = {}
        for model in models:
            price = self.get(model)
            priced[model] = price.to_dict() if price is not None else None
        if priced:
            manifest["models"] = priced
        return manifest

    def search(self, substring: str) -> list[str]:
        """Catalogue ids containing ``substring`` (case-insensitive)."""
        needle = substring.lower()
        return sorted(mid for mid in self.snapshot.entries if needle in mid.lower())

    def iter_prices(self) -> Iterator[ModelPrice]:
        """Yield a :class:`ModelPrice` for every catalogue entry that has one."""
        for model_id, entry in self.snapshot.entries.items():
            price = _price_from_entry(
                entry,
                requested=model_id,
                resolved_id=model_id,
                fetched_at=self.snapshot.fetched_at,
                is_batch=split_batch_suffix(model_id)[1],
            )
            if price is not None:
                yield price


# --------------------------------------------------------------------------- #
# Native provider cross-check
# --------------------------------------------------------------------------- #

#: Prices read directly off the provider pricing pages on 2026-08-26, keyed by
#: OpenRouter-style id so they line up with the catalogue. USD per 1M tokens,
#: standard short-context tier unless the id carries ``:batch``.
#:
#: Sources:
#:   OpenAI    https://platform.openai.com/docs/pricing.md
#:   Anthropic https://docs.claude.com/en/docs/about-claude/pricing
#:   Google    https://ai.google.dev/gemini-api/docs/pricing
#:   xAI       https://docs.x.ai/docs/models  (page dated 2026-08-21)
#:
#: This map is **not** applied automatically. Pass it (or a subset) as
#: ``overrides=`` when you would rather bill against the provider's published
#: number than OpenRouter's, and use :meth:`PricingTable.crosscheck` to see
#: where the two disagree before committing spend.
VERIFIED_NATIVE_PRICES: dict[str, PriceOverride] = {
    # -- OpenAI ------------------------------------------------------------ #
    "openai/gpt-5.6-sol": PriceOverride(4.00, 20.00, note="openai 2026-08-26"),
    "openai/gpt-5.6-sol:batch": PriceOverride(2.00, 10.00, note="openai 2026-08-26"),
    "openai/gpt-5.6-terra": PriceOverride(2.00, 12.00, note="openai 2026-08-26"),
    "openai/gpt-5.6-terra:batch": PriceOverride(1.00, 6.00, note="openai 2026-08-26"),
    "openai/gpt-5.6-luna": PriceOverride(0.20, 1.20, note="openai 2026-08-26"),
    "openai/gpt-5.6-luna:batch": PriceOverride(0.10, 0.60, note="openai 2026-08-26"),
    "openai/gpt-5.2": PriceOverride(1.75, 14.00, note="openai 2026-08-26"),
    "openai/gpt-5.1": PriceOverride(1.25, 10.00, note="openai 2026-08-26"),
    # -- Anthropic --------------------------------------------------------- #
    "anthropic/claude-opus-5": PriceOverride(5.00, 25.00, note="anthropic 2026-08-26"),
    "anthropic/claude-opus-5:batch": PriceOverride(
        2.50, 12.50, note="anthropic 2026-08-26"
    ),
    "anthropic/claude-sonnet-5": PriceOverride(
        2.00, 10.00, note="anthropic 2026-08-26"
    ),
    "anthropic/claude-sonnet-5:batch": PriceOverride(
        1.00, 5.00, note="anthropic 2026-08-26"
    ),
    "anthropic/claude-haiku-4.5": PriceOverride(
        1.00, 5.00, note="anthropic 2026-08-26"
    ),
    "anthropic/claude-haiku-4.5:batch": PriceOverride(
        0.50, 2.50, note="anthropic 2026-08-26"
    ),
    "anthropic/claude-fable-5": PriceOverride(
        10.00, 50.00, note="anthropic 2026-08-26"
    ),
    # -- Google (standard tier, prompts <= 200k) --------------------------- #
    "google/gemini-3.1-pro-preview": PriceOverride(
        2.00, 12.00, note="google 2026-08-26 (<=200k prompt)"
    ),
    "google/gemini-3.1-pro-preview:batch": PriceOverride(
        1.00, 6.00, note="google 2026-08-26 (<=200k prompt)"
    ),
    "google/gemini-3.7-flash": PriceOverride(
        0.75, 3.75, note="google 2026-08-26 (promo through 2026-12-31)"
    ),
    "google/gemini-3.7-flash:batch": PriceOverride(
        0.375, 1.875, note="google 2026-08-26 (promo through 2026-12-31)"
    ),
    "google/gemini-3.6-flash": PriceOverride(
        0.75, 3.75, note="google 2026-08-26 (promo through 2026-12-31)"
    ),
    # -- xAI --------------------------------------------------------------- #
    "x-ai/grok-4.6": PriceOverride(2.00, 6.00, note="x-ai 2026-08-21"),
}


@dataclass(frozen=True, slots=True)
class PriceDiscrepancy:
    """A model where the catalogue and a reference price disagree."""

    model: str
    catalogue_input_per_1m: float
    catalogue_output_per_1m: float
    reference_input_per_1m: float
    reference_output_per_1m: float
    note: str = ""

    @property
    def input_ratio(self) -> float:
        """Catalogue input price divided by the reference input price."""
        if self.reference_input_per_1m == 0:
            return float("inf")
        return self.catalogue_input_per_1m / self.reference_input_per_1m

    @property
    def output_ratio(self) -> float:
        """Catalogue output price divided by the reference output price."""
        if self.reference_output_per_1m == 0:
            return float("inf")
        return self.catalogue_output_per_1m / self.reference_output_per_1m

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form, keeping *both* numbers."""
        return {
            "model": self.model,
            "catalogue": {
                "input_per_1m": self.catalogue_input_per_1m,
                "output_per_1m": self.catalogue_output_per_1m,
            },
            "reference": {
                "input_per_1m": self.reference_input_per_1m,
                "output_per_1m": self.reference_output_per_1m,
                "note": self.note,
            },
            "input_ratio": round(self.input_ratio, 6),
            "output_ratio": round(self.output_ratio, 6),
        }


# --------------------------------------------------------------------------- #
# Overrides I/O
# --------------------------------------------------------------------------- #


def load_overrides(path: Path | str) -> dict[str, PriceOverride]:
    """Load a manual override map from JSON.

    Expected shape::

        {"anthropic/claude-opus-5": {"input_per_1m": 5.0, "output_per_1m": 25.0}}
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"override file {path} must contain a JSON object")

    overrides: dict[str, PriceOverride] = {}
    for model, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"override for {model!r} must be an object")
        try:
            overrides[str(model)] = PriceOverride(
                input_per_1m=float(spec["input_per_1m"]),
                output_per_1m=float(spec["output_per_1m"]),
                context_length=(
                    int(spec["context_length"]) if spec.get("context_length") else None
                ),
                note=str(spec.get("note", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid override for {model!r}: {exc}") from exc
    return overrides


# --------------------------------------------------------------------------- #
# Module-level convenience (process-wide singleton)
# --------------------------------------------------------------------------- #

_default_table: PricingTable | None = None
_default_lock = threading.Lock()


def get_pricing_table(
    *,
    refresh: bool = False,
    **kwargs: Any,
) -> PricingTable:
    """Return the process-wide pricing table, loading it once.

    Args:
        refresh: Rebuild the singleton even if one exists.
        **kwargs: Forwarded to :meth:`PricingTable.load` on first use.
    """
    global _default_table
    with _default_lock:
        if _default_table is None or refresh:
            _default_table = PricingTable.load(**kwargs)
        return _default_table


def set_pricing_table(table: PricingTable | None) -> None:
    """Install (or clear) the process-wide table. Mainly for tests and CLIs."""
    global _default_table
    with _default_lock:
        _default_table = table


def price_of(model: str, provider: str | None = None) -> ModelPrice:
    """Resolve prices for a model using the process-wide table."""
    return get_pricing_table().price_of(model, provider)


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int = 0,
    *,
    provider: str | None = None,
) -> float:
    """Estimate one request's USD cost using the process-wide table."""
    return get_pricing_table().estimate_cost(
        model, input_tokens, output_tokens, reasoning_tokens, provider=provider
    )


__all__ = [
    "BATCH_SUFFIX",
    "DEFAULT_BATCH_DISCOUNT",
    "DEFAULT_CACHE_PATH",
    "DEFAULT_TTL_SECONDS",
    "OPENROUTER_MODELS_URL",
    "ModelPrice",
    "PriceOverride",
    "PriceTier",
    "PricingError",
    "PricingFetchError",
    "PricingSnapshot",
    "PricingTable",
    "PricingUnavailableError",
    "UnknownModelError",
    "batch_variant",
    "estimate_cost",
    "fetch_openrouter_models",
    "get_pricing_table",
    "load_overrides",
    "load_snapshot",
    "model_candidates",
    "price_of",
    "set_pricing_table",
    "split_batch_suffix",
]
