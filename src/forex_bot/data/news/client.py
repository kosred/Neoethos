from __future__ import annotations

import asyncio
import glob
import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ...core.config import Settings
from .scorers import OpenAIScorer
from .searchers import PerplexitySearcher
from .store import NewsDatabase, NewsEvent

logger = logging.getLogger(__name__)

try:
    import forex_bindings as _fb  # type: ignore
except Exception:
    _fb = None  # type: ignore


class _NewsFrame:
    def __init__(self, data: dict[str, np.ndarray] | None = None, *, index: np.ndarray | None = None):
        self._data: dict[str, np.ndarray] = {}
        self.index = np.asarray(index) if index is not None else np.zeros(0, dtype="datetime64[ns]")
        self.attrs: dict[str, Any] = {}
        if data:
            for k, v in data.items():
                self[str(k)] = v

    @property
    def columns(self) -> list[str]:
        return list(self._data.keys())

    @property
    def empty(self) -> bool:
        return len(self.index) <= 0

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[str(key)]

    def __setitem__(self, key: str, value: Any) -> None:
        arr = np.asarray(value)
        if self.index.size <= 0:
            self.index = np.arange(arr.reshape(-1).shape[0], dtype=np.int64).astype("datetime64[ns]")
        self._data[str(key)] = arr


def _make_frame(data: Any | None = None, *, index: Any | None = None, columns: Any | None = None) -> Any:
    idx_np = _to_datetime64_ns(index) if index is not None else np.zeros(0, dtype="datetime64[ns]")
    out = _NewsFrame(index=idx_np)
    if data is not None:
        if isinstance(data, dict):
            for k, v in data.items():
                out[k] = v
        else:
            try:
                for k in (columns or []):
                    out[str(k)] = np.asarray(data[str(k)])
            except Exception:
                pass
    elif columns is not None:
        for k in columns:
            out[str(k)] = np.zeros(idx_np.size, dtype=np.float32)
    return out


def _is_dataframe(value: Any) -> bool:
    return bool(hasattr(value, "columns") and hasattr(value, "index") and hasattr(value, "to_dict"))


def _to_datetime64_ns(values: Any) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    if arr.size <= 0:
        return np.zeros(0, dtype="datetime64[ns]")
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]", copy=False)
    if arr.dtype.kind in {"i", "u"}:
        vals = arr.astype(np.int64, copy=False)
        vmax = int(np.max(np.abs(vals))) if vals.size > 0 else 0
        if vmax > 10**14:
            return vals.astype("datetime64[ns]")
        if vmax > 10**11:
            return vals.astype("datetime64[ms]").astype("datetime64[ns]")
        return vals.astype("datetime64[s]").astype("datetime64[ns]")
    if arr.dtype.kind == "f":
        vals = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)
        return vals.astype("datetime64[s]").astype("datetime64[ns]")
    with np.errstate(all="ignore"):
        try:
            return arr.astype("datetime64[ns]")
        except Exception:
            return np.zeros(0, dtype="datetime64[ns]")


def _datetime_to_ns(dt: datetime | None) -> int | None:
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    try:
        return int(np.datetime64(dt.replace(tzinfo=None), "ns").astype(np.int64))
    except Exception:
        return None


def _index_to_ns_int64(values: Any) -> np.ndarray:
    arr = _to_datetime64_ns(values)
    if arr.size <= 0:
        return np.zeros(0, dtype=np.int64)
    return arr.astype(np.int64, copy=False)


def _rust_sorted_index_order(index_like: Any) -> np.ndarray | None:
    if _fb is None or not hasattr(_fb, "sorted_index_order"):
        return None
    idx_ns = _index_to_ns_int64(index_like)
    if idx_ns.size <= 0:
        return None
    try:
        out = _fb.sorted_index_order(np.asarray(idx_ns, dtype=np.int64))
    except Exception:
        return None
    order = np.asarray(out, dtype=np.int64).reshape(-1)
    if order.size != idx_ns.size:
        return None
    return order


def _sorted_time_order(index_like: Any) -> np.ndarray | None:
    idx_ns = _index_to_ns_int64(index_like)
    if idx_ns.size <= 1:
        return None
    if not bool(np.any(idx_ns[1:] < idx_ns[:-1])):
        return None
    order = _rust_sorted_index_order(idx_ns)
    if order is not None:
        return order
    return np.argsort(idx_ns, kind="mergesort")


def _rust_aggregate_news_features(
    base_idx_ns: np.ndarray,
    event_idx_ns: np.ndarray,
    event_sent: np.ndarray,
    event_conf: np.ndarray,
    *,
    lookback_ns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if _fb is None or not hasattr(_fb, "aggregate_news_features"):
        return None
    try:
        out = _fb.aggregate_news_features(
            np.asarray(base_idx_ns, dtype=np.int64),
            np.asarray(event_idx_ns, dtype=np.int64),
            np.asarray(event_sent, dtype=np.float64),
            np.asarray(event_conf, dtype=np.float64),
            int(max(0, lookback_ns)),
        )
    except Exception:
        return None
    if not isinstance(out, tuple) or len(out) != 4:
        return None
    sent, conf, count, recency = out
    sent_arr = np.asarray(sent, dtype=np.float32).reshape(-1)
    conf_arr = np.asarray(conf, dtype=np.float32).reshape(-1)
    count_arr = np.asarray(count, dtype=np.float32).reshape(-1)
    recency_arr = np.asarray(recency, dtype=np.float32).reshape(-1)
    n = int(np.asarray(base_idx_ns, dtype=np.int64).size)
    if sent_arr.size != n or conf_arr.size != n or count_arr.size != n or recency_arr.size != n:
        return None
    return sent_arr, conf_arr, count_arr, recency_arr


def _rust_aggregate_news_activation(
    base_idx_ns: np.ndarray,
    event_idx_ns: np.ndarray,
    event_sent: np.ndarray,
    event_conf: np.ndarray,
    *,
    back_ns: int,
    fwd_ns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if _fb is None or not hasattr(_fb, "aggregate_news_activation"):
        return None
    try:
        out = _fb.aggregate_news_activation(
            np.asarray(base_idx_ns, dtype=np.int64),
            np.asarray(event_idx_ns, dtype=np.int64),
            np.asarray(event_sent, dtype=np.float64),
            np.asarray(event_conf, dtype=np.float64),
            int(max(0, back_ns)),
            int(max(0, fwd_ns)),
        )
    except Exception:
        return None
    if not isinstance(out, tuple) or len(out) != 3:
        return None
    nearby, conf_max, sent_max = out
    nearby_arr = np.asarray(nearby, dtype=np.int8).reshape(-1)
    conf_arr = np.asarray(conf_max, dtype=np.float32).reshape(-1)
    sent_arr = np.asarray(sent_max, dtype=np.float32).reshape(-1)
    n = int(np.asarray(base_idx_ns, dtype=np.int64).size)
    if nearby_arr.size != n or conf_arr.size != n or sent_arr.size != n:
        return None
    return nearby_arr, conf_arr, sent_arr


class SentimentAnalyzer:
    def __init__(self, settings: Settings):
        self.settings = settings
        db_path = Path(settings.system.cache_dir) / "news.sqlite"
        self.db = NewsDatabase(db_path)
        self.openai_scorer = OpenAIScorer(settings)
        self.pplx_searcher = PerplexitySearcher(settings)
        self._local_seeded = False
        self.degraded = False
        self.last_error: str | None = None
        self.cross_influence = {
            "AUD": ["NZD", "CNY", "XAU", "XAG"],
            "NZD": ["AUD", "CNY"],
            "CAD": ["USD", "OIL"],
            "CHF": ["EUR"],
            "EUR": ["CHF"],
            "GBP": ["EUR"],
            "USD": ["CAD", "JPY", "XAU", "OIL", "BTC"], # USD moves everything
            "JPY": ["USD"],
            "XAU": ["USD", "CNY", "FED", "RATE"], # Gold is moved by USD and Rates
            "OIL": ["CAD", "USD", "ME"] # Oil moves CAD and is moved by Middle East
        }

        self.currency_synonyms = {
            "USD": ["USD", "US DOLLAR", "GREENBACK"],
            "EUR": ["EUR", "EURO", "EUROZONE"],
        }
        self._last_daily_fetch: date | None = None
        self._seed_local_archives()

    def ensure_news_history(
        self, symbol: str, start: datetime, end: datetime, force: bool = False, use_live_llm: bool = True
    ) -> int:
        """
        Fetch, score, and persist news between start/end.
        Returns the count of newly inserted events (best effort).
        """
        if not force:
            latest = self.db.latest_timestamp()
            if latest and latest >= end:
                return 0
        events = self.fetch_news_bundle(symbol, start, end, use_live_llm=use_live_llm)
        if not use_live_llm:
            scored = events  # Keep existing sentiment/confidence (e.g., from archives/DB)
        else:
            scored = self.score_events(events)
        try:
            # When LLM scoring is enabled, update existing rows so backfilled archives can be rescored.
            return self.db.insert_events(scored, update_existing=bool(use_live_llm))
        except Exception as exc:
            logger.warning(f"Failed to persist news events: {exc}")
            return 0

    def update_sentiments(self, force: bool = False) -> int:
        """
        Fetch new articles and update sentiment scores.
        By default this runs at most once per UTC day to avoid constant polling; set force=True to override.
        """
        count = 0
        now = datetime.now(UTC)
        start = datetime(now.year, now.month, now.day, tzinfo=UTC)
        end = now
        try:
            count = self.ensure_news_history(symbol="ALL", start=start, end=end, force=force)
        except Exception as exc:
            logger.warning(f"Sentiment update skipped: {exc}")
        return count

    def fetch_news_bundle(
        self, symbol: str, start: datetime, end: datetime, use_live_llm: bool = True
    ) -> list[NewsEvent]:
        """
        End-to-end fetch using Perplexity (primary) and OpenAI (augment/fallback).
        Returns list of NewsEvent (unscored).
        """
        self.degraded = False
        self.last_error = None
        all_events: list[NewsEvent] = []
        target_currencies = [symbol[:3], symbol[3:]]

        existing_db_events = self.db.fetch_events(start, end, currencies=target_currencies)
        all_events.extend(existing_db_events)

        known_event_ids = set()
        for ev in existing_db_events:
            if ev.published_at and ev.title:
                # Use date+hour for deduplication to avoid microsecond differences
                pub_key = ev.published_at.strftime("%Y-%m-%d %H")
                known_event_ids.add((ev.title.strip().lower(), pub_key))

        if use_live_llm and self.settings.news.perplexity_enabled:
            try:
                # HPC FIX: Institutional-Grade Macro Search Queries
                macro_query = (
                    f"High-impact fundamental catalysts for {symbol} and {target_currencies}: "
                    "Focus on Central Bank policy shifts (Fed, ECB, BoJ), "
                    "Interest Rate differentials, Liquidity shocks, and Geopolitical risk. "
                    f"Timeframe: {start.date()} to {end.date()}. "
                    "Prioritize Bloomberg, Reuters, and Financial Times sources."
                )
                pplx_events = self._fetch_perplexity_headlines(macro_query, target_currencies)
                new_pplx_events = [
                    ev
                    for ev in pplx_events
                    if (ev.title.strip().lower(), ev.published_at.isoformat()) not in known_event_ids
                ]
                if new_pplx_events:
                    logger.info(f"Perplexity found {len(new_pplx_events)} new relevant news items.")
                    all_events.extend(new_pplx_events)
                    for ev in new_pplx_events:
                        known_event_ids.add((ev.title.strip().lower(), ev.published_at.isoformat()))
            except Exception as e:
                self.degraded = True
                self.last_error = f"Perplexity search failed: {e}"
                logger.warning(self.last_error)

        if use_live_llm and getattr(self.settings.news, "openai_news_enabled", True):
            try:
                oa_query = (
                    f"Latest high-impact FX and macro headlines affecting {symbol} "
                    f"between {start.isoformat()} and {end.isoformat()}"
                )
                oa_events = self._fetch_openai_headlines(oa_query, target_currencies)
                new_oa_events = [
                    ev
                    for ev in oa_events
                    if (ev.title.strip().lower(), ev.published_at.isoformat()) not in known_event_ids
                ]
                if new_oa_events:
                    logger.info(f"OpenAI provided {len(new_oa_events)} headline candidates.")
                    all_events.extend(new_oa_events)
                    for ev in new_oa_events:
                        known_event_ids.add((ev.title.strip().lower(), ev.published_at.isoformat()))
            except Exception as exc:
                logger.warning(f"OpenAI headline fetch failed: {exc}")

        dedup_final: dict[str, NewsEvent] = {}
        for ev in all_events:
            key = (ev.url or ev.title).lower().strip()
            if key in dedup_final:
                existing = dedup_final[key]
                if ev.source in ["Perplexity", "OpenAI"] and existing.source not in ["Perplexity", "OpenAI"]:
                    dedup_final[key] = ev
            else:
                dedup_final[key] = ev
        all_events = list(dedup_final.values())

        for ev in all_events:
            cur_list = {c for c in (ev.currencies or []) if c}
            for c in list(cur_list):
                for xc in self.cross_influence.get(c, []):
                    cur_list.add(xc)
            ev.currencies = list(cur_list)

        return all_events

    def _parse_timestamp(self, raw: Any) -> datetime:
        if isinstance(raw, datetime):
            return raw.astimezone(UTC)
        if isinstance(raw, (int, float)):
            try:
                val = float(raw)
                abs_val = abs(val)
                if abs_val > 1e14:
                    val /= 1e9
                elif abs_val > 1e11:
                    val /= 1e3
                return datetime.fromtimestamp(val, tz=UTC)
            except Exception:
                pass
        if isinstance(raw, str) and raw:
            # Clean string
            clean = raw.strip().replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(clean).astimezone(UTC)
            except Exception:
                for fmt in (
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d",
                    "%d/%m/%Y %H:%M:%S",
                    "%d/%m/%Y",
                    "%m/%d/%Y %H:%M:%S",
                    "%m/%d/%Y",
                ):
                    try:
                        return datetime.strptime(clean, fmt).replace(tzinfo=UTC)
                    except Exception:
                        continue
                logger.debug(f"Timestamp parse failed for '{raw}'")

        # Fallback to epoch if parsing fails - DO NOT use now() to avoid triggering kill switches
        return datetime(1970, 1, 1, tzinfo=UTC)

    def _seed_local_archives(self) -> None:
        """
        Ingest local news CSV archives (best-effort) into the news DB.
        Supports improved currency heuristics and column normalization.
        """
        if self._local_seeded:
            return

        # Default directory + configured glob
        search_paths = [Path("data/news"), Path("data")]
        glob_pattern = getattr(self.settings.news, "news_local_glob", "")

        files = []
        if glob_pattern:
            files.extend(glob.glob(glob_pattern))

        for base in search_paths:
            if base.exists():
                files.extend(list(base.glob("*.csv")))
                files.extend(list(base.glob("*.json")))

        if not files:
            self._local_seeded = True
            return

        total_inserted = 0
        # Deduplicate file paths
        unique_files = {str(p) for p in files}

        impact_map = {
            "HIGH": 0.9,
            "MEDIUM": 0.6,
            "MED": 0.6,
            "LOW": 0.35,
            "NON-ECONOMIC": 0.2,
            "NONE": 0.2,
        }

        def _to_float(value: Any, default: float = 0.0) -> float:
            if value is None:
                return float(default)
            if isinstance(value, str) and value.strip() == "":
                return float(default)
            try:
                out = float(value)
                if out != out:
                    return float(default)
                return out
            except Exception:
                return float(default)

        def _parse_currencies(raw: Any, title: str) -> list[str]:
            txt = str(raw or "").replace(";", ",")
            tokens = [
                tok.strip().upper()
                for tok in txt.split(",")
                if tok and len(tok.strip()) >= 3 and tok.strip().isalpha()
            ]
            if tokens:
                return list(dict.fromkeys(tokens))[:3]
            upper = str(title or "").upper()
            return [c for c in ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "XAU", "XAG"] if c in upper]

        for path_str in unique_files:
            path = Path(path_str)
            try:
                from .ingest import _extract_rows

                rows = _extract_rows(path)
            except Exception as exc:
                logger.warning(f"Failed to read news archive {path}: {exc}")
                continue

            if not rows:
                continue

            events: list[NewsEvent] = []
            for i, row_raw in enumerate(rows):
                row = {str(k).strip().lower(): v for k, v in dict(row_raw).items()}
                title = str(row.get("title") or row.get("event") or row.get("headline") or "").strip()
                if not title:
                    continue
                ts_raw = row.get("published_at", row.get("timestamp", row.get("date", row.get("time", row.get("datetime")))))
                ts = self._parse_timestamp(ts_raw)
                if ts.year < 1990:
                    continue

                summary = str(row.get("summary") or row.get("detail") or row.get("description") or row.get("snippet") or "").strip()
                extras = []
                if "actual" in row:
                    extras.append(f"Actual={row.get('actual')}")
                if "forecast" in row:
                    extras.append(f"Forecast={row.get('forecast')}")
                if "previous" in row:
                    extras.append(f"Previous={row.get('previous')}")
                if extras:
                    summary = (summary + " | " + ", ".join(extras)).strip(" |,")

                url = str(row.get("url") or "").strip()
                if not url:
                    url = f"localcsv://{path.name}-{i}"
                sentiment = _to_float(row.get("sentiment"), 0.0)
                confidence = _to_float(row.get("confidence"), 0.0)
                impact = str(row.get("impact") or "").strip().upper()
                if impact in impact_map:
                    confidence = float(impact_map[impact])
                currencies = _parse_currencies(row.get("currencies", row.get("currency", row.get("ccy"))), title)

                try:
                    events.append(
                        NewsEvent(
                            title=title,
                            summary=summary,
                            url=str(url),
                            source="LocalArchive",
                            published_at=ts,
                            sentiment=sentiment,
                            confidence=confidence,
                            currencies=currencies,
                        )
                    )
                except Exception as exc:
                    logger.debug(f"Skipping malformed news row {i} in {path.name}: {exc}")

            if events:
                try:
                    inserted = self.db.insert_events(events)
                    total_inserted += inserted
                    logger.info(f"Seeded {inserted} events from {path.name}")
                except Exception as exc:
                    logger.warning(f"Failed to seed archive {path}: {exc}")

        if total_inserted > 0:
            logger.info(f"Local news archive ingest complete. Inserted {total_inserted} events.")
        self._local_seeded = True

    def _fetch_perplexity_headlines(self, query: str, currencies: list[str]) -> list[NewsEvent]:
        events: list[NewsEvent] = []
        if not self.pplx_searcher.available:
            return events
        raw = self.pplx_searcher.search(
            query,
            num_results=int(getattr(self.settings.news, "perplexity_num_results", 10)),
        )
        for item in raw:
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            ts = self._parse_timestamp(item.get("published_at") or item.get("date"))
            url = str(item.get("url") or f"perplexity://{hash(title + ts.isoformat())}")
            summary = str(item.get("snippet", "") or "")
            events.append(
                NewsEvent(
                    title=title,
                    summary=summary,
                    url=url,
                    source="Perplexity",
                    published_at=ts,
                    sentiment=0.0,
                    confidence=0.0,
                    currencies=currencies,
                )
            )
        return events

    def _fetch_openai_headlines(self, query: str, currencies: list[str]) -> list[NewsEvent]:
        events: list[NewsEvent] = []
        if not self.openai_scorer.available or self.openai_scorer.client is None:
            return events

        system_prompt = (
            "You are a real-time FX news scout. "
            "Return high-impact macro/FX headlines as JSON: "
            '{ "headlines": [ {"title": str, "summary": str, "url": str, "published_at": iso8601} ] } '
            "Only include items from reputable sources and avoid speculation."
        )
        try:
            resp = self.openai_scorer.client.chat.completions.create(
                model=self.openai_scorer.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                max_completion_tokens=self.openai_scorer.max_tokens,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)
            headlines = data.get("headlines", [])
            if isinstance(headlines, dict):
                headlines = headlines.get("items", [])
            for item in headlines:
                title = str(item.get("title", "")).strip()
                if not title:
                    continue
                ts = self._parse_timestamp(item.get("published_at") or item.get("date"))
                url = str(item.get("url") or f"openai://{hash(title + ts.isoformat())}")
                summary = str(item.get("summary") or item.get("snippet") or "")
                events.append(
                    NewsEvent(
                        title=title,
                        summary=summary,
                        url=url,
                        source="OpenAI",
                        published_at=ts,
                        sentiment=0.0,
                        confidence=0.0,
                        currencies=currencies,
                    )
                )
        except Exception as exc:
            logger.warning(f"OpenAI headline parsing failed: {exc}")
        return events

    async def score_events_async(
        self,
        events: list[NewsEvent],
        max_openai: int | None = None,
        *,
        force_openai: bool = False,
    ) -> list[NewsEvent]:
        """
        Score events with OpenAI in parallel (HPC Optimized).
        """
        if max_openai is None:
            max_openai = int(getattr(self.settings.news, "openai_max_events_per_fetch", 50) or 50)
        
        # Parallel Scoring Logic
        sem = asyncio.Semaphore(10) # Max 10 concurrent API calls
        openai_available = self.openai_scorer.available
        
        async def _score_single(ev):
            if not openai_available: return ev
            try:
                async with sem:
                    # Run the synchronous scorer in a thread to keep loop responsive
                    result = await asyncio.to_thread(self.openai_scorer.score, ev.title, ev.currencies or [])
                    if result:
                        ev.sentiment = result.sentiment
                        ev.confidence = result.confidence
                        self.db.set_cached_score(ev.url, ev.title, result.sentiment, result.confidence, result.direction)
            except Exception as e:
                logger.warning(f"Async scoring failed for {ev.title}: {e}")
            return ev

        # Skip cached ones first
        to_score = []
        final_scored = []
        for ev in events:
            cached = self.db.get_cached_score(ev.url, ev.title)
            if cached:
                ev.sentiment = cached["sentiment"]
                ev.confidence = cached["confidence"]
                final_scored.append(ev)
            else:
                to_score.append(ev)
        
        if to_score:
            logger.info(f"HPC: Scoring {len(to_score)} new events in parallel...")
            parallel_results = await asyncio.gather(*[_score_single(ev) for ev in to_score[:max_openai]])
            final_scored.extend(parallel_results)
            # Add remaining unscored if cap hit
            if len(to_score) > max_openai:
                final_scored.extend(to_score[max_openai:])
        
        logger.info(
            f"[TELEMETRY] News: Processed {len(events)} events. "
            f"LLM Scores: {len(to_score[:max_openai])} | Cached: {len(events) - len(to_score)}"
        )
                
        return final_scored

    def score_events(self, *args, **kwargs):
        # Compatibility wrapper for sync callers
        return asyncio.run(self.score_events_async(*args, **kwargs))

    def rescore_existing_events(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        max_events: int | None = None,
        only_missing: bool | None = None,
    ) -> int:
        """
        Rescore existing DB/archive events (no web search) and persist updated sentiment/confidence.

        Intended for backfilled archives that have missing or heuristic scores.
        """
        if not self.openai_scorer.available:
            return 0

        if max_events is None:
            try:
                max_events = int(getattr(self.settings.news, "auto_rescore_max_events", 200) or 200)
            except Exception:
                max_events = 200
        max_events = int(max(0, max_events))
        if max_events <= 0:
            return 0

        if only_missing is None:
            only_missing = bool(getattr(self.settings.news, "auto_rescore_only_missing", True))

        target_currencies = [symbol[:3], symbol[3:]] if isinstance(symbol, str) and len(symbol) == 6 else []
        try:
            events = self.db.fetch_events(start, end, currencies=target_currencies or None)
        except Exception as exc:
            logger.warning(f"Rescore fetch failed: {exc}")
            return 0

        if not events:
            return 0

        candidates: list[NewsEvent] = []
        for ev in events:
            if not only_missing:
                candidates.append(ev)
                continue
            try:
                sent = float(ev.sentiment or 0.0)
                conf = float(ev.confidence or 0.0)
                if sent == 0.0 and conf == 0.0:
                    candidates.append(ev)
            except Exception:
                continue

        if not candidates:
            return 0

        # Prefer most recent rows first to improve current training/live relevance under a tight budget.
        candidates.sort(key=lambda e: e.published_at or datetime(1970, 1, 1, tzinfo=UTC), reverse=True)
        candidates = candidates[:max_events]

        try:
            scored = self.score_events(candidates, max_openai=max_events, force_openai=True)
        except Exception as exc:
            logger.warning(f"Rescore scoring failed: {exc}")
            return 0

        try:
            self.db.insert_events(scored, update_existing=True)
        except Exception as exc:
            logger.warning(f"Rescore persist failed: {exc}")

        return len(scored)

    def build_features(
        self,
        events: list[NewsEvent],
        base_index: Any,
        *,
        include_activation: bool = False,
    ) -> Any:
        """
        Aggregate news events into time-aligned features for models.

        Outputs columns:
          - news_sentiment: last known sentiment (ffill)
          - news_confidence: last known confidence (ffill)
          - news_count: number of events in the prior 6 hours (inclusive)
          - news_recency_minutes: minutes since the most recent event (9999 if none)
        """
        if events is None:
            return _make_frame(index=base_index) if base_index is not None else _make_frame()
        if _is_dataframe(events):
            try:
                events = events.to_dict("records")
            except Exception:
                events = list(events)
        if len(events) == 0 or base_index is None or len(base_index) == 0:
            return _make_frame(index=base_index) if base_index is not None else _make_frame()

        base_times = _to_datetime64_ns(base_index)
        if base_times.size <= 0:
            return _make_frame(index=base_index)
        valid = ~np.isnat(base_times)
        if not np.any(valid):
            return _make_frame(index=base_index)
        base_times = base_times[valid]
        order = _sorted_time_order(base_times)
        if order is not None:
            base_times = base_times[order]
        if base_times.size > 1:
            keep = np.ones(base_times.size, dtype=bool)
            keep[1:] = base_times[1:] != base_times[:-1]
            base_times = base_times[keep]
        base_ns = base_times.astype(np.int64, copy=False)
        n = int(base_ns.shape[0])
        if n <= 0:
            return _make_frame(index=base_index)

        ev_ns_list: list[int] = []
        ev_sent_list: list[float] = []
        ev_conf_list: list[float] = []
        for ev in events:
            try:
                ts_raw = getattr(ev, "published_at", None)
                if ts_raw is None and isinstance(ev, dict):
                    ts_raw = ev.get("published_at")
                ts = self._parse_timestamp(ts_raw)
                ts_ns = _datetime_to_ns(ts)
                if ts_ns is None:
                    continue

                sent_raw = getattr(ev, "sentiment", None)
                conf_raw = getattr(ev, "confidence", None)
                if isinstance(ev, dict):
                    sent_raw = ev.get("sentiment", sent_raw)
                    conf_raw = ev.get("confidence", conf_raw)

                ev_ns_list.append(int(ts_ns))
                ev_sent_list.append(float(sent_raw or 0.0))
                ev_conf_list.append(float(conf_raw or 0.0))
            except Exception:
                continue
        base_idx_out = base_times

        if not ev_ns_list:
            feat = _make_frame(index=base_idx_out)
            feat["news_sentiment"] = np.zeros(n, dtype=np.float32)
            feat["news_confidence"] = np.zeros(n, dtype=np.float32)
            feat["news_count"] = np.zeros(n, dtype=np.float32)
            feat["news_recency_minutes"] = np.full(n, 9999.0, dtype=np.float32)
            if include_activation:
                feat["news_nearby"] = np.zeros(n, dtype=np.int8)
                feat["news_conf_max"] = np.zeros(n, dtype=np.float32)
                feat["news_sent_max"] = np.zeros(n, dtype=np.float32)
            return feat

        ev_ns = np.asarray(ev_ns_list, dtype=np.int64)
        ev_sent = np.asarray(ev_sent_list, dtype=np.float64)
        ev_conf = np.asarray(ev_conf_list, dtype=np.float64)
        ev_order = _sorted_time_order(ev_ns)
        if ev_order is not None:
            ev_ns = ev_ns[ev_order]
            ev_sent = ev_sent[ev_order]
            ev_conf = ev_conf[ev_order]

        rust = _rust_aggregate_news_features(
            base_ns,
            ev_ns,
            ev_sent,
            ev_conf,
            lookback_ns=6 * 60 * 1_000_000_000,
        )
        if rust is not None:
            news_sentiment, news_confidence, news_count, recency = rust
        else:
            uniq_ns, inv = np.unique(ev_ns, return_inverse=True)
            counts = np.bincount(inv, minlength=uniq_ns.size).astype(np.float64)
            sent_sum = np.bincount(inv, weights=ev_sent, minlength=uniq_ns.size)
            conf_sum = np.bincount(inv, weights=ev_conf, minlength=uniq_ns.size)
            sent_mean = np.divide(sent_sum, np.maximum(counts, 1.0))
            conf_mean = np.divide(conf_sum, np.maximum(counts, 1.0))

            prev_pos = np.searchsorted(uniq_ns, base_ns, side="right") - 1
            news_sentiment = np.zeros(n, dtype=np.float32)
            news_confidence = np.zeros(n, dtype=np.float32)
            valid_prev = prev_pos >= 0
            if np.any(valid_prev):
                take = np.clip(prev_pos[valid_prev], 0, uniq_ns.size - 1)
                news_sentiment[valid_prev] = sent_mean[take].astype(np.float32, copy=False)
                news_confidence[valid_prev] = conf_mean[take].astype(np.float32, copy=False)

            right = np.searchsorted(uniq_ns, base_ns, side="right")
            left = np.searchsorted(uniq_ns, base_ns - np.int64(6 * 60 * 1_000_000_000), side="left")
            news_count = (right - left).astype(np.float32)

            recency = np.full(n, 9999.0, dtype=np.float32)
            has_prev = right > 0
            if np.any(has_prev):
                prev_idx = right[has_prev] - 1
                prev_ns = uniq_ns[np.clip(prev_idx, 0, uniq_ns.size - 1)]
                recency[has_prev] = ((base_ns[has_prev] - prev_ns) / 60_000_000_000.0).astype(np.float32, copy=False)

        feat = _make_frame(index=base_idx_out)
        feat["news_sentiment"] = news_sentiment
        feat["news_confidence"] = news_confidence
        feat["news_count"] = news_count
        feat["news_recency_minutes"] = recency

        if include_activation:
            act = self._build_activation(events, base_idx_out, back_minutes=60, fwd_minutes=15)
            for col in act.columns:
                feat[col] = act[col]
        return feat

    def _build_activation(
        self, events: list[NewsEvent], base_index: Any, back_minutes: int = 60, fwd_minutes: int = 15
    ) -> Any:
        """Flag whether news is near each timestamp and the max sentiment/confidence in window."""
        base_times = _to_datetime64_ns(base_index)
        if base_times.size <= 0:
            df = _make_frame(index=base_index)
            df["news_nearby"] = 0
            df["news_conf_max"] = 0.0
            df["news_sent_max"] = 0.0
            return df
        valid = ~np.isnat(base_times)
        base_times = base_times[valid]
        if base_times.size <= 0:
            df = _make_frame(index=base_index)
            df["news_nearby"] = 0
            df["news_conf_max"] = 0.0
            df["news_sent_max"] = 0.0
            return df
        base_ns = base_times.astype(np.int64, copy=False)
        n = int(base_ns.size)

        ev_ns_list: list[int] = []
        ev_sent_list: list[float] = []
        ev_conf_list: list[float] = []
        for ev in events:
            ts_raw = None
            if isinstance(ev, dict):
                ts_raw = ev.get("published_at")
            else:
                ts_raw = getattr(ev, "published_at", None)
            ts = ts_raw if isinstance(ts_raw, datetime) else self._parse_timestamp(ts_raw)
            ts_ns = _datetime_to_ns(ts)
            if ts_ns is None:
                continue
            ev_ns_list.append(int(ts_ns))
            if isinstance(ev, dict):
                ev_sent_list.append(float(ev.get("sentiment", 0.0) or 0.0))
                ev_conf_list.append(float(ev.get("confidence", 0.0) or 0.0))
            else:
                ev_sent_list.append(float(getattr(ev, "sentiment", 0.0) or 0.0))
                ev_conf_list.append(float(getattr(ev, "confidence", 0.0) or 0.0))

        base_idx_out = base_times
        df = _make_frame(index=base_idx_out)
        if not ev_ns_list:
            df["news_nearby"] = np.zeros(n, dtype=np.int8)
            df["news_conf_max"] = np.zeros(n, dtype=np.float32)
            df["news_sent_max"] = np.zeros(n, dtype=np.float32)
            return df

        ev_ns = np.asarray(ev_ns_list, dtype=np.int64)
        ev_sent = np.asarray(ev_sent_list, dtype=np.float32)
        ev_conf = np.asarray(ev_conf_list, dtype=np.float32)
        ev_order = _sorted_time_order(ev_ns)
        if ev_order is not None:
            ev_ns = ev_ns[ev_order]
            ev_sent = ev_sent[ev_order]
            ev_conf = ev_conf[ev_order]

        back_ns = np.int64(max(0, int(back_minutes)) * 60 * 1_000_000_000)
        fwd_ns = np.int64(max(0, int(fwd_minutes)) * 60 * 1_000_000_000)
        rust = _rust_aggregate_news_activation(
            base_ns,
            ev_ns,
            ev_sent,
            ev_conf,
            back_ns=int(back_ns),
            fwd_ns=int(fwd_ns),
        )
        if rust is not None:
            nearby, conf_max, sent_max = rust
        else:
            left = np.searchsorted(ev_ns, base_ns - back_ns, side="left")
            right = np.searchsorted(ev_ns, base_ns + fwd_ns, side="right")
            nearby = (right > left).astype(np.int8)
            conf_max = np.zeros(n, dtype=np.float32)
            sent_max = np.zeros(n, dtype=np.float32)
            for i in range(n):
                l = int(left[i])
                r = int(right[i])
                if r <= l:
                    continue
                conf_max[i] = float(np.max(ev_conf[l:r]))
                sent_max[i] = float(np.max(ev_sent[l:r]))
        df["news_nearby"] = nearby
        df["news_conf_max"] = conf_max
        df["news_sent_max"] = sent_max
        return df


_ANALYZER: SentimentAnalyzer | None = None
_ANALYZER_LOCK = asyncio.Lock()


async def get_sentiment_analyzer(settings: Settings) -> SentimentAnalyzer:
    """Thread-safe singleton for SentimentAnalyzer with async lock to prevent race conditions."""
    global _ANALYZER
    async with _ANALYZER_LOCK:
        if _ANALYZER is None:
            _ANALYZER = SentimentAnalyzer(settings)
        return _ANALYZER

