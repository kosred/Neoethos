from __future__ import annotations

"""
Local News Ingestion for Backfilling.

Scans data/news directory for CSV/JSON files and populates the SQLite database.
This allows the bot to have historical context without expensive API calls.
"""

import csv
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_POLARS_MOD: Any | None = None
_POLARS_IMPORT_FAILED = False


def _polars_module(*, required: bool = False):
    global _POLARS_MOD, _POLARS_IMPORT_FAILED
    if _POLARS_MOD is not None:
        return _POLARS_MOD
    if _POLARS_IMPORT_FAILED:
        if required:
            raise RuntimeError("polars is unavailable")
        return None
    try:
        import polars as _pl  # type: ignore

        _POLARS_MOD = _pl
        return _pl
    except Exception as exc:
        _POLARS_IMPORT_FAILED = True
        if required:
            raise RuntimeError("polars is unavailable") from exc
        return None


def _normalize_row_keys(row: dict[str, Any]) -> dict[str, Any]:
    return {str(k).strip().lower(): v for k, v in row.items()}


def _rows_from_polars(file_path: Path) -> list[dict[str, Any]]:
    pl = _polars_module(required=False)
    if pl is None:
        return []
    try:
        if file_path.suffix.lower() == ".csv":
            frame = pl.read_csv(file_path, try_parse_dates=True)
        else:
            try:
                frame = pl.read_json(file_path)
            except Exception:
                frame = pl.read_ndjson(file_path)
    except Exception:
        return []
    if frame is None or int(frame.height) <= 0:
        return []
    return [_normalize_row_keys(dict(row)) for row in frame.iter_rows(named=True)]


def _rows_from_stdlib(file_path: Path) -> list[dict[str, Any]]:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        try:
            with file_path.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                return [_normalize_row_keys(dict(row)) for row in reader if row]
        except Exception:
            return []
    if suffix == ".json":
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if isinstance(payload, list):
            return [_normalize_row_keys(dict(item)) for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("records", "items", "data", "events"):
                val = payload.get(key)
                if isinstance(val, list):
                    return [_normalize_row_keys(dict(item)) for item in val if isinstance(item, dict)]
            return [_normalize_row_keys(payload)]
    return []


def _extract_rows(file_path: Path) -> list[dict[str, Any]]:
    rows = _rows_from_polars(file_path)
    if rows:
        return rows
    return _rows_from_stdlib(file_path)


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if isinstance(value, (int, float)):
        try:
            num = float(value)
            abs_num = abs(num)
            if abs_num > 1e14:
                num /= 1e9
            elif abs_num > 1e11:
                num /= 1e3
            return datetime.fromtimestamp(num, tz=UTC)
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    with_iso = text
    try:
        dt = datetime.fromisoformat(with_iso)
        return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except Exception:
            continue
    return None


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, str) and value.strip() == "":
        return float(default)
    try:
        out = float(value)
        if out != out:  # NaN check
            return float(default)
        return out
    except Exception:
        return float(default)


def _parse_currencies(raw: Any, title: str) -> list[str]:
    text = str(raw or "").replace(";", ",")
    tokens = [tok.strip().upper() for tok in text.split(",") if tok and len(tok.strip()) == 3 and tok.strip().isalpha()]
    if tokens:
        return list(dict.fromkeys(tokens))[:3]
    title_u = str(title or "").upper()
    return [
        ccy
        for ccy in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")
        if ccy in title_u
    ]


# Add project root to sys.path for imports to work
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forex_bot.core.config import settings  # noqa: E402
from forex_bot.data.news.store import NewsDatabase, NewsEvent  # noqa: E402

logger = logging.getLogger(__name__)


def ingest_local_news(data_dir: Path = Path("data/news")):
    """
    Scan data_dir for news files and ingest them into the database.
    Supported formats: .csv, .json
    Expected columns/fields: title, date/published_at, summary/snippet (optional), sentiment (optional)
    """
    if not data_dir.exists():
        logger.warning(f"News data directory {data_dir} does not exist.")
        return

    db_path = Path(settings.system.cache_dir) / "news.sqlite"
    db = NewsDatabase(db_path)

    files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.json"))
    if not files:
        logger.info(f"No news files found in {data_dir}")
        return

    total_inserted = 0

    for file_path in files:
        logger.info(f"Processing {file_path}...")
        try:
            rows = _extract_rows(file_path)
            if not rows:
                continue

            events: list[NewsEvent] = []
            for i, row in enumerate(rows):
                title = str(row.get("title") or row.get("event") or row.get("headline") or "").strip()
                if not title:
                    continue

                ts_raw = row.get("published_at", row.get("date", row.get("time", row.get("datetime"))))
                ts = _parse_timestamp(ts_raw)
                if ts is None or ts.year < 1990:
                    continue

                summary = str(row.get("summary") or row.get("snippet") or row.get("description") or "").strip()
                url = str(row.get("url") or "").strip() or f"file://{file_path.name}/{i}"
                source = str(row.get("source") or "LocalArchive").strip() or "LocalArchive"
                sentiment = _to_float(row.get("sentiment"), 0.0)
                confidence = _to_float(row.get("confidence"), 0.0)
                currencies = _parse_currencies(row.get("currencies", row.get("currency")), title)

                events.append(
                    NewsEvent(
                        title=title,
                        summary=summary,
                        url=url,
                        source=source,
                        published_at=ts,
                        sentiment=sentiment,
                        confidence=confidence,
                        currencies=currencies,
                    )
                )

            if events:
                count = db.insert_events(events)
                total_inserted += count
                logger.info(f"Inserted {count} events from {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")

    logger.info(f"News ingestion complete. Total new events: {total_inserted}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_local_news()

