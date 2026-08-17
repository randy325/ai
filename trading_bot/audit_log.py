"""Structured JSON-lines logging of orders, fills, errors and P&L changes.

One JSON object per line, so the file is greppable, diffable and loadable with
``pandas.read_json(path, lines=True)`` without a custom parser. This is a
compliance trail, not the human-readable console output — the two are
deliberately separate handlers so turning one off never silences the other.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return str(value)


class JSONLinesHandler(logging.Handler):
    """Writes each record as one JSON line: timestamp, level, event, fields."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = {
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "event": getattr(record, "event", record.getMessage()),
            }
            fields = getattr(record, "fields", None)
            if fields:
                payload["fields"] = fields
            if record.exc_info:
                payload["exception"] = self.formatException(record.exc_info)
            self._file.write(json.dumps(payload, default=_json_default) + "\n")
            self._file.flush()
        except Exception:  # noqa: BLE001 - a logging handler must never raise
            self.handleError(record)

    def close(self) -> None:
        try:
            self._file.close()
        finally:
            super().close()


class AuditLogger:
    """Convenience wrapper: one call per event type, structured consistently.

    Every entry point a run needs to be reconstructable from is covered here —
    orders, fills, errors and equity changes — so "did this bot do X" is a grep
    away rather than a question that needs the code re-read.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("trading_bot.audit")
        self._last_equity: float | None = None

    def _emit(self, level: int, event: str, **fields: Any) -> None:
        self.logger.log(level, event, extra={"event": event, "fields": fields})

    def order_submitted(self, order) -> None:
        self._emit(
            logging.INFO, "order.submitted",
            client_order_id=order.client_order_id, symbol=order.symbol,
            side=order.side, quantity=order.quantity, type=order.type,
            limit_price=order.limit_price, reason=order.reason,
        )

    def order_result(self, result) -> None:
        level = logging.INFO if result.filled else logging.WARNING
        self._emit(
            level, f"order.{result.status.value}",
            client_order_id=result.order.client_order_id, symbol=result.order.symbol,
            requested_quantity=result.requested_quantity,
            filled_quantity=result.filled_quantity,
            unfilled_quantity=result.unfilled_quantity, reason=result.reason,
        )

    def fill(self, fill) -> None:
        self._emit(
            logging.INFO, "fill",
            client_order_id=fill.client_order_id, symbol=fill.symbol,
            side=fill.side, quantity=fill.quantity, price=fill.price,
            commission=fill.commission, notional=fill.notional, reason=fill.reason,
        )

    def error(self, message: str, **fields: Any) -> None:
        self._emit(logging.ERROR, "error", message=message, **fields)

    def equity(self, timestamp, equity: float, cash: float, exposure: float = 0.0) -> None:
        """Log an equity point, but only when it actually moved.

        Called once per bar in a live session, this would otherwise write a
        near-duplicate line every few seconds — the signal is in the change.
        """
        if self._last_equity is not None and abs(equity - self._last_equity) < 1e-9:
            return
        delta = None if self._last_equity is None else equity - self._last_equity
        self._emit(
            logging.INFO, "equity",
            timestamp=timestamp, equity=equity, cash=cash, exposure=exposure, change=delta,
        )
        self._last_equity = equity

    def risk_event(self, event: str, **fields: Any) -> None:
        """Halts, pauses, tier changes — anything the risk layer decided on its own."""
        self._emit(logging.WARNING, f"risk.{event}", **fields)

    def kill_switch(self, reason: str, **fields: Any) -> None:
        self._emit(logging.CRITICAL, "kill_switch", reason=reason, **fields)

    def reconciliation(self, problems: list[str], **fields: Any) -> None:
        level = logging.ERROR if problems else logging.INFO
        self._emit(level, "reconciliation", problems=problems, clean=not problems, **fields)


def configure_audit_log(path: str | Path, level: int = logging.INFO) -> AuditLogger:
    """Attach a :class:`JSONLinesHandler` to the audit logger and return it.

    Idempotent per path: calling this twice with the same path does not attach
    a second handler and duplicate every line.
    """
    logger = logging.getLogger("trading_bot.audit")
    logger.setLevel(level)
    resolved = str(Path(path).resolve())
    already_attached = any(
        isinstance(h, JSONLinesHandler) and str(h.path.resolve()) == resolved
        for h in logger.handlers
    )
    if not already_attached:
        handler = JSONLinesHandler(path)
        handler.setLevel(level)
        logger.addHandler(handler)
    logger.propagate = False
    return AuditLogger(logger)
