import json
import logging
import os
from datetime import datetime
from typing import Any

from app.core.consts import TRACE_ENABLED, TRACE_LOG_PATH

_trace_logger = logging.getLogger("app.trace")
_trace_logger.addHandler(logging.NullHandler())
_trace_logger.propagate = False

if TRACE_ENABLED:
    try:
        os.makedirs(os.path.dirname(TRACE_LOG_PATH), exist_ok=True)
    except Exception:
        pass

    fh = logging.FileHandler(TRACE_LOG_PATH)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    _trace_logger.addHandler(fh)
    _trace_logger.setLevel(logging.INFO)


def _format_event(event: str, **kwargs: Any) -> str:
    payload: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event": event,
    }
    payload.update(kwargs)
    return json.dumps(payload, ensure_ascii=False)


def log_trace(event: str, **kwargs: Any) -> None:
    """
    Log a single JSON object (one-line) with fields:
      - timestamp (UTC ISO)
      - event (name)
      - any additional kwargs passed
    Silent on any internal errors to avoid affecting agent flow.
    """
    try:
        _trace_logger.info(_format_event(event, **kwargs))
    except Exception:
        # Avoid raising from tracing
        return