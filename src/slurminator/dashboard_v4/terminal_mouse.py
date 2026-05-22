"""Terminal mouse reporting helpers for dashboard v4."""

from __future__ import annotations

from typing import Any

ENABLE_MOUSE_REPORTING = "\x1b[?1000h\x1b[?1003h\x1b[?1015h\x1b[?1006h"
DISABLE_MOUSE_REPORTING = "\x1b[?1000l\x1b[?1003l\x1b[?1015l\x1b[?1006l"


def set_terminal_mouse_reporting(app_or_driver: Any, *, enabled: bool) -> bool:
    """Enable or disable terminal mouse reporting for native terminal selection."""
    driver = getattr(app_or_driver, "_driver", app_or_driver)
    if driver is None:
        return False

    method_name = "_enable_mouse_support" if enabled else "_disable_mouse_support"
    method = getattr(driver, method_name, None)
    if callable(method):
        method()
        return True

    write = getattr(driver, "write", None)
    if not callable(write):
        return False
    write(ENABLE_MOUSE_REPORTING if enabled else DISABLE_MOUSE_REPORTING)
    flush = getattr(driver, "flush", None)
    if callable(flush):
        flush()
    return True


__all__ = ["DISABLE_MOUSE_REPORTING", "ENABLE_MOUSE_REPORTING", "set_terminal_mouse_reporting"]
