"""Clipboard helpers for Textual dashboard screens."""

from __future__ import annotations

import base64
import os
from typing import Any


def copy_text_to_clipboard(app: Any, text: str) -> None:
    """Copy text through Textual and tmux OSC 52 passthrough when available."""
    app.copy_to_clipboard(text)
    write_tmux_clipboard_passthrough(getattr(app, "_driver", None), text)


def write_tmux_clipboard_passthrough(driver: Any, text: str) -> bool:
    """Write an OSC 52 clipboard sequence wrapped for tmux passthrough."""
    if not os.environ.get("TMUX") or driver is None:
        return False
    write = getattr(driver, "write", None)
    if not callable(write):
        return False
    write(tmux_clipboard_passthrough_sequence(text))
    return True


def tmux_clipboard_passthrough_sequence(text: str) -> str:
    """Return an OSC 52 clipboard sequence wrapped in tmux DCS passthrough."""
    base64_text = base64.b64encode(text.encode("utf-8")).decode("utf-8")
    return f"\x1bPtmux;\x1b\x1b]52;c;{base64_text}\a\x1b\\"


__all__ = ["copy_text_to_clipboard", "tmux_clipboard_passthrough_sequence", "write_tmux_clipboard_passthrough"]
