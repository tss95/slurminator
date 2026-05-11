"""Logging helpers for Slurminator CLIs and adopters.

This module copies the clickable-path logging pattern used by PMT, but keeps the
defaults package-generic. PMT intentionally keeps its own logger module; this
copy is for Slurminator's ``slurminator`` logger and external adopters that want
the same console format.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

try:
    import colorlog  # type: ignore

    _HAS_COLORLOG = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    colorlog = None
    _HAS_COLORLOG = False


LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}

DEFAULT_LOG_COLORS = {
    "DEBUG": "blue",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "magenta",
}


class ClickablePathHandler(logging.StreamHandler):
    """Stream handler that adds ``record.clickable_file_line`` as ``path:line``."""

    def __init__(self, project_root: str | Path, stream: Any | None = None) -> None:
        super().__init__(stream=stream)
        self.project_root = Path(project_root).resolve()

    def emit(self, record: logging.LogRecord) -> None:
        """Attach a path relative to ``project_root`` before formatting."""
        try:
            rel_path = os.path.relpath(record.pathname, start=self.project_root)
        except ValueError:
            rel_path = record.pathname
        record.clickable_file_line = f"{rel_path}:{record.lineno}"
        super().emit(record)


def parse_log_level(level_name: str | None, *, default: int = logging.INFO) -> int:
    """Return a logging level from a name, falling back to ``default``."""
    if not level_name:
        return default
    return LOG_LEVELS.get(level_name.upper(), default)


def resolve_log_level(*, env_var: str = "SLURMINATOR_LOG_LEVEL", default: str = "INFO") -> int:
    """Resolve a log level from an environment variable, then ``LOG_LEVEL``."""
    return parse_log_level(os.getenv(env_var, os.getenv("LOG_LEVEL", default)), default=parse_log_level(default))


def build_formatter(logger_label: str = "Slurminator") -> logging.Formatter:
    """Build Slurminator's aligned console formatter."""
    base_format = (
        f"{logger_label} - %(clickable_file_line)-40s "
        "- %(funcName)-20s "
        "- %(levelname)-5s: "
        "%(message)s"
    )
    if _HAS_COLORLOG:
        return colorlog.ColoredFormatter(  # type: ignore[union-attr]
            "%(log_color)s" + base_format,
            log_colors=DEFAULT_LOG_COLORS,
        )
    return logging.Formatter(base_format)


def build_logging_config(
    *,
    logger_name: str = "slurminator",
    logger_label: str = "Slurminator",
    level: str = "INFO",
) -> dict[str, Any]:
    """Return a ``dictConfig``-compatible logging config.

    ``setup_clickable_logger`` or ``configure_logging`` should still be called
    afterward when clickable paths should be relative to a project root.
    """
    fmt = (
        f"{logger_label} - %(clickable_file_line)-40s "
        "- %(funcName)-20s "
        "- %(levelname)-5s: "
        "%(message)s"
    )
    formatter: dict[str, Any]
    if _HAS_COLORLOG:
        formatter = {
            "()": colorlog.ColoredFormatter,  # type: ignore[union-attr]
            "format": "%(log_color)s" + fmt,
            "log_colors": DEFAULT_LOG_COLORS,
        }
    else:
        formatter = {"format": fmt}
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"customFormatter": formatter},
        "handlers": {
            "consoleHandler": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "customFormatter",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {logger_name: {"level": level, "handlers": ["consoleHandler"], "propagate": False}},
        "root": {"level": "NOTSET", "handlers": []},
    }


LOGGING_CONFIG = build_logging_config()


def setup_clickable_logger(
    project_root: str | Path,
    *,
    logger_name: str = "slurminator",
    logger_label: str = "Slurminator",
    level: int | None = None,
    env_var: str = "SLURMINATOR_LOG_LEVEL",
    stream: Any | None = None,
    force: bool = True,
) -> logging.Logger:
    """Install a clickable console handler on ``logger_name``.

    The handler emits ``path:line`` relative to ``project_root``. By default it
    replaces existing stream handlers on that logger to avoid duplicate console
    lines when the CLI is invoked repeatedly in the same Python process.
    """
    logger = logging.getLogger(logger_name)
    level_value = resolve_log_level(env_var=env_var) if level is None else level

    if force:
        for handler in list(logger.handlers):
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)
                handler.close()

    if not force and any(isinstance(handler, ClickablePathHandler) for handler in logger.handlers):
        logger.setLevel(level_value)
        for handler in logger.handlers:
            if isinstance(handler, ClickablePathHandler):
                handler.setLevel(level_value)
        return logger

    clickable_handler = ClickablePathHandler(project_root, stream=sys.stdout if stream is None else stream)
    clickable_handler.setFormatter(build_formatter(logger_label))
    clickable_handler.setLevel(level_value)
    logger.setLevel(level_value)
    logger.propagate = False
    logger.addHandler(clickable_handler)
    return logger


def configure_logging(
    project_root: str | Path = ".",
    *,
    logger_name: str = "slurminator",
    logger_label: str = "Slurminator",
    force: bool = False,
) -> logging.Logger:
    """Configure Slurminator console logging if it is not already configured."""
    return setup_clickable_logger(
        project_root,
        logger_name=logger_name,
        logger_label=logger_label,
        force=force,
    )


__all__ = [
    "ClickablePathHandler",
    "LOGGING_CONFIG",
    "build_formatter",
    "build_logging_config",
    "configure_logging",
    "parse_log_level",
    "resolve_log_level",
    "setup_clickable_logger",
]
