from io import StringIO
import logging

import pytest

from slurminator.logging_config import ClickablePathHandler, parse_log_level, setup_clickable_logger

pytestmark = pytest.mark.unit


def test_clickable_path_handler_adds_relative_path(tmp_path) -> None:
    stream = StringIO()
    handler = ClickablePathHandler(tmp_path, stream=stream)
    handler.setFormatter(logging.Formatter("%(clickable_file_line)s - %(message)s"))

    source = tmp_path / "pkg" / "module.py"
    record = logging.LogRecord(
        name="slurminator-test", level=logging.INFO, pathname=str(source), lineno=7, msg="hello", args=(), exc_info=None
    )

    handler.handle(record)

    assert "pkg/module.py:7 - hello" in stream.getvalue()


def test_setup_clickable_logger_installs_single_console_handler(tmp_path) -> None:
    logger_name = "slurminator-test-logger"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()

    stream = StringIO()
    setup_clickable_logger(
        tmp_path, logger_name=logger_name, logger_label="Test", level=logging.INFO, stream=stream, force=True
    )

    logger.info("ready")

    output = stream.getvalue()
    assert "Test -" in output
    assert "ready" in output
    assert len(logger.handlers) == 1

    logger.handlers.clear()


def test_parse_log_level_falls_back_for_unknown_value() -> None:
    assert parse_log_level("DEBUG") == logging.DEBUG
    assert parse_log_level("not-a-level", default=logging.WARNING) == logging.WARNING
