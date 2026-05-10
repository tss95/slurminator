import logging

import pytest

from slurminator.experiments import ExperimentStatus
from slurminator.orchestrator_ui import print_overview

pytestmark = pytest.mark.unit


def test_print_overview_basic(caplog: pytest.LogCaptureFixture) -> None:
    exps = [
        {"status": ExperimentStatus.COMPLETED},
        {"status": ExperimentStatus.RUNNING},
        {"status": ExperimentStatus.FAILED},
    ]

    logger = logging.getLogger("test-slurminator-ui")
    with caplog.at_level("INFO", logger="test-slurminator-ui"):
        print_overview(exps, overview_logger=logger)

    assert "Status breakdown" in caplog.text
    assert "COMPLETED=1" in caplog.text
    assert "RUNNING=1" in caplog.text
    assert "FAILED=1" in caplog.text
