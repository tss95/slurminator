import pytest

from slurminator.display_helpers import extract_display_metrics

pytestmark = pytest.mark.unit


def test_extract_display_metrics_uses_shortform_metadata() -> None:
    exp = {
        "all_metrics": {"val/acc": 0.91, "loss": 0.2},
        "display_metric_info": {"val/acc": {"shortform": "acc"}, "val/loss": {"shortform": "loss"}},
    }

    assert extract_display_metrics(exp) == {"acc": 0.91, "loss": 0.2}
