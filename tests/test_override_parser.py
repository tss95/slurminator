import pytest

from slurminator.cli.override_parser import parse_override_list

pytestmark = pytest.mark.unit


def test_parse_override_list_preserves_list_values_from_single_string() -> None:
    parsed = parse_override_list(
        "probe_parameters.probe_explicit_epochs=[-1, 1, 2, 5, 15];seed=42;probe_parameters.probe_interval=5"
    )
    assert parsed["probe_parameters.probe_explicit_epochs"] == [-1, 1, 2, 5, 15]
    assert parsed["seed"] == 42
    assert parsed["probe_parameters.probe_interval"] == 5


def test_parse_override_list_preserves_commas_inside_quoted_value() -> None:
    parsed = parse_override_list('run_name="sb,stead,nontriviality";seed=7')
    assert parsed["run_name"] == "sb,stead,nontriviality"
    assert parsed["seed"] == 7


def test_parse_override_list_supports_space_separated_overrides() -> None:
    parsed = parse_override_list('seed=11 run_name="ppick non triviality" training_mode=self_supervised')
    assert parsed["seed"] == 11
    assert parsed["run_name"] == "ppick non triviality"
    assert parsed["training_mode"] == "self_supervised"


def test_parse_override_list_preserves_none_string_enum_values() -> None:
    parsed = parse_override_list(
        "contrast_parameters.hgcl.negative_filter_mode=none;"
        "probe_parameters.tuning_split=none;"
        "ppick_params.feature_control=none"
    )

    assert parsed["contrast_parameters.hgcl.negative_filter_mode"] == "none"
    assert parsed["probe_parameters.tuning_split"] == "none"
    assert parsed["ppick_params.feature_control"] == "none"


def test_parse_override_list_coerces_explicit_null_spellings() -> None:
    parsed = parse_override_list(
        "contrast_parameters.hgcl.positive_radius_tokens=null;"
        "contrast_parameters.hgcl.positive_radius_windows=~;"
        "paths.checkpoint_path=None"
    )

    assert parsed["contrast_parameters.hgcl.positive_radius_tokens"] is None
    assert parsed["contrast_parameters.hgcl.positive_radius_windows"] is None
    assert parsed["paths.checkpoint_path"] is None


def test_parse_override_list_rejects_missing_equals() -> None:
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_override_list("seed=1 broken")
