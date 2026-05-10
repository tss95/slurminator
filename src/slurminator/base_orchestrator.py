"""Experiment-list generation for slurminator."""

from __future__ import annotations

import dataclasses
import logging
import math
import re
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

from slurminator.experiments import ExperimentConfig, ExperimentStatus
from slurminator.experiments.yaml_utils import dump_yaml

if TYPE_CHECKING:
    from slurminator.experiments import MasterExperimentConfig

logger = logging.getLogger("slurminator")


class BaseOrchestrator:
    """Generate experiment records without assigning HPC resources."""

    def __init__(self, master_config: "MasterExperimentConfig"):
        self.master_config = master_config
        self.experiments: list[ExperimentConfig] = []
        self.output_dir = Path("experiment_lists")
        self.output_dir.mkdir(exist_ok=True)
        self.project_names: dict[str, str] = {}

    def _seeded(
        self,
        template: ExperimentConfig,
        dataset_name: str,
        seeds_override: list[int] | None = None,
        num_seeds: int | None = None,
    ) -> list[ExperimentConfig]:
        """Return one experiment copy for each resolved seed."""
        seeds = self._resolve_seeds(dataset_name, seeds_override, num_seeds)

        runs = []
        for seed in seeds:
            metadata = self._apply_common_metadata(template.metadata)
            metadata["seed"] = seed
            exp = dataclasses.replace(template, experiment_id=f"{template.experiment_id}_s{seed}", metadata=metadata)
            runs.append(exp)
        return runs

    def _resolve_seeds(self, dataset_name: str, seeds_override: list[int] | None, num_seeds: int | None) -> list[int]:
        """Resolve the seed list for an experiment."""
        if num_seeds is not None and num_seeds <= 0:
            raise ValueError("num_seeds must be positive for seeded experiment generation.")

        if seeds_override is not None:
            seeds = [int(seed) for seed in seeds_override]
            if not seeds:
                raise ValueError("Custom sweep seeds must contain at least one seed.")
            if num_seeds is not None and len(seeds) != num_seeds:
                raise ValueError(
                    "Custom sweep specifies both seeds and num_seeds with mismatched counts: "
                    f"len(seeds)={len(seeds)} != num_seeds={num_seeds}."
                )
            return seeds

        base_seeds = self.master_config.dataset_seed_overrides.get(dataset_name, self.master_config.seeds)
        if not base_seeds:
            base_seeds = [getattr(self.master_config, "seed", 42)]

        seeds = [int(seed) for seed in base_seeds]
        if num_seeds is None:
            return seeds

        if len(seeds) >= num_seeds:
            return seeds[:num_seeds]

        next_seed = seeds[-1]
        while len(seeds) < num_seeds:
            next_seed += 1
            seeds.append(next_seed)
        return seeds

    def _build_override_str(self, **kv: object) -> str:
        """Format key-value pairs into a semicolon-separated override string."""
        normalised = {k.replace("__", "."): v for k, v in kv.items()}
        return ";".join(f"{k}={v}" for k, v in normalised.items())

    def _merge_override_strings(self, *parts: str | None) -> str:
        """Join already-formatted override fragments with semicolons."""
        cleaned = [p.strip(" ;") for p in parts if p]
        return ";".join(p for p in cleaned if p)

    def _dict_to_override(self, data: dict[str, object]) -> str:
        """Return an override string from an existing dictionary."""
        return self._build_override_str(**data)

    def _apply_common_metadata(self, metadata: object | None) -> dict[str, Any]:
        """Attach orchestrator-wide metadata to one experiment."""
        merged = dict(metadata or {}) if isinstance(metadata, dict) else {}
        if self.master_config.config_profile and "config_profile" not in merged:
            merged["config_profile"] = self.master_config.config_profile
        return merged

    def _get_project_name(self, base_name: str) -> str:
        """Generate or retrieve a timestamped project name for a base name."""
        if base_name not in self.project_names:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.project_names[base_name] = f"{base_name}_{timestamp}"
            logger.info("Generated project name for '%s': %s", base_name, self.project_names[base_name])
        return self.project_names[base_name]

    def _sanitize_id_component(self, text: str) -> str:
        """Make experiment-id safe fragments."""
        return re.sub(r"[^A-Za-z0-9_]+", "_", text)

    def _canonicalize_sweep_key(self, key: str) -> str:
        """Accept dot-notation and common tokenizer shorthand keys."""
        normalised = key.replace("__", ".")
        if "." not in normalised and normalised in {"tokenizer_patch_size", "tokenizer_stride"}:
            return f"model.{normalised}"
        return normalised

    def _resolve_dataset_scoped_value(self, value: Any, dataset_name: str, *, field_name: str) -> Any:
        """Resolve a scalar or per-dataset custom-sweep field."""
        if value is None:
            return None
        if not isinstance(value, dict):
            return value

        for key in (dataset_name, str(dataset_name), "default", "*"):
            if key in value:
                return value[key]
        available = ", ".join(str(k) for k in value.keys())
        raise ValueError(
            f"Custom sweep field '{field_name}' is dataset-scoped but has no value for dataset "
            f"{dataset_name!r}. Available keys: {available or '<none>'}."
        )

    def _infer_resume_checkpoint_epoch(self, checkpoint_path: str) -> int:
        """Return the saved epoch index for ``checkpoint_path``.

        Checkpoint formats are project-specific, so package users that enable
        ``checkpoint_probe`` must override this hook in an adapter subclass.
        """
        raise NotImplementedError(
            "checkpoint_probe requires a project adapter to implement _infer_resume_checkpoint_epoch(). "
            f"Cannot inspect checkpoint: {checkpoint_path}"
        )

    def _custom_sweep_values_equal(self, existing: Any, required: Any) -> bool:
        """Compare typed YAML values with their string override equivalents."""
        if existing == required:
            return True
        if required is None and isinstance(existing, str):
            return existing.strip().lower() in {"none", "null"}
        if isinstance(required, bool) and isinstance(existing, str):
            return existing.strip().lower() == str(required).lower()
        if isinstance(required, int) and isinstance(existing, str):
            try:
                return int(existing.strip()) == required
            except ValueError:
                return False
        if isinstance(required, list) and isinstance(existing, str):
            import ast

            try:
                return ast.literal_eval(existing) == required
            except (SyntaxError, ValueError):
                return False
        return False

    def _apply_checkpoint_probe_overrides(
        self, overrides: dict[str, Any], *, dataset_name: str, resume_from: str, epoch_offset: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Force a resumed custom sweep to run only the first scheduled probe."""
        try:
            offset_i = int(epoch_offset)
        except Exception as exc:
            raise ValueError(
                f"checkpoint_probe_epoch_offset must be integer-compatible, got {epoch_offset!r}."
            ) from exc
        if offset_i <= 0:
            raise ValueError(f"checkpoint_probe_epoch_offset must be > 0, got {offset_i}.")

        checkpoint_epoch = self._infer_resume_checkpoint_epoch(resume_from)
        probe_epoch = checkpoint_epoch + offset_i
        finalized = dict(overrides)
        required = {
            "training_configs.num_epochs": probe_epoch,
            "training_configs.max_train_steps": None,
            "probe_parameters.probe_explicit_epochs": [probe_epoch],
            "probe_parameters.probe_final_epoch": False,
            "probe_parameters.skip_probing": False,
        }

        missing = object()
        for key, value in required.items():
            existing = finalized.get(key, missing)
            if existing is not missing and not self._custom_sweep_values_equal(existing, value):
                raise ValueError(
                    f"checkpoint_probe for dataset {dataset_name!r} requires {key}={value!r}, "
                    f"but the custom sweep already set {key}={existing!r}."
                )
            finalized[key] = value

        metadata = {
            "resume_from": resume_from,
            "checkpoint_probe": True,
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_probe_epoch_offset": offset_i,
            "probe_epoch": probe_epoch,
        }
        return finalized, metadata

    def _format_sweep_tag(self, key: str, value: Any, prefix_map: dict[str, str] | None = None) -> str:
        """Create a compact tag for experiment IDs or run names."""
        prefix_map = prefix_map or {}
        suffix = str(value).replace(".", "_")
        key_tail = key.split(".")[-1]
        prefix = prefix_map.get(key, prefix_map.get(key_tail, None))
        if prefix is None:
            if "patch" in key_tail:
                prefix = "p"
            elif "stride" in key_tail:
                prefix = "st"
            else:
                prefix = key_tail
        return f"{prefix}{suffix}"

    def _resolve_task_type(self, task_type: object | None) -> str:
        """Coerce a task taxonomy value into an opaque task-type string."""
        value = getattr(task_type, "value", task_type)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered.startswith("tasktype."):
                lowered = lowered.split(".", 1)[1].lower()
            if lowered == "supervised":
                return "supervised"
            if lowered == "semi_supervised":
                return "semi_supervised"
            if lowered == "forecasting":
                return "forecasting"
            if lowered in {"anomaly", "anomaly_detection"}:
                return "anomaly_detection"
            if lowered:
                return lowered
        return "self_supervised"

    def generate_all_experiments(self) -> None:
        """Generate experiments from the master config."""
        self.experiments = []
        existing_results = self._load_existing_results()
        self.project_names = {}

        if getattr(self.master_config, "run_custom_sweeps", False):
            self._generate_custom_sweep_experiments(existing_results)

        logger.info("Generated %s total experiments", len(self.experiments))
        skipped_count = sum(1 for exp in self.experiments if exp.status == ExperimentStatus.COMPLETED)
        if skipped_count > 0:
            logger.info("Skipped %s already completed experiments", skipped_count)

    def generate_experiment_file(self) -> Path:
        """Generate a YAML file containing all experiments."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"experiments_{timestamp}.yaml"

        yaml_data = {
            "metadata": {"creation_time": datetime.now().isoformat(), "experiment_count": len(self.experiments)},
            "experiments": [],
        }

        for exp in self.experiments:
            serializable_metadata = {}
            if hasattr(exp, "metadata") and exp.metadata:
                if isinstance(exp.metadata, dict):
                    serializable_metadata = exp.metadata
                elif hasattr(exp.metadata, "__dict__"):
                    meta_dict = asdict(exp.metadata)
                    for key, value in meta_dict.items():
                        if isinstance(value, Enum):
                            serializable_metadata[key] = value.value
                        elif isinstance(value, datetime):
                            serializable_metadata[key] = value.isoformat()
                        else:
                            serializable_metadata[key] = value
                else:
                    serializable_metadata = str(exp.metadata)

            exp_dict = {
                "experiment_id": exp.experiment_id,
                "task_type": exp.task_type.value if isinstance(exp.task_type, Enum) else str(exp.task_type),
                "dataset_name": exp.dataset_name,
                "status": exp.status.value if isinstance(exp.status, Enum) else str(exp.status),
                "hpc_assignment": None,
                "sweep_params": exp.sweep_params,
                "extra_command": exp.extra_command,
                "metadata": serializable_metadata,
            }
            yaml_data["experiments"].append({key: value for key, value in exp_dict.items() if value is not None})

        dump_yaml(yaml_data, output_path)
        logger.info("Generated experiment file: %s", output_path)
        return output_path

    def _generate_custom_sweep_experiments(self, existing_results: dict[str, dict]) -> None:
        """Generate experiments from arbitrary sweep specifications."""
        sweeps = getattr(self.master_config, "custom_sweeps", []) or []
        for sweep_cfg in sweeps:
            sweep_profile = getattr(sweep_cfg, "config_profile", None)
            master_profile = getattr(self.master_config, "config_profile", None)
            if sweep_profile and master_profile and sweep_profile != master_profile:
                logger.info(
                    "[CustomSweep] Overriding master config_profile=%s with sweep config_profile=%s for prefix=%s",
                    master_profile,
                    sweep_profile,
                    getattr(sweep_cfg, "experiment_prefix", "abl"),
                )

            datasets = sweep_cfg.datasets or (
                [sweep_cfg.dataset_name] if getattr(sweep_cfg, "dataset_name", None) else []
            )
            if not datasets:
                raise ValueError("Custom sweep requires at least one dataset (set datasets or dataset_name).")

            def _canonicalize_overrides(override_dict: dict[str, Any]) -> dict[str, Any]:
                return {self._canonicalize_sweep_key(k): v for k, v in (override_dict or {}).items()}

            def _coerce_positive_int(value: Any, *, field_name: str) -> int:
                try:
                    ivalue = int(value)
                except Exception as exc:
                    raise ValueError(
                        f"Custom sweep field '{field_name}' must be an integer-compatible value; got {value!r}."
                    ) from exc
                if ivalue <= 0:
                    raise ValueError(f"Custom sweep field '{field_name}' must be > 0; got {ivalue}.")
                return ivalue

            def _step_budget_epoch_horizon(overrides: dict[str, Any]) -> int | None:
                max_steps = overrides.get("training_configs.max_train_steps")
                pseudo_steps = overrides.get("training_configs.pseudo_epoch_steps")
                if max_steps is None and pseudo_steps is None:
                    return None
                if max_steps is None or pseudo_steps is None:
                    raise ValueError(
                        "Custom sweep step-budget runs require both "
                        "'training_configs.max_train_steps' and 'training_configs.pseudo_epoch_steps'."
                    )
                max_steps_i = _coerce_positive_int(max_steps, field_name="training_configs.max_train_steps")
                pseudo_steps_i = _coerce_positive_int(pseudo_steps, field_name="training_configs.pseudo_epoch_steps")
                return int(math.ceil(max_steps_i / float(pseudo_steps_i)))

            def _resolve_custom_sweep_epochs(overrides: dict[str, Any]) -> int:
                explicit_epochs = overrides.get("training_configs.num_epochs")
                if explicit_epochs is not None:
                    return _coerce_positive_int(explicit_epochs, field_name="training_configs.num_epochs")

                if sweep_cfg.num_epochs is not None:
                    return _coerce_positive_int(sweep_cfg.num_epochs, field_name="custom_sweeps.num_epochs")

                inferred = _step_budget_epoch_horizon(overrides)
                if inferred is not None:
                    return inferred
                return 150

            def _checkpoint_probe_context(case: Any | None, dataset_name: str) -> tuple[bool, str | None, int]:
                case_probe = getattr(case, "checkpoint_probe", None) if case is not None else None
                checkpoint_probe = bool(
                    case_probe if case_probe is not None else getattr(sweep_cfg, "checkpoint_probe", False)
                )

                resume_spec = getattr(case, "resume_from", None) if case is not None else None
                if resume_spec is None:
                    resume_spec = getattr(sweep_cfg, "resume_from", None)
                resume_from = self._resolve_dataset_scoped_value(
                    resume_spec, dataset_name, field_name="custom_sweeps.resume_from"
                )
                if checkpoint_probe and not resume_from:
                    raise ValueError(
                        f"Custom sweep checkpoint_probe=True for dataset {dataset_name!r} requires resume_from."
                    )

                case_offset = getattr(case, "checkpoint_probe_epoch_offset", None) if case is not None else None
                epoch_offset = (
                    case_offset if case_offset is not None else getattr(sweep_cfg, "checkpoint_probe_epoch_offset", 1)
                )
                return checkpoint_probe, str(resume_from) if resume_from is not None else None, epoch_offset

            def _finalize_overrides(
                overrides: dict[str, Any],
                *,
                dataset_name: str,
                checkpoint_probe: bool,
                resume_from: str | None,
                epoch_offset: int,
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                finalized = dict(overrides)
                if checkpoint_probe:
                    if not resume_from:
                        raise ValueError(
                            f"Custom sweep checkpoint_probe=True for dataset {dataset_name!r} requires resume_from."
                        )
                    return self._apply_checkpoint_probe_overrides(
                        finalized, dataset_name=dataset_name, resume_from=resume_from, epoch_offset=epoch_offset
                    )
                finalized["training_configs.num_epochs"] = _resolve_custom_sweep_epochs(finalized)
                metadata = {"resume_from": resume_from} if resume_from else {}
                return finalized, metadata

            base_overrides = _canonicalize_overrides(dict(getattr(sweep_cfg, "base_overrides", {}) or {}))
            sweep_keys = sweep_cfg.sweep_keys or {}
            if sweep_keys and any(len(values) == 0 for values in sweep_keys.values()):
                logger.warning("Skipping custom sweep because one of the sweep lists is empty.")
                continue

            cases = getattr(sweep_cfg, "cases", None) or []
            keys = list(sweep_keys.keys())
            combos = list(product(*sweep_keys.values())) if sweep_keys else [()]
            run_cartesian = bool(sweep_keys) or not cases
            task_type = self._resolve_task_type(getattr(sweep_cfg, "task_type", None))
            exp_prefix = getattr(sweep_cfg, "experiment_prefix", "abl")
            tag_prefix_map = getattr(sweep_cfg, "parameters_prefix", {}) or {}

            for dataset_name in datasets:
                project = getattr(sweep_cfg, "wandb_project", None) or self._get_project_name(
                    f"CustomSweep_{self._sanitize_id_component(dataset_name)}"
                )

                for case in cases:
                    case_tag = self._sanitize_id_component(getattr(case, "name", "case"))
                    overrides = dict(base_overrides)
                    overrides.update(_canonicalize_overrides(getattr(case, "base_overrides", {}) or {}))
                    overrides.update(_canonicalize_overrides(getattr(case, "overrides", {}) or {}))

                    exp_id = f"{exp_prefix}_{self._sanitize_id_component(dataset_name)}_{case_tag}"
                    if exp_id in existing_results or any(e.experiment_id.startswith(exp_id) for e in self.experiments):
                        logger.debug("Skipping duplicate custom sweep experiment %s", exp_id)
                        continue

                    run_name = self._custom_sweep_run_name(sweep_cfg, dataset_name=dataset_name, tag=case_tag)
                    overrides["run_name"] = run_name
                    checkpoint_probe, resume_from, epoch_offset = _checkpoint_probe_context(case, dataset_name)
                    overrides, checkpoint_metadata = _finalize_overrides(
                        overrides,
                        dataset_name=dataset_name,
                        checkpoint_probe=checkpoint_probe,
                        resume_from=resume_from,
                        epoch_offset=epoch_offset,
                    )

                    metadata = self._custom_sweep_metadata(
                        sweep_cfg,
                        project=project,
                        run_name=run_name,
                        checkpoint_metadata=checkpoint_metadata,
                        sweep_profile=sweep_profile,
                    )
                    template = ExperimentConfig(
                        task_type=task_type,
                        dataset_name=dataset_name,
                        experiment_id=exp_id,
                        status=ExperimentStatus.PENDING,
                        metadata=metadata,
                        sweep_params=self._build_override_str(**overrides),
                        extra_command=None,
                    )
                    self.experiments.extend(
                        self._seeded(
                            template,
                            dataset_name,
                            seeds_override=getattr(sweep_cfg, "seeds", None),
                            num_seeds=getattr(sweep_cfg, "num_seeds", None),
                        )
                    )

                if run_cartesian:
                    for values in combos:
                        overrides = dict(base_overrides)
                        tags = []
                        for key, val in zip(keys, values):
                            canonical_key = self._canonicalize_sweep_key(key)
                            overrides[canonical_key] = val
                            tags.append(self._format_sweep_tag(canonical_key, val, tag_prefix_map))

                        tag_str = "_".join(tags) if tags else "base"
                        exp_id = f"{exp_prefix}_{self._sanitize_id_component(dataset_name)}_{tag_str}"
                        if exp_id in existing_results or any(
                            e.experiment_id.startswith(exp_id) for e in self.experiments
                        ):
                            logger.debug("Skipping duplicate custom sweep experiment %s", exp_id)
                            continue

                        run_name = self._custom_sweep_run_name(sweep_cfg, dataset_name=dataset_name, tag=tag_str)
                        overrides["run_name"] = run_name
                        checkpoint_probe, resume_from, epoch_offset = _checkpoint_probe_context(None, dataset_name)
                        overrides, checkpoint_metadata = _finalize_overrides(
                            overrides,
                            dataset_name=dataset_name,
                            checkpoint_probe=checkpoint_probe,
                            resume_from=resume_from,
                            epoch_offset=epoch_offset,
                        )

                        metadata = self._custom_sweep_metadata(
                            sweep_cfg,
                            project=project,
                            run_name=run_name,
                            checkpoint_metadata=checkpoint_metadata,
                            sweep_profile=sweep_profile,
                        )
                        template = ExperimentConfig(
                            task_type=task_type,
                            dataset_name=dataset_name,
                            experiment_id=exp_id,
                            status=ExperimentStatus.PENDING,
                            metadata=metadata,
                            sweep_params=self._build_override_str(**overrides),
                            extra_command=None,
                        )
                        self.experiments.extend(
                            self._seeded(
                                template,
                                dataset_name,
                                seeds_override=getattr(sweep_cfg, "seeds", None),
                                num_seeds=getattr(sweep_cfg, "num_seeds", None),
                            )
                        )

    def _custom_sweep_run_name(self, sweep_cfg: object, *, dataset_name: str, tag: str) -> str:
        run_name_prefix = getattr(sweep_cfg, "run_name_prefix", None) or getattr(sweep_cfg, "experiment_prefix", "abl")
        run_name_suffix = getattr(sweep_cfg, "run_name_suffix", None)
        run_name_parts = [run_name_prefix, dataset_name]
        if tag:
            run_name_parts.append(tag)
        if run_name_suffix:
            run_name_parts.append(run_name_suffix)
        return "_".join(run_name_parts)

    def _custom_sweep_metadata(
        self,
        sweep_cfg: object,
        *,
        project: str,
        run_name: str,
        checkpoint_metadata: dict[str, Any],
        sweep_profile: str | None,
    ) -> dict[str, Any]:
        metadata = self._apply_common_metadata(
            {"wandb_project": project, "ablation_type": "custom_sweep", "use_full_labels": False, "run_name": run_name}
        )
        metadata.update(checkpoint_metadata)
        if sweep_profile:
            metadata["config_profile"] = sweep_profile
        return metadata

    def _load_existing_results(self) -> dict[str, dict]:
        """Return existing result records used to skip completed runs."""
        return {}


__all__ = ["BaseOrchestrator"]
