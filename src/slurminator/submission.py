"""Slurm submission helpers for the Slurminator orchestrator."""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from typing import Any

from slurminator.experiments import ExperimentStatus
from slurminator.config import HPCType
from slurminator.experiment_policy import resolve_pinned_hpc, resolve_resource_overrides, resolve_sbatch_export_vars
from slurminator.git_provenance import capture_provenance
from slurminator.timeout_policy import SubmissionResources, apply_timeout_retry_to_resources

logger = logging.getLogger("slurminator")

CommandBuilder = Callable[[dict[str, Any], int, HPCType], str]
SaveYaml = Callable[[dict[str, Any]], None]
RecordSubmission = Callable[[Mapping[str, Any], str | None], object]
ReplaceExperiment = Callable[[list[dict[str, Any]], dict[str, Any]], list[dict[str, Any]]]
SubmitExperiment = Callable[[dict[str, Any], HPCType], str | None]
IsLocalHPC = Callable[[HPCType], bool]


@dataclass
class SubmissionContext:
    """Dependencies needed to submit one experiment without a full orchestrator instance."""

    experiment_file: Path
    concurrency_limits: Mapping[HPCType, int]
    hpc_configs: Mapping[HPCType, Any]
    connection_manager: Any
    build_commands_line: CommandBuilder
    is_local_hpc: IsLocalHPC
    max_gpus_per_job: int | None = None
    time_hours_override: int | None = None
    memory_gb_override: int | None = None
    partition_overrides: Mapping[HPCType, str] = field(default_factory=dict)
    prepared_repositories: set[HPCType] | None = None


def maybe_submit(
    exp: dict[str, Any],
    concurrency_used: dict[HPCType, int],
    data: dict[str, Any],
    *,
    concurrency_limits: Mapping[HPCType, int],
    hpc_configs: Mapping[HPCType, Any],
    submit_experiment: SubmitExperiment,
    replace_exp_in_list: ReplaceExperiment,
    save_yaml: SaveYaml,
    record_submission: RecordSubmission | None = None,
) -> bool:
    """Submit a pending/partial experiment and return whether submission succeeded."""
    status = exp.get("status")
    if status not in [ExperimentStatus.PENDING, ExperimentStatus.PARTIAL]:
        return False

    forced_hpc = resolve_pinned_hpc(exp, hpc_configs)
    if forced_hpc and exp.get("hpc_assignment") != forced_hpc:
        exp["hpc_assignment"] = forced_hpc

    hpc_type = exp.get("hpc_assignment")
    if not hpc_type:
        chosen = _choose_hpc(concurrency_used, concurrency_limits)
        if not chosen:
            logger.debug("%s: no HPC free => remain PENDING", exp["experiment_id"])
            return False
        exp["hpc_assignment"] = chosen
        hpc_type = chosen

    limit = concurrency_limits.get(hpc_type, 0)
    if limit <= 0:
        logger.warning("%s: assigned to %s with limit 0 – skipping submission.", exp["experiment_id"], hpc_type)
        exp["hpc_assignment"] = None
        return False

    used = concurrency_used.get(hpc_type, 0)
    if used >= limit:
        logger.debug("HPC %s concurrency limit => %s/%s", hpc_type, used, limit)
        return False

    logger.info("Submitting %s => HPC=%s, usage=%s/%s", exp["experiment_id"], hpc_type, used, limit)
    previous_job_id = _optional_job_id(exp.get("job_id"))
    exp["git_sha_at_submission"] = capture_provenance()
    job_id = submit_experiment(exp, hpc_type)
    if not job_id:
        return False

    logger.info("Got job_id=%s => %s => QUEUED", job_id, exp["experiment_id"])
    exp["status"] = ExperimentStatus.QUEUED
    exp["job_id"] = job_id
    exp["queued_timestamp"] = time.time()
    data["experiments"] = replace_exp_in_list(data["experiments"], exp)
    if record_submission is None:
        save_yaml(data)
    else:
        record_submission(exp, previous_job_id)
    concurrency_used[hpc_type] = used + 1
    return True


def submit_experiment_universal(exp: dict[str, Any], hpc_type: HPCType, context: SubmissionContext) -> str | None:
    """Submit one experiment through ``universal_job.sh`` and return the Slurm job id."""
    if context.concurrency_limits.get(hpc_type, 0) <= 0:
        logger.warning("Skipping submission to %s as concurrency limit is 0", hpc_type)
        return None

    cluster_config = context.hpc_configs.get(hpc_type)
    if not cluster_config:
        logger.error("No HPC config for %s; cannot submit job", hpc_type)
        return None

    resources = resolve_submission_resources(exp, hpc_type, context)
    partition = context.partition_overrides.get(hpc_type) or cluster_config.partition.value
    if hpc_type in context.partition_overrides:
        logger.info(
            "Using partition override from CLI: %s (instead of config default %s)",
            partition,
            cluster_config.partition.value,
        )

    commands_line = context.build_commands_line(exp, resources.gpu_count, hpc_type)
    exp_output_dir = _experiment_output_dir(exp, context.experiment_file, cluster_config.save_path)
    if not context.is_local_hpc(hpc_type):
        context.connection_manager.run_command(hpc_type, f"mkdir -p {exp_output_dir}")
    else:
        exp_output_dir.mkdir(parents=True, exist_ok=True)

    job_name = exp.get("experiment_id", "UnnamedExp")
    sbatch_command = build_sbatch_command(
        exp=exp,
        hpc_type=hpc_type,
        cluster_config=cluster_config,
        resources=resources,
        partition=partition,
        job_name=str(job_name),
        exp_output_dir=exp_output_dir,
        commands_line=commands_line,
    )
    logger.debug("sbatch command: %s", sbatch_command)

    if context.prepared_repositories is None or hpc_type not in context.prepared_repositories:
        _prepare_repo_for_submission(hpc_type, cluster_config, context.connection_manager)
        if context.prepared_repositories is not None:
            context.prepared_repositories.add(hpc_type)
    submission_runner = getattr(context.connection_manager, "run_submission_command", None)
    if callable(submission_runner):
        out, err = submission_runner(hpc_type, sbatch_command)
    else:
        # Compatibility path for injected connection-manager implementations.
        # The built-in manager always uses ``run_submission_command`` so an
        # ambiguous transport failure cannot re-execute sbatch.
        out, err = context.connection_manager.run_command(hpc_type, sbatch_command, prefer_remote=True)
    job_id = _parse_sbatch_job_id(out)

    if job_id is not None:
        if err.strip():
            logger.warning("sbatch warning HPC=%s, exp=%s: %s", hpc_type, job_name, err.strip())
        exp["output_dir"] = str(exp_output_dir)
        exp["save_path"] = cluster_config.save_path
        exp["requested_time_hours"] = int(resources.time_hours)
        exp["requested_ram_gb"] = resources.requested_ram_gb
        exp["requested_gpu_count"] = int(resources.gpu_count)
        logger.info("Parsed job_id=%s from sbatch output for %s", job_id, job_name)
        return job_id

    if err.strip():
        logger.error("sbatch error HPC=%s, exp=%s: %s", hpc_type, job_name, err.strip())
    logger.error("Could not parse job_id from sbatch output: %r", out.strip())
    return None


def resolve_submission_resources(
    exp: dict[str, Any], hpc_type: HPCType, context: SubmissionContext
) -> SubmissionResources:
    """Return resources for one submission without mutating ``exp``."""
    cluster_config = context.hpc_configs[hpc_type]
    dataset = exp.get("dataset_name")
    resource_overrides = resolve_resource_overrides(exp, hpc_type=hpc_type, cluster_configs=context.hpc_configs)

    if context.max_gpus_per_job is not None:
        gpu_count = context.max_gpus_per_job
        logger.info(
            "Using GPU count %s from CLI argument (--max_gpus_per_job) for %s", gpu_count, exp.get("experiment_id")
        )
    elif "gpu_count" in resource_overrides:
        gpu_count = resource_overrides["gpu_count"]
        logger.info("Using GPU count %s from dataset override for %s", gpu_count, dataset)
    else:
        gpu_count = cluster_config.gpu_count
        logger.info("Using default GPU count %s from HPC config for %s", gpu_count, exp.get("experiment_id"))

    if cluster_config.request_gpu_pair and cluster_config.gpu_type.lower() == "mi250" and gpu_count == 1:
        if context.max_gpus_per_job != 1:
            gpu_count = 2
            logger.info(
                "%s: Upgraded to 2 GPUs on LUMI so that both GPU halves of the MI250 card are allocated "
                "(peak XGMI bandwidth).",
                exp["experiment_id"],
            )

    base_resources = SubmissionResources(
        time_hours=int(resource_overrides.get("time_hours", cluster_config.base_time_hours)),
        memory_gb=int(resource_overrides.get("mem_gb", cluster_config.base_memory_gb)),
        cpus=int(cluster_config.cpus_per_task),
        gpu_count=int(gpu_count),
        mem_per_gpu_gb=resource_overrides.get("mem_per_gpu_gb", cluster_config.mem_per_gpu_gb),
    )
    if resource_overrides:
        logger.info(
            "Resource override for %s: gpus=%s, time=%sh, mem=%sG",
            dataset,
            base_resources.gpu_count,
            base_resources.time_hours,
            base_resources.memory_gb,
        )

    resources = apply_timeout_retry_to_resources(
        exp,
        base_resources,
        global_time_hours_override=context.time_hours_override,
        global_memory_gb_override=context.memory_gb_override,
    )
    if context.time_hours_override is not None:
        logger.info(
            "Using global time override %sh from CLI (--job-time-hours) for %s",
            context.time_hours_override,
            exp.get("experiment_id"),
        )
    if exp.get("time_hours_override") is not None and resources.time_hours != base_resources.time_hours:
        logger.info("Using per-experiment time override %sh for %s", resources.time_hours, exp.get("experiment_id"))
    if context.memory_gb_override is not None:
        logger.info(
            "Using global memory override %sG from CLI (--job-ram-gb) for %s",
            context.memory_gb_override,
            exp.get("experiment_id"),
        )

    return resources


def build_sbatch_command(
    *,
    exp: dict[str, Any],
    hpc_type: HPCType,
    cluster_config: Any,
    resources: SubmissionResources,
    partition: str,
    job_name: str,
    exp_output_dir: Path,
    commands_line: str,
) -> str:
    """Build the sbatch command string for one experiment."""
    sbatch = [
        "sbatch",
        "--parsable",
        f"--partition={partition}",
        f"--account={cluster_config.account}",
        f"--job-name={job_name}",
        f"--time={resources.time_hours}:00:00",
        f"--output={exp_output_dir}/slurm-%j.out",
        f"--error={exp_output_dir}/slurm-%j.err",
    ]

    if cluster_config.cpus_per_gpu is not None:
        sbatch.append(f"--cpus-per-gpu={cluster_config.cpus_per_gpu}")
    else:
        sbatch.append(f"--cpus-per-task={resources.cpus}")

    if resources.mem_per_gpu_gb is not None:
        sbatch.append(f"--mem-per-gpu={resources.mem_per_gpu_gb}G")
    else:
        sbatch.append(f"--mem={resources.memory_gb}G")

    exclude_nodes = _exclude_nodes(hpc_type, cluster_config)
    if exclude_nodes:
        sbatch.append(f"--exclude={','.join(exclude_nodes)}")

    export_vars = dict(resolve_sbatch_export_vars(cluster_config))
    export_vars["SLURMINATOR_NPROC_PER_NODE"] = str(resources.gpu_count)
    if export_vars:
        env_block = ",".join(f"{key}={value}" for key, value in export_vars.items())
        sbatch = ["sbatch", f"--export=ALL,{env_block}"] + sbatch[1:]
        sbatch = [cmd for cmd in sbatch if cmd != "--export=ALL"]

    if resources.gpu_count > 0:
        gres_name = getattr(cluster_config, "gpu_gres_name", None)

        if cluster_config.gpu_type.lower() in ["mi250", "amd"]:
            sbatch.append(f"--gres=gpu:{cluster_config.gpu_type}:{resources.gpu_count}")
        elif gres_name:
            sbatch.append(f"--gres=gpu:{gres_name}:{resources.gpu_count}")
        else:
            sbatch.append(f"--gpus={cluster_config.gpu_type}:{resources.gpu_count}")

    gpu_bind_pattern = getattr(cluster_config, "gpu_bind_pattern", None)
    if gpu_bind_pattern:
        sbatch.append(f"--gpu-bind={gpu_bind_pattern}")
        logger.info("Applying gpu-bind pattern: %s", gpu_bind_pattern)

    if not any(item.startswith("--export=") for item in sbatch if isinstance(item, str)):
        sbatch.append("--export=ALL")

    universal_script = f"{cluster_config.repo_path}/universal_job.sh"
    sbatch.extend(
        [
            universal_script,
            (
                f"--env_script {cluster_config.repo_path}/{cluster_config.environment_setup}"
                if cluster_config.environment_setup
                else ""
            ),
            f"--repo_path {cluster_config.repo_path}",
            f"--save_path {cluster_config.save_path}",
            "--commands",
            quote(commands_line),
            f"--exp_id {job_name}",
            f"--outdir {exp_output_dir}",
        ]
    )
    return " ".join(item for item in sbatch if item)


def _parse_sbatch_job_id(output: str) -> str | None:
    """Return a Slurm job ID from parsable or legacy ``sbatch`` output."""
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        parsable_job_id = stripped.split(";", maxsplit=1)[0]
        if re.fullmatch(r"[0-9]+", parsable_job_id):
            return parsable_job_id

        legacy_match = re.fullmatch(r"Submitted batch job ([0-9]+)", stripped)
        if legacy_match is not None:
            return legacy_match.group(1)
    return None


def _optional_job_id(value: object) -> str | None:
    """Normalize an existing job ID before a retry replaces it."""
    if value is None:
        return None
    job_id = str(value).strip()
    return job_id or None


def _choose_hpc(concurrency_used: Mapping[HPCType, int], concurrency_limits: Mapping[HPCType, int]) -> HPCType | None:
    """Return the enabled HPC with the most free slots."""
    chosen = None
    best_free = -1
    for hpc_type, usage in concurrency_used.items():
        limit = concurrency_limits.get(hpc_type, 0)
        if limit <= 0:
            continue
        free_slots = limit - usage
        if free_slots > best_free and free_slots > 0:
            best_free = free_slots
            chosen = hpc_type
    return chosen


def _experiment_output_dir(exp: Mapping[str, Any], experiment_file: Path, save_path: str) -> Path:
    """Return output directory for one submitted experiment."""
    base_folder_name = experiment_file.stem
    sweep_id = exp.get("sweep_id")
    if sweep_id:
        base_folder_name = f"{base_folder_name}_{sweep_id}"
    return Path(save_path) / "experiment_lists" / "outputs" / base_folder_name / str(exp["experiment_id"])


def _exclude_nodes(hpc_type: HPCType, cluster_config: Any) -> list[str]:
    """Return deduplicated exclude-node list."""
    exclude_nodes: list[str] = []
    if cluster_config.exclude_nodes:
        exclude_nodes.extend(cluster_config.exclude_nodes)

    seen: set[str] = set()
    return [node for node in exclude_nodes if not (node in seen or seen.add(node))]


def _prepare_repo_for_submission(hpc_type: HPCType, cluster_config: Any, connection_manager: Any) -> None:
    """Prepare repository/scripts on the submission host before ``sbatch``."""
    if not cluster_config.repo_path:
        return
    try:
        check_cmd = f"bash -lc 'cd {cluster_config.repo_path} 2>/dev/null && [ -d .git ] && echo git || echo nogit'"
        out, _ = connection_manager.run_command(hpc_type, check_cmd, prefer_remote=True)
        mode = out.strip()

        if mode == "git" and not os.environ.get("SLURMINATOR_SKIP_GIT_PULL"):
            git_cmd = (
                "source ~/.bashrc >/dev/null 2>&1 || true; "
                f"cd {cluster_config.repo_path} && git pull --ff-only || true"
            )
            logger.info("Pre-submit git pull in %s on %s", cluster_config.repo_path, hpc_type)
            connection_manager.run_command(hpc_type, git_cmd, prefer_remote=True)
        elif mode == "nogit":
            _push_critical_scripts(hpc_type, cluster_config, connection_manager)
    except Exception as exc:
        logger.warning("Pre-submit repo prep failed on %s: %s", hpc_type, exc)


def _push_critical_scripts(hpc_type: HPCType, cluster_config: Any, connection_manager: Any) -> None:
    """Upload wrapper scripts when the remote repo path is not a git checkout."""
    try:
        root = Path(__file__).resolve().parents[2]
        files = [
            (str(root / "universal_job.sh"), f"{cluster_config.repo_path}/universal_job.sh"),
            (str(root / "step_0.sh"), f"{cluster_config.repo_path}/step_0.sh"),
            (str(root / "multi_gpu.sh"), f"{cluster_config.repo_path}/multi_gpu.sh"),
        ]
        logger.info("Remote path %s is not a git repo – pushing critical scripts", cluster_config.repo_path)
        for local_path, remote_path in files:
            try:
                if os.path.exists(local_path):
                    connection_manager.upload_file(hpc_type, local_path, remote_path)
            except Exception as exc:
                logger.warning("Upload failed for %s -> %s: %s", local_path, remote_path, exc)
        connection_manager.run_command(
            hpc_type,
            f"chmod +x {cluster_config.repo_path}/universal_job.sh {cluster_config.repo_path}/multi_gpu.sh 2>/dev/null || true",
            prefer_remote=True,
        )
    except Exception as exc:
        logger.warning("Remote script sync skipped/failed: %s", exc)


__all__ = [
    "SubmissionContext",
    "build_sbatch_command",
    "maybe_submit",
    "resolve_submission_resources",
    "submit_experiment_universal",
]
