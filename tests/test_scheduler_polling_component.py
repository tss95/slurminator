import pytest

from slurminator.experiments import ExperimentStatus
from slurminator.config import HPCType
from slurminator.scheduler_polling import expand_short, map_state, poll_hpc, update_scheduler_statuses

pytestmark = pytest.mark.unit


class _PollConnection:
    def __init__(self, *, squeue_out: str = "", sacct_out: str = "") -> None:
        self.squeue_out = squeue_out
        self.sacct_out = sacct_out
        self.commands: list[tuple[HPCType, str, bool]] = []

    def run_command(self, hpc_type, cmd, prefer_remote=False):  # noqa: ANN001
        self.commands.append((hpc_type, cmd, prefer_remote))
        if cmd.startswith("squeue"):
            return (self.squeue_out, "")
        if cmd.startswith("sacct"):
            return (self.sacct_out, "")
        return ("", "")


def test_poll_hpc_uses_squeue_and_sacct_without_orchestrator() -> None:
    connection = _PollConnection(
        squeue_out="451101 R\n",
        sacct_out=("451102 CANCELLED+\n" "451102.batch CANCELLED\n" "451102.extern COMPLETED\n" "451103 COMPLETED\n"),
    )

    states = poll_hpc(connection, HPCType.OLIVIA, ["451101", "451102", "451103"])

    assert states == {"451101": "RUNNING", "451102": "CANCELLED", "451103": "COMPLETED"}
    assert connection.commands == [
        (HPCType.OLIVIA, "squeue -h -o '%i %t' -j 451101,451102,451103", True),
        (HPCType.OLIVIA, "sacct -n -o 'JobID,State' -j 451102,451103", True),
    ]


def test_poll_hpc_preserves_sacct_fallback_job_order_without_orchestrator() -> None:
    connection = _PollConnection(squeue_out="451101 R\n", sacct_out="451103 COMPLETED\n451102 CANCELLED+\n")

    states = poll_hpc(connection, HPCType.OLIVIA, ["451101", "451102", "451103"])

    assert states == {"451101": "RUNNING", "451102": "CANCELLED", "451103": "COMPLETED"}
    assert connection.commands[1] == (HPCType.OLIVIA, "sacct -n -o 'JobID,State' -j 451102,451103", True)


def test_update_scheduler_statuses_marks_terminal_without_scanning_logs() -> None:
    connection = _PollConnection(squeue_out="", sacct_out="451102 COMPLETED\n")
    experiments = [
        {
            "experiment_id": "exp-1",
            "status": ExperimentStatus.RUNNING,
            "hpc_assignment": HPCType.OLIVIA,
            "job_id": "451102",
        }
    ]
    gathered: list[tuple[str, str, HPCType]] = []

    update_scheduler_statuses(
        experiments,
        connection_manager=connection,
        concurrency_limits={HPCType.OLIVIA: 1},
        gather_logs=lambda exp, job_id, hpc_type: gathered.append((exp["experiment_id"], job_id, hpc_type)),
    )

    assert experiments[0]["status"] == ExperimentStatus.COMPLETED
    assert "last_change_ts" in experiments[0]
    assert gathered == []


def test_expand_and_map_are_component_level_helpers() -> None:
    assert expand_short("PD") == "PENDING"
    assert expand_short("CANCELLED+") == "CANCELLED"
    assert map_state("PENDING") == ExperimentStatus.QUEUED
    assert map_state("COMPLETED") == ExperimentStatus.COMPLETED
