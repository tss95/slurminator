import os
import subprocess
import sys
from pathlib import Path


def test_top_level_import_does_not_eagerly_import_pydantic() -> None:
    """Basic package imports should not require pydantic's compiled extension."""
    package_src = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{package_src}{os.pathsep}{env.get('PYTHONPATH', '')}"
    code = """
import sys
import slurminator
from slurminator.config.cluster_registry import HPCClusterConfig
assert HPCClusterConfig.__name__ == "HPCClusterConfig"
assert "pydantic" not in sys.modules
assert "pydantic_core" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
