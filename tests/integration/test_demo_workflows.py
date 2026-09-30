import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("command, scripts", [
    ("demo-dataset", [
        "extract_3dsg_values.py", "generate_3dsg_seed_templates.py", "prepare_dataset.py",
        "generate_traces.py", "prepare_dataset.py",
    ]),
    ("demo-train", ["train_agent_lora.py"]),
    ("demo-validate", ["evaluate_student.py"] * 5 + ["generate_workflow_report.py"]),
])
def test_demo_commands_run_only_their_own_stages(tmp_path, command, scripts):
    log = tmp_path / "commands.jsonl"
    docker = tmp_path / "docker"
    docker.write_text('''#!/usr/bin/env python3
import json
import os
import sys
with open(os.environ["COMMAND_LOG"], "a") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\\n")
if "--services" in sys.argv:
    print("main\\nneo4j")
''')
    docker.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "docker" / "stack.sh"), command],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
             "COMMAND_LOG": str(log), "DEMO_DATA_CONFIG": "custom-data.yaml",
             "DEMO_STUDENT_CONFIG": "custom-student.yaml"},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    stages = [call for call in calls if "exec" in call]
    assert [Path(next(arg for arg in call if arg.startswith("scripts/"))).name
            for call in stages] == scripts
    assert all("custom-data.yaml" in call for call in stages)
    if command == "demo-validate":
        assert sum("--baseline" in call for call in stages) == 2
    if command == "demo-train":
        assert "custom-student.yaml" in stages[0]


def test_demo_stops_when_a_stage_fails(tmp_path):
    docker = tmp_path / "docker"
    docker.write_text('''#!/usr/bin/env python3
import sys
if "--services" in sys.argv:
    print("main\\nneo4j")
if "scripts/extract_3dsg_values.py" in sys.argv:
    sys.exit(1)
if "scripts/generate_3dsg_seed_templates.py" in sys.argv:
    print("UNEXPECTED NEXT STAGE")
''')
    docker.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "docker" / "stack.sh"), "demo-dataset"],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Step 1/5 failed" in result.stderr
    assert "UNEXPECTED NEXT STAGE" not in result.stdout
