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
if "scripts/graph/extract_3dsg_values.py" in sys.argv:
    sys.exit(1)
if "scripts/cypher/generate_3dsg_seed_templates.py" in sys.argv:
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


@pytest.mark.parametrize("quantizations,outtype,variants", [
    (["Q4_K_M"], "bf16", ["BF16", "Q4_K_M"]),
    (["q8_0", "Q5_K_M", "Q8_0"], "bf16", ["BF16", "Q8_0", "Q5_K_M"]),
    (["Q8_0"], "f16", ["F16", "Q8_0"]),
    (None, "bf16", ["BF16", "Q4_K_M"]),
])
def test_deploy_uses_configured_variants_throughout(tmp_path, quantizations, outtype, variants):
    import sys

    import yaml

    config = tmp_path / "student.yaml"
    export = {"model_name": "custom-model", "gguf_outtype": outtype}
    if quantizations is not None:
        export["quantizations"] = quantizations
    config.write_text(yaml.safe_dump({
        "model": {"base_model": "vendor/model"},
        "training": {"output_dir": str(tmp_path / "adapter")},
        "ollama_export": export,
    }))
    log = tmp_path / "commands.jsonl"
    docker = tmp_path / "docker"
    docker.write_text('''#!/usr/bin/env python3
import json
import os
import subprocess
import sys
with open(os.environ["COMMAND_LOG"], "a") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\\n")
if "--services" in sys.argv:
    print("main\\nneo4j")
if "scripts/training/deploy_helpers.py" in sys.argv and "config" in sys.argv:
    index = sys.argv.index("scripts/training/deploy_helpers.py")
    sys.exit(subprocess.call([os.environ["TEST_PYTHON"], *sys.argv[index:]]))
''')
    docker.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "docker/stack.sh"), "demo-deploy"],
        cwd=ROOT,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
             "COMMAND_LOG": str(log), "TEST_PYTHON": sys.executable,
             "DEMO_STUDENT_CONFIG": str(config)},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    export_calls = [call for call in calls if "scripts/training/export_agent_ollama.py" in call]
    assert len(export_calls) == 2
    assert all("--quantizations" not in call for call in export_calls)
    modelfiles = export_calls[1]
    assert modelfiles[modelfiles.index("--modelfile-variants") + 1:] == variants
    registrations = [call for call in calls if "create" in call]
    assert [call[call.index("create") + 1] for call in registrations] == [
        f"custom-model-{variant.lower()}" for variant in variants
    ]
    assert [call[-1] for call in registrations] == [
        f"/models/custom-model/Modelfile.{variant.lower()}" for variant in variants
    ]
    evaluations = [call for call in calls if "scripts/agents/evaluate_student.py" in call]
    assert [call[call.index("--ollama-model") + 1] for call in evaluations] == [
        f"custom-model-{variant.lower()}" for variant in variants
    ]
    total = 4 + len(variants)
    assert f"Step {total}/{total}" in result.stdout
    assert not any("-c" in call for call in calls)
    for call in calls:
        for argument in call:
            if argument.startswith("scripts/"):
                assert (ROOT / argument).is_file()
