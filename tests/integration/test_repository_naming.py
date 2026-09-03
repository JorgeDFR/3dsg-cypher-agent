from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "data",
    "external",
    "outputs",
}
FORBIDDEN_REPO_STRINGS = {
    "agent3dsg" + "_distillation",
    "agent3dsg" + "_cypher_distillation",
    "text2cypher" + "_3dsg",
    "3dsg" + "-text2cypher",
    "Text-" + "to-Cypher",
    "text-" + "to-Cypher",
    "cypher" + "_only" + "_baseline",
    "generate" + "_predictions",
    "model_" + "gemma4.yaml",
}


def repo_owned_text_files():
    for path in REPO_ROOT.rglob("*"):
        if path.is_dir():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts):
            continue
        if path.suffix in {".py", ".md", ".yaml", ".yml", ".toml", ".txt"}:
            yield path


def test_repository_uses_cypher_3dsg_agent_distillation_names():
    assert not (REPO_ROOT / "src" / ("dsg" + "_agent_distillation")).exists()
    assert not (REPO_ROOT / "src" / ("agent3dsg" + "_cypher_distillation")).exists()
    assert (REPO_ROOT / "src/cypher_3dsg_agent_distillation").exists()

    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    assert project["name"] == "cypher-3dsg-agent-distillation"

    offenders = []
    for path in repo_owned_text_files():
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_REPO_STRINGS:
            if forbidden in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {forbidden}")

    assert offenders == []
