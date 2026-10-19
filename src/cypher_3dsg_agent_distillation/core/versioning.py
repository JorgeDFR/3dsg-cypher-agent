from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

VERSION_FORMAT = "%Y-%m-%d-T-%H-%M-%S-Z"
VERSION_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}-T-\d{2}-\d{2}-\d{2}-Z$")
SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
METADATA_NAME = "metadata.yaml"
SENSITIVE_KEY_PARTS = ("api_key", "apikey", "password", "secret")
SENSITIVE_TOKEN_KEYS = {"token", "access_token", "auth_token", "hf_token"}


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_now() -> str:
    return utc_now().isoformat()


def version_id(moment: datetime | None = None) -> str:
    value = moment or utc_now()
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).strftime(VERSION_FORMAT)


def validate_version(value: str) -> str:
    normalized = str(value).strip()
    if not VERSION_PATTERN.fullmatch(normalized):
        raise ValueError(f"Version must use UTC format YYYY-MM-DD-T-HH-MM-SS-Z; got `{value}`.")
    try:
        datetime.strptime(normalized, VERSION_FORMAT).replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(f"Invalid version timestamp `{value}`.") from exc
    return normalized


def validate_safe_name(value: str, *, label: str) -> str:
    normalized = str(value).strip()
    if not normalized or not SAFE_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"{label} must be a single path-safe name containing letters, numbers, "
            "dots, underscores, or hyphens."
        )
    if normalized in {".", ".."}:
        raise ValueError(f"Invalid {label}: `{value}`.")
    return normalized


def slugify(value: str, *, fallback: str = "artifact") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-._")
    return slug or fallback


def metadata_path(version_dir: str | Path) -> Path:
    return Path(version_dir) / METADATA_NAME


def read_metadata(path_or_dir: str | Path) -> dict[str, Any]:
    path = Path(path_or_dir)
    if path.is_dir() or path.suffix == "":
        path = metadata_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected metadata mapping at {path}.")
    return payload


def atomic_write_yaml(payload: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def create_version_dir(root: str | Path, requested: str | None = None) -> tuple[str, Path]:
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    selected = validate_version(requested) if requested else version_id()
    target = directory / selected
    try:
        target.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"Version `{selected}` already exists under {directory}. Wait one second or "
            "select another timestamp."
        ) from exc
    return selected, target


def version_directories(root: str | Path, *, completed_only: bool = False) -> list[Path]:
    directory = Path(root)
    if not directory.is_dir():
        return []
    versions = []
    for candidate in directory.iterdir():
        if not candidate.is_dir() or not VERSION_PATTERN.fullmatch(candidate.name):
            continue
        try:
            validate_version(candidate.name)
        except ValueError:
            continue
        if completed_only:
            try:
                status = str(read_metadata(candidate).get("status") or "")
            except (FileNotFoundError, TypeError, yaml.YAMLError):
                continue
            if status != "complete":
                continue
        versions.append(candidate)
    return sorted(versions, key=lambda path: path.name)


def resolve_version_dir(
    root: str | Path,
    selected: str | None = None,
    *,
    completed_only: bool = False,
    artifact_label: str = "version",
) -> tuple[str, Path]:
    directory = Path(root)
    if selected and selected != "latest":
        requested = validate_version(selected)
        target = directory / requested
        if not target.is_dir():
            raise FileNotFoundError(
                f"{artifact_label.capitalize()} `{requested}` does not exist under {directory}."
            )
        if completed_only and str(read_metadata(target).get("status") or "") != "complete":
            raise ValueError(f"{artifact_label.capitalize()} `{requested}` is not complete.")
        return requested, target

    versions = version_directories(directory, completed_only=completed_only)
    if not versions:
        qualifier = " completed" if completed_only else ""
        raise FileNotFoundError(f"No{qualifier} {artifact_label}s exist under {directory}.")
    target = versions[-1]
    return target.name, target


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: str | Path, *, relative_to: str | Path | None = None) -> dict[str, Any]:
    resolved = Path(path)
    display_path: Path = resolved
    if relative_to is not None:
        try:
            display_path = resolved.resolve().relative_to(Path(relative_to).resolve())
        except ValueError:
            pass
    record: dict[str, Any] = {"path": str(display_path)}
    if not resolved.is_file():
        record["exists"] = False
        return record
    stat = resolved.stat()
    record.update(
        {
            "exists": True,
            "size": stat.st_size,
            "sha256": sha256_file(resolved),
        }
    )
    return record


def mapping_fingerprint(values: dict[str, Any]) -> str:
    rendered = yaml.safe_dump(values, sort_keys=True, allow_unicode=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def git_provenance(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    status = run("status", "--short")
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": bool(status),
        "git_changed_files": status.splitlines() if status else [],
    }


def update_metadata(path_or_dir: str | Path, **updates: Any) -> dict[str, Any]:
    path = Path(path_or_dir)
    if path.is_dir() or path.suffix == "":
        path = metadata_path(path)
    payload = read_metadata(path)
    payload.update(updates)
    atomic_write_yaml(payload, path)
    return payload


def record_stage(
    metadata_file: str | Path,
    name: str,
    *,
    arguments: dict[str, Any],
    outputs: dict[str, str | Path],
    project_root: str | Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = read_metadata(metadata_file)
    stages = dict(payload.get("stages") or {})
    output_records = {
        key: file_fingerprint(path, relative_to=project_root) for key, path in outputs.items()
    }
    stage = {
        "completed_at": iso_now(),
        "arguments": arguments,
        "outputs": output_records,
    }
    if extra:
        stage.update(extra)
    stages[name] = stage
    payload["stages"] = stages
    payload["updated_at"] = iso_now()
    atomic_write_yaml(payload, metadata_file)
    return payload


def require_new_outputs(paths: list[str | Path], *, force: bool = False) -> None:
    existing = [Path(path) for path in paths if Path(path).exists()]
    if existing and not force:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing version artifacts: {rendered}. "
            "Start a new version or pass --force."
        )


def serializable_arguments(args: Any, *, excluded: set[str] | None = None) -> dict[str, Any]:
    omitted = {"func", *(excluded or set())}
    values = vars(args) if hasattr(args, "__dict__") else dict(args)
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in values.items()
        if key not in omitted and not sensitive_key(key)
    }


def sensitive_key(key: Any) -> bool:
    normalized = str(key).strip().lower()
    return normalized in SENSITIVE_TOKEN_KEYS or any(
        part in normalized for part in SENSITIVE_KEY_PARTS
    )


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>" if sensitive_key(key) else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def config_record(path: str | Path, *, project_root: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    record = file_fingerprint(resolved, relative_to=project_root)
    if resolved.is_file():
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        record["snapshot"] = redact_secrets(payload)
    return record


def split_fingerprints(
    splits: dict[str, str | Path], *, project_root: str | Path
) -> tuple[dict[str, dict[str, Any]], str]:
    records = {
        name: file_fingerprint(path, relative_to=project_root)
        for name, path in sorted(splits.items())
    }
    hashes = {name: record.get("sha256") for name, record in records.items()}
    return records, mapping_fingerprint(hashes)


def lineage_difference(model_metadata: dict[str, Any], data_config: dict[str, Any]) -> list[str]:
    training_dataset = model_metadata.get("dataset") or {}
    differences = []
    expected_dataset = training_dataset.get("dataset_id")
    expected_version = training_dataset.get("version")
    if expected_dataset != data_config.get("dataset_id"):
        selected_dataset = data_config.get("dataset_id")
        differences.append(
            f"dataset differs (trained on {expected_dataset!r}, selected {selected_dataset!r})"
        )
    if expected_version != data_config.get("dataset_version"):
        differences.append(
            "dataset version differs "
            f"(trained on {expected_version!r}, selected {data_config.get('dataset_version')!r})"
        )
    expected_fingerprint = training_dataset.get("split_fingerprint")
    if expected_fingerprint:
        _, selected_fingerprint = split_fingerprints(
            data_config["splits"], project_root=Path(data_config["dataset_dir"])
        )
        if selected_fingerprint != expected_fingerprint:
            differences.append("the selected split files differ from the training split files")
    return differences


def confirm_dataset_lineage(
    *,
    model_metadata: dict[str, Any],
    data_config: dict[str, Any],
    allow_mismatch: bool = False,
) -> bool:
    """Return whether the operation may continue with the selected lineage."""
    differences = lineage_difference(model_metadata, data_config)
    if not differences:
        return True
    details = "; ".join(differences)
    message = f"Model/dataset lineage mismatch: {details}."
    if allow_mismatch:
        print(f"WARNING: {message}", file=sys.stderr)
        return True
    if not sys.stdin.isatty():
        raise RuntimeError(
            f"{message} Re-run with --allow-dataset-mismatch to continue non-interactively."
        )
    print(f"WARNING: {message}", file=sys.stderr)
    response = input("Continue with this model and dataset? [y/N] ").strip().lower()
    if response not in {"y", "yes"}:
        print("Cancelled: model and dataset lineage do not match.", file=sys.stderr)
        return False
    return True
