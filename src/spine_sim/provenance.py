"""Capture the public checkout used by a run, including uncommitted sources."""

import hashlib
import json
import platform
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], text=True, stderr=subprocess.DEVNULL
    ).rstrip("\n")


def capture_source(root: Path, output: Path) -> dict:
    """Archive actual source bytes, never .git or ignored/generated outputs.

    A run is intentionally made from its source checkout. A replay extracted
    from this archive has no Git history; its metadata records that fact.
    """
    root = root.resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "src/spine_sim").is_dir():
        raise ValueError(
            "run from the package source directory containing pyproject.toml"
        )
    if Path(__file__).resolve() != root / "src/spine_sim/provenance.py":
        raise ValueError(
            "the imported package must be the editable installation "
            "of this source checkout"
        )
    try:
        git_root = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    except subprocess.CalledProcessError, FileNotFoundError:
        git_root = None
    if git_root is not None and git_root != root:
        raise ValueError("the source checkout must be its own Git repository")
    excluded = {
        ".git",
        ".venv",
        "results",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
    }
    if git_root is not None:
        try:
            revision = _git(root, "rev-parse", "HEAD")
        except subprocess.CalledProcessError:
            revision = None
        status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
        candidates = _git(
            root, "ls-files", "-z", "--cached", "--others", "--exclude-standard"
        ).split("\0")
        git_info = {"revision": revision, "dirty": bool(status), "status": status}
    else:
        candidates = []
        for directory, subdirectories, filenames in root.walk():
            subdirectories[:] = [
                name
                for name in subdirectories
                if name not in excluded and not name.endswith(".egg-info")
            ]
            candidates.extend(
                str((directory / name).relative_to(root)) for name in filenames
            )
        git_info = {
            "revision": None,
            "dirty": None,
            "status": "No Git history (e.g. source snapshot replay).",
        }
    manifest = {}
    archive_path = output / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name in sorted(set(candidates)):
            relative = Path(name)
            if not name or relative.is_absolute() or ".." in relative.parts:
                continue
            if any(
                part in excluded or part.endswith(".egg-info")
                for part in relative.parts
            ):
                continue
            if any(
                part == ".env" or part.startswith(".env.") for part in relative.parts
            ):
                continue
            path = root / relative
            if path.is_relative_to(output.resolve()) or not path.exists():
                continue
            if path.is_symlink():
                raise ValueError(
                    f"source snapshot requires regular files, found symlink: {name}"
                )
            if not path.is_file():
                continue
            manifest[name] = file_hash(path)
            info = archive.gettarinfo(str(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    if "uv.lock" not in manifest:
        raise ValueError("uv.lock is required for a reproducible run")
    return {
        "git": git_info,
        "source": {
            "archive": archive_path.name,
            "archive_sha256": file_hash(archive_path),
            "files_sha256": manifest,
            "description": (
                "Actual tracked and untracked public source files; "
                "no Git history or ignored outputs."
            ),
        },
    }


def environment_metadata() -> dict:
    try:
        uv_version = subprocess.check_output(["uv", "--version"], text=True).strip()
    except OSError, subprocess.CalledProcessError:
        uv_version = None
    return {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "uv_version": uv_version,
        "dependencies": {
            distribution.metadata["Name"]: distribution.version
            for distribution in sorted(
                metadata.distributions(), key=lambda item: item.metadata["Name"].lower()
            )
        },
    }
