"""Dirty tracked and untracked source bytes survive without repository history."""

import subprocess
import tarfile
from pathlib import Path

from spine_sim import provenance


def test_source_snapshot_captures_dirty_and_untracked_files(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "src/spine_sim").mkdir(parents=True)
    for name in ("pyproject.toml", "uv.lock", "src/spine_sim/provenance.py"):
        (root / name).write_text("original\n")
    (root / ".gitignore").write_text("results/\n.env\n")
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test Fixture",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Synthetic fixture",
        ],
        check=True,
    )
    (root / "pyproject.toml").write_text("modified source\n")
    (root / "src/spine_sim/new.py").write_text("untracked source\n")
    (root / ".env").write_text("private local settings\n")
    output = root / "results/test"
    output.mkdir(parents=True)
    monkeypatch.setattr(
        provenance, "__file__", str(root / "src/spine_sim/provenance.py")
    )
    info = provenance.capture_source(root, output)
    assert info["git"]["dirty"] is True
    assert len(info["git"]["revision"]) == 40
    with tarfile.open(output / "source.tar.gz") as archive:
        assert archive.extractfile("pyproject.toml").read() == b"modified source\n"
        assert (
            archive.extractfile("src/spine_sim/new.py").read() == b"untracked source\n"
        )
        assert ".env" not in archive.getnames()
        assert not any(
            Path(name).parts[0] in {".git", "results"} for name in archive.getnames()
        )
