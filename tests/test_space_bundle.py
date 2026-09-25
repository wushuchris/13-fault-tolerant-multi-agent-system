from pathlib import Path

from scripts.build_space_bundle import build_bundle


def test_space_bundle_uses_explicit_public_allowlist(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = build_bundle(tmp_path / "space", repo_root=root)

    assert (output / "app.py").is_file()
    assert (output / "requirements.txt").is_file()
    assert (output / "src" / "fault_tolerant_agents").is_dir()
    assert (output / "EVALUATION.md").is_file()
    assert (output / "LICENSE").is_file()

    assert not (output / ".github").exists()
    assert not (output / "tests").exists()
    assert not (output / ".env.example").exists()
    assert not (output / "scripts").exists()


def test_space_readme_has_required_hugging_face_metadata(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    output = build_bundle(tmp_path / "space", repo_root=root)
    readme = (output / "README.md").read_text(encoding="utf-8")

    assert readme.startswith("---\n")
    assert "sdk: gradio" in readme
    assert "sdk_version: 6.28.0" in readme
    assert "python_version: 3.11" in readme
    assert "app_file: app.py" in readme
    assert "# Fault-Tolerant Multi-Agent System" in readme


def test_space_bundle_never_copies_unknown_repository_file(
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    (fake_root / "README.md").write_text("# Demo\n", encoding="utf-8")
    (fake_root / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (fake_root / "requirements.txt").write_text("", encoding="utf-8")
    (fake_root / "PRIVATE_CURRICULUM.md").write_text(
        "do not deploy",
        encoding="utf-8",
    )
    (fake_root / "src").mkdir()
    (fake_root / "src" / "module.py").write_text("x = 1\n", encoding="utf-8")

    output = build_bundle(tmp_path / "bundle", repo_root=fake_root)

    assert not (output / "PRIVATE_CURRICULUM.md").exists()
