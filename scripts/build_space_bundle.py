"""Build the public Hugging Face Space deployment bundle.

The bundle uses an explicit allowlist so CI never mirrors arbitrary repository
contents to the public Space.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SPACE_METADATA = """---
title: Fault-Tolerant Multi-Agent System
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.28.0
python_version: 3.11
app_file: app.py
license: mit
pinned: false
short_description: Resilient multi-agent operations with trust-aware recovery.
---

"""

ALLOWED_FILES = (
    "app.py",
    "requirements.txt",
    "LICENSE",
    "EVALUATION.md",
)


def build_bundle(
    output_dir: Path,
    *,
    repo_root: Path | None = None,
) -> Path:
    """Build a minimal public Space bundle from an explicit file allowlist."""

    root = repo_root or Path(__file__).resolve().parents[1]
    output = output_dir.resolve()

    if output == root.resolve():
        raise ValueError("deployment bundle cannot overwrite repository root")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    readme = root / "README.md"
    if not readme.is_file():
        raise FileNotFoundError("README.md is required to build Space bundle")

    (output / "README.md").write_text(
        SPACE_METADATA + readme.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    for relative_path in ALLOWED_FILES:
        source = root / relative_path
        if source.is_file():
            shutil.copy2(source, output / relative_path)

    source_package = root / "src"
    if not source_package.is_dir():
        raise FileNotFoundError("src directory is required to build Space bundle")
    shutil.copytree(source_package, output / "src")

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=".hf_bundle",
        help="Directory to create for Hugging Face Space deployment.",
    )
    args = parser.parse_args()
    build_bundle(Path(args.output))


if __name__ == "__main__":
    main()
