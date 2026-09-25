"""Deploy the prebuilt public bundle to a Hugging Face Space."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol


class HubApi(Protocol):
    def create_repo(self, **kwargs):
        ...

    def upload_folder(self, **kwargs):
        ...


def deploy_bundle(
    api: HubApi,
    *,
    repo_id: str,
    bundle_path: Path,
) -> None:
    """Create/update the Space and mirror the allowlisted bundle."""

    if not bundle_path.is_dir():
        raise FileNotFoundError("Hugging Face deployment bundle does not exist")

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(bundle_path),
        delete_patterns="*",
        commit_message="Deploy tested GitHub commit",
    )


def main() -> None:
    token = os.getenv("HF_DEPLOY_TOKEN", "").strip()
    repo_id = os.getenv("HF_SPACE_ID", "").strip()

    if not token:
        raise RuntimeError("HF_DEPLOY_TOKEN is not configured")
    if not repo_id:
        raise RuntimeError("HF_SPACE_ID is not configured")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    deploy_bundle(
        api,
        repo_id=repo_id,
        bundle_path=Path(".hf_bundle"),
    )


if __name__ == "__main__":
    main()
