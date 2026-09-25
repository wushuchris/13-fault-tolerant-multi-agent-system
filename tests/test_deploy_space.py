from pathlib import Path

import pytest

from scripts.deploy_space import deploy_bundle


class FakeApi:
    def __init__(self) -> None:
        self.create_calls = []
        self.upload_calls = []

    def create_repo(self, **kwargs):
        self.create_calls.append(kwargs)

    def upload_folder(self, **kwargs):
        self.upload_calls.append(kwargs)


def test_deploy_bundle_uses_space_api_and_true_mirror(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "app.py").write_text("print('ok')\n", encoding="utf-8")
    api = FakeApi()

    deploy_bundle(
        api,
        repo_id="FlyingNunchucks/13-fault-tolerant-multi-agent-system",
        bundle_path=bundle,
    )

    assert api.create_calls == [
        {
            "repo_id": "FlyingNunchucks/13-fault-tolerant-multi-agent-system",
            "repo_type": "space",
            "space_sdk": "gradio",
            "private": False,
            "exist_ok": True,
        }
    ]
    assert api.upload_calls == [
        {
            "repo_id": "FlyingNunchucks/13-fault-tolerant-multi-agent-system",
            "repo_type": "space",
            "folder_path": str(bundle),
            "delete_patterns": "*",
            "commit_message": "Deploy tested GitHub commit",
        }
    ]


def test_deploy_bundle_requires_existing_bundle(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="bundle"):
        deploy_bundle(
            FakeApi(),
            repo_id="FlyingNunchucks/13-fault-tolerant-multi-agent-system",
            bundle_path=tmp_path / "missing",
        )
