from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.sync_public_github_repo import sync_public_repo


def test_sync_public_repo_curates_and_deletes_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    src_root = tmp_path / "src"
    src_root.mkdir(parents=True, exist_ok=True)
    (src_root / "README.md").write_text("# demo\n", encoding="utf-8")
    (src_root / "omega").mkdir(parents=True, exist_ok=True)
    (src_root / "omega" / "core.py").write_text("x = 1\n", encoding="utf-8")
    (src_root / "redteam").mkdir(parents=True, exist_ok=True)
    (src_root / "redteam" / "secret.txt").write_text("secret\n", encoding="utf-8")

    manifest_path = tmp_path / "oss_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "include": ["README.md", "omega", "redteam"],
                "exclude_globs": ["redteam/**"],
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    public_repo = tmp_path / "public_repo"
    (public_repo / ".git").mkdir(parents=True, exist_ok=True)
    (public_repo / "stale.txt").parent.mkdir(parents=True, exist_ok=True)
    (public_repo / "stale.txt").write_text("old\n", encoding="utf-8")

    import scripts.sync_public_github_repo as sync_mod

    monkeypatch.setattr(sync_mod, "ROOT", src_root)

    summary = sync_public_repo(
        manifest_path=manifest_path,
        staging_dir=tmp_path / "staging",
        target_repo_dir=public_repo,
        clean_staging=True,
        delete_extra=True,
        git_commit=False,
        commit_message="sync",
        git_push=False,
        git_remote="origin",
        git_branch=None,
    )

    assert summary["status"] == "ok"
    assert (public_repo / "README.md").exists()
    assert (public_repo / "omega" / "core.py").exists()
    assert not (public_repo / "redteam" / "secret.txt").exists()
    assert not (public_repo / "stale.txt").exists()
    assert (tmp_path / "staging" / "sync_report.json").exists()


def test_sync_public_repo_requires_git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    src_root = tmp_path / "src"
    src_root.mkdir(parents=True, exist_ok=True)
    (src_root / "README.md").write_text("# demo\n", encoding="utf-8")
    manifest_path = tmp_path / "oss_manifest.json"
    manifest_path.write_text(
        json.dumps({"include": ["README.md"], "exclude_globs": []}, ensure_ascii=True),
        encoding="utf-8",
    )
    target = tmp_path / "not_git_repo"
    target.mkdir(parents=True, exist_ok=True)

    import scripts.sync_public_github_repo as sync_mod

    monkeypatch.setattr(sync_mod, "ROOT", src_root)

    with pytest.raises(RuntimeError, match="not a git repository"):
        sync_public_repo(
            manifest_path=manifest_path,
            staging_dir=tmp_path / "staging",
            target_repo_dir=target,
            clean_staging=True,
            delete_extra=False,
            git_commit=False,
            commit_message="sync",
            git_push=False,
            git_remote="origin",
            git_branch=None,
        )
