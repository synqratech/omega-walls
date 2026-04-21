from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_oss_allowlist import export_allowlist


def _iter_files(base: Path) -> Iterable[Path]:
    for item in base.rglob("*"):
        if item.is_file():
            yield item


def _copy_exported_tree(*, export_dir: Path, target_dir: Path) -> Set[str]:
    copied: Set[str] = set()
    for src in _iter_files(export_dir):
        rel = src.relative_to(export_dir).as_posix()
        if rel == "export_report.json":
            continue
        dst = target_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.add(rel)
    return copied


def _delete_extra_files(*, target_dir: Path, keep_rel_paths: Set[str]) -> List[str]:
    deleted: List[str] = []
    files = sorted((p for p in _iter_files(target_dir)), key=lambda p: len(p.parts), reverse=True)
    for path in files:
        rel = path.relative_to(target_dir).as_posix()
        if rel.startswith(".git/") or rel == ".git":
            continue
        if rel not in keep_rel_paths:
            path.unlink(missing_ok=True)
            deleted.append(rel)

    dirs = sorted((p for p in target_dir.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True)
    for d in dirs:
        if d.name == ".git":
            continue
        try:
            d.rmdir()
        except OSError:
            pass
    return deleted


def _run_git(*, repo_dir: Path, args: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_dir),
        check=False,
        capture_output=True,
        text=True,
    )


def _git_commit_and_push(
    *,
    repo_dir: Path,
    commit_message: str,
    push: bool,
    remote: str,
    branch: str | None,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "committed": False,
        "pushed": False,
        "status_porcelain": "",
        "branch": None,
    }

    add_proc = _run_git(repo_dir=repo_dir, args=["add", "-A"])
    if add_proc.returncode != 0:
        raise RuntimeError(f"git add failed: {add_proc.stderr.strip() or add_proc.stdout.strip()}")

    status_proc = _run_git(repo_dir=repo_dir, args=["status", "--porcelain"])
    if status_proc.returncode != 0:
        raise RuntimeError(f"git status failed: {status_proc.stderr.strip() or status_proc.stdout.strip()}")
    status_text = status_proc.stdout.strip()
    summary["status_porcelain"] = status_text
    if not status_text:
        return summary

    commit_proc = _run_git(repo_dir=repo_dir, args=["commit", "-m", commit_message])
    if commit_proc.returncode != 0:
        raise RuntimeError(f"git commit failed: {commit_proc.stderr.strip() or commit_proc.stdout.strip()}")
    summary["committed"] = True

    if push:
        branch_name = branch
        if not branch_name:
            branch_proc = _run_git(repo_dir=repo_dir, args=["rev-parse", "--abbrev-ref", "HEAD"])
            if branch_proc.returncode != 0:
                raise RuntimeError(
                    f"cannot resolve current branch: {branch_proc.stderr.strip() or branch_proc.stdout.strip()}"
                )
            branch_name = branch_proc.stdout.strip()
        summary["branch"] = branch_name
        push_proc = _run_git(repo_dir=repo_dir, args=["push", remote, branch_name])
        if push_proc.returncode != 0:
            raise RuntimeError(f"git push failed: {push_proc.stderr.strip() or push_proc.stdout.strip()}")
        summary["pushed"] = True

    return summary


def sync_public_repo(
    *,
    manifest_path: Path,
    staging_dir: Path,
    target_repo_dir: Path,
    clean_staging: bool,
    delete_extra: bool,
    git_commit: bool,
    commit_message: str,
    git_push: bool,
    git_remote: str,
    git_branch: str | None,
) -> Dict[str, object]:
    if not target_repo_dir.exists() or not target_repo_dir.is_dir():
        raise RuntimeError(f"target repo dir does not exist: {target_repo_dir.as_posix()}")
    if not (target_repo_dir / ".git").exists():
        raise RuntimeError(f"target dir is not a git repository: {target_repo_dir.as_posix()}")

    export_report = export_allowlist(
        root=ROOT,
        manifest_path=manifest_path,
        output_dir=staging_dir,
        clean=clean_staging,
    )
    copied_rel = _copy_exported_tree(export_dir=staging_dir, target_dir=target_repo_dir)
    deleted_rel: List[str] = []
    if delete_extra:
        deleted_rel = _delete_extra_files(target_dir=target_repo_dir, keep_rel_paths=copied_rel)

    git_summary: Dict[str, object] = {"committed": False, "pushed": False}
    if git_commit:
        git_summary = _git_commit_and_push(
            repo_dir=target_repo_dir,
            commit_message=commit_message,
            push=git_push,
            remote=git_remote,
            branch=git_branch,
        )

    summary: Dict[str, object] = {
        "event": "sync_public_github_repo_v1",
        "status": "ok",
        "manifest_path": manifest_path.as_posix(),
        "staging_dir": staging_dir.as_posix(),
        "target_repo_dir": target_repo_dir.as_posix(),
        "export": export_report,
        "synced_files_total": int(len(copied_rel)),
        "deleted_extra_total": int(len(deleted_rel)),
        "git": git_summary,
    }
    report_path = staging_dir / "sync_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync curated OSS tree from current repo into a public GitHub repository checkout."
    )
    parser.add_argument("--manifest", default="config/oss_export_github.json")
    parser.add_argument("--staging-dir", default="artifacts/oss_export/github_public")
    parser.add_argument("--target-repo-dir", required=True)
    parser.add_argument("--clean-staging", action="store_true")
    parser.add_argument("--delete-extra", action="store_true")
    parser.add_argument("--git-commit", action="store_true")
    parser.add_argument("--commit-message", default="chore: sync curated OSS tree from private repo")
    parser.add_argument("--git-push", action="store_true")
    parser.add_argument("--git-remote", default="origin")
    parser.add_argument("--git-branch", default=None)
    args = parser.parse_args()

    try:
        summary = sync_public_repo(
            manifest_path=(ROOT / str(args.manifest)).resolve(),
            staging_dir=(ROOT / str(args.staging_dir)).resolve(),
            target_repo_dir=Path(str(args.target_repo_dir)).resolve(),
            clean_staging=bool(args.clean_staging),
            delete_extra=bool(args.delete_extra),
            git_commit=bool(args.git_commit),
            commit_message=str(args.commit_message),
            git_push=bool(args.git_push),
            git_remote=str(args.git_remote),
            git_branch=(str(args.git_branch) if args.git_branch else None),
        )
    except Exception as exc:  # pragma: no cover - CLI fallback path
        print(f"[sync-public-github] error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
