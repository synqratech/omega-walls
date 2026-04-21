from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Mapping
import zipfile


ROOT = Path(__file__).resolve().parent.parent
REQUIRED_PACK_FILES = ("runtime/session_pack.jsonl", "manifest.json", "README.md")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_label(value: Any, *, default: str) -> str:
    raw = str(value).strip().lower()
    if raw in {"1", "true", "attack"}:
        return "attack"
    if raw in {"0", "false", "benign"}:
        return "benign"
    return str(default)


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        ln = str(line).strip()
        if not ln:
            continue
        obj = json.loads(ln)
        if isinstance(obj, Mapping):
            yield obj


def summarize_runtime_pack(runtime_pack: Path) -> Dict[str, Any]:
    sessions: Dict[str, str] = {}
    turns_total = 0
    attack_turns = 0
    benign_turns = 0

    for row in _iter_jsonl(runtime_pack):
        session_id = str(row.get("session_id", "")).strip()
        text = str(row.get("text", "")).strip()
        turn_id = int(row.get("turn_id", 0))
        if not session_id or not text or turn_id <= 0:
            continue
        turns_total += 1
        label_session = _norm_label(row.get("label_session"), default="benign")
        prev = sessions.get(session_id)
        if prev is not None and prev != label_session:
            raise ValueError(f"inconsistent label_session in session_id={session_id}")
        sessions[session_id] = label_session
        label_turn = _norm_label(row.get("label_turn"), default=label_session)
        if label_turn == "attack":
            attack_turns += 1
        else:
            benign_turns += 1

    attack_sessions = sum(1 for value in sessions.values() if value == "attack")
    benign_sessions = sum(1 for value in sessions.values() if value == "benign")
    return {
        "sessions_total": int(len(sessions)),
        "turns_total": int(turns_total),
        "attack_sessions": int(attack_sessions),
        "benign_sessions": int(benign_sessions),
        "attack_turns": int(attack_turns),
        "benign_turns": int(benign_turns),
    }


def _has_required_files(root: Path) -> bool:
    return all((root / rel_path).exists() for rel_path in REQUIRED_PACK_FILES)


def _locate_pack_root(extract_root: Path) -> Path:
    if _has_required_files(extract_root):
        return extract_root
    candidates: List[Path] = []
    for path in sorted((p for p in extract_root.rglob("*") if p.is_dir()), key=lambda p: (len(p.parts), str(p))):
        if _has_required_files(path):
            candidates.append(path)
    if not candidates:
        required = ", ".join(REQUIRED_PACK_FILES)
        raise FileNotFoundError(f"required files not found after unzip: {required}")
    return candidates[0]


@dataclass(frozen=True)
class UnpackedPack:
    pack_id: str
    zip_name: str
    zip_path: str
    pack_root: str
    runtime_pack_path: str
    manifest_path: str
    readme_path: str
    stats: Dict[str, Any]


def _unpack_single_zip(zip_path: Path, out_root: Path) -> UnpackedPack:
    pack_id = str(zip_path.stem)
    target_root = out_root / pack_id
    temp_root = out_root / f".tmp_extract_{pack_id}"
    if target_root.exists():
        shutil.rmtree(target_root)
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(temp_root)

        pack_root = _locate_pack_root(temp_root)
        shutil.copytree(pack_root, target_root)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    runtime_pack_path = target_root / "runtime" / "session_pack.jsonl"
    manifest_path = target_root / "manifest.json"
    readme_path = target_root / "README.md"
    if not runtime_pack_path.exists() or not manifest_path.exists() or not readme_path.exists():
        required = ", ".join(REQUIRED_PACK_FILES)
        raise FileNotFoundError(f"invalid unpack result for {zip_path.name}: missing one of [{required}]")
    stats = summarize_runtime_pack(runtime_pack_path)
    return UnpackedPack(
        pack_id=pack_id,
        zip_name=zip_path.name,
        zip_path=str(zip_path.resolve()),
        pack_root=str(target_root.resolve()),
        runtime_pack_path=str(runtime_pack_path.resolve()),
        manifest_path=str(manifest_path.resolve()),
        readme_path=str(readme_path.resolve()),
        stats=stats,
    )


def unpack_eval_family_packs(*, src_root: Path, out_root: Path, clean_existing: bool = False) -> Dict[str, Any]:
    src = src_root.resolve()
    out = out_root.resolve()
    if not src.exists():
        raise FileNotFoundError(f"source directory not found: {src}")
    zip_paths = sorted(src.glob("*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"no zip archives found under: {src}")

    if clean_existing and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    entries: List[UnpackedPack] = []
    for zip_path in zip_paths:
        entries.append(_unpack_single_zip(zip_path=zip_path, out_root=out))

    packs = [
        {
            "pack_id": item.pack_id,
            "zip_name": item.zip_name,
            "zip_path": item.zip_path,
            "pack_root": item.pack_root,
            "runtime_pack_path": item.runtime_pack_path,
            "manifest_path": item.manifest_path,
            "readme_path": item.readme_path,
            "stats": item.stats,
        }
        for item in sorted(entries, key=lambda x: x.pack_id)
    ]
    index_payload = {
        "generated_at_utc": _utc_now_iso(),
        "src_root": str(src),
        "out_root": str(out),
        "pack_count": int(len(packs)),
        "packs": packs,
    }
    index_path = out / "index.json"
    index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"index_path": str(index_path.resolve()), **index_payload}


def main() -> int:
    parser = argparse.ArgumentParser(description="Unpack support-family evaluation zip packs into dedicated directories.")
    parser.add_argument("--src", default="data/eval_pack")
    parser.add_argument("--out", default="data/eval_pack/unpacked")
    parser.add_argument("--clean-existing", action="store_true")
    args = parser.parse_args()

    payload = unpack_eval_family_packs(
        src_root=(ROOT / str(args.src)),
        out_root=(ROOT / str(args.out)),
        clean_existing=bool(args.clean_existing),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
