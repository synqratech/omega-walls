from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.pitheta.dataset_builder import build_pitheta_dataset_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PiTheta training dataset artifacts.")
    parser.add_argument("--registry", default="config/pitheta_dataset_registry.yml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--use-semantic-labeling", action="store_true")
    args = parser.parse_args()

    report = build_pitheta_dataset_artifacts(
        registry_path=str(args.registry),
        output_dir=str(args.output_dir),
        seed=int(args.seed),
        profile=str(args.profile),
        strict=bool(args.strict),
        use_semantic_labeling=bool(args.use_semantic_labeling),
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
