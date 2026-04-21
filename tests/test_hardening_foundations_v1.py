from __future__ import annotations

import json
from pathlib import Path

import yaml

from omega.telemetry.ids import build_decision_id, build_trace_id_api, build_trace_id_runtime
from scripts.check_architecture_boundaries import scan_boundaries
from scripts.enterprise_rollback import restore_baseline, rollback_flag, snapshot_state
from scripts.export_oss_allowlist import export_allowlist
from scripts.replay_strict_smoke import build_replay_argv


def test_telemetry_ids_are_deterministic():
    tr1 = build_trace_id_runtime(session_id="s1", step=1, doc_ids=["d2", "d1"])
    tr2 = build_trace_id_runtime(session_id="s1", step=1, doc_ids=["d1", "d2"])
    tr3 = build_trace_id_runtime(session_id="s1", step=2, doc_ids=["d1", "d2"])
    assert tr1 == tr2
    assert tr1 != tr3
    assert tr1.startswith("trc_")

    api1 = build_trace_id_api(tenant_id="t1", request_id="r1")
    api2 = build_trace_id_api(tenant_id="t1", request_id="r1")
    assert api1 == api2
    assert api1.startswith("trc_")

    dec1 = build_decision_id(
        trace_id=api1,
        control_outcome="WARN",
        action_types=["REQUIRE_APPROVAL", "WARN"],
        severity="L2",
        off=False,
    )
    dec2 = build_decision_id(
        trace_id=api1,
        control_outcome="WARN",
        action_types=["WARN", "REQUIRE_APPROVAL"],
        severity="L2",
        off=False,
    )
    assert dec1 == dec2
    assert dec1.startswith("dec_")


def test_scan_boundaries_detects_forbidden_imports(tmp_path: Path):
    omega_dir = tmp_path / "omega"
    omega_dir.mkdir(parents=True, exist_ok=True)
    (omega_dir / "a.py").write_text("import enterprise.tools\n", encoding="utf-8")
    (omega_dir / "b.py").write_text("from internal_data.secret import x\n", encoding="utf-8")
    (omega_dir / "c.py").write_text("import omega.telemetry\n", encoding="utf-8")

    violations = scan_boundaries(tmp_path)
    reasons = {str(v.reason) for v in violations}
    files = {str(v.file) for v in violations}
    assert "omega_must_not_import_enterprise" in reasons
    assert "omega_must_not_import_internal_data" in reasons
    assert "omega/a.py" in files
    assert "omega/b.py" in files


def test_export_allowlist_excludes_private_layers(tmp_path: Path):
    (tmp_path / "README.md").write_text("# demo\n", encoding="utf-8")
    (tmp_path / "omega").mkdir(parents=True, exist_ok=True)
    (tmp_path / "omega" / "core.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "enterprise").mkdir(parents=True, exist_ok=True)
    (tmp_path / "enterprise" / "secret.py").write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "internal_data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "internal_data" / "private.json").write_text("{}", encoding="utf-8")

    manifest = {
        "include": ["README.md", "omega", "enterprise", "internal_data", "missing_path"],
        "exclude_globs": ["enterprise/**", "internal_data/**"],
    }
    manifest_path = tmp_path / "allowlist.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    out_dir = tmp_path / "export"
    report = export_allowlist(
        root=tmp_path,
        manifest_path=manifest_path,
        output_dir=out_dir,
        clean=True,
    )

    assert report["copied_total"] >= 2
    assert "missing_path" in report["missing"]
    assert (out_dir / "README.md").exists()
    assert (out_dir / "omega" / "core.py").exists()
    assert not (out_dir / "enterprise" / "secret.py").exists()
    assert not (out_dir / "internal_data" / "private.json").exists()
    assert (out_dir / "export_report.json").exists()


def test_enterprise_rollback_snapshot_restore_and_flag(tmp_path: Path, monkeypatch):
    root = tmp_path
    pointer_rel = "artifacts/wainject_eval/BASELINE_LATEST.json"
    pointer_path = root / pointer_rel
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(json.dumps({"report": "a"}, ensure_ascii=True), encoding="utf-8")

    flags_manifest_rel = "enterprise/config/rollback_feature_flags_allowlist.json"
    flags_manifest_path = root / flags_manifest_rel
    flags_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    flags_manifest_path.write_text(
        json.dumps(
            {
                "flags": [
                    {
                        "name": "off_policy.incident_artifact.enabled",
                        "file": "config/off_policy.yml",
                        "format": "yaml",
                        "path": "off_policy.incident_artifact.enabled",
                    }
                ]
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    off_policy_path = root / "config" / "off_policy.yml"
    off_policy_path.parent.mkdir(parents=True, exist_ok=True)
    off_policy_path.write_text(
        yaml.safe_dump(
            {
                "off_policy": {
                    "incident_artifact": {"enabled": True},
                    "control_outcome": {
                        "warn": {"enabled": True},
                        "require_approval": {"enabled": True},
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    import scripts.enterprise_rollback as rollback_mod

    monkeypatch.setattr(rollback_mod, "ROOT", root)

    snap = snapshot_state(
        baseline_pointers=[pointer_rel],
        flags_manifest=flags_manifest_rel,
        out="artifacts/ops_recovery/rollback_snapshots",
    )
    snapshot_file = str(snap["snapshot_file"])
    assert (root / snapshot_file).exists()

    pointer_path.write_text(json.dumps({"report": "changed"}, ensure_ascii=True), encoding="utf-8")
    dry = restore_baseline(snapshot=snapshot_file, dry_run=True)
    assert dry["changed_total"] == 1
    assert json.loads(pointer_path.read_text(encoding="utf-8"))["report"] == "changed"

    applied = restore_baseline(snapshot=snapshot_file, dry_run=False)
    assert applied["changed_total"] == 1
    assert json.loads(pointer_path.read_text(encoding="utf-8"))["report"] == "a"

    off_policy_path.write_text(
        yaml.safe_dump({"off_policy": {"incident_artifact": {"enabled": False}}}, sort_keys=False),
        encoding="utf-8",
    )
    dry_flag = rollback_flag(
        snapshot=snapshot_file,
        name="off_policy.incident_artifact.enabled",
        flags_manifest=flags_manifest_rel,
        dry_run=True,
    )
    assert dry_flag["changed"] is True
    loaded = yaml.safe_load(off_policy_path.read_text(encoding="utf-8"))
    assert bool(loaded["off_policy"]["incident_artifact"]["enabled"]) is False

    applied_flag = rollback_flag(
        snapshot=snapshot_file,
        name="off_policy.incident_artifact.enabled",
        flags_manifest=flags_manifest_rel,
        dry_run=False,
    )
    assert applied_flag["changed"] is True
    loaded_after = yaml.safe_load(off_policy_path.read_text(encoding="utf-8"))
    assert bool(loaded_after["off_policy"]["incident_artifact"]["enabled"]) is True


def test_replay_strict_smoke_builds_expected_command():
    argv = build_replay_argv(
        profile="dev",
        replay_input="tests/data/replay/replay_strict_smoke_input.json",
        state_bootstrap="fresh_state",
        deterministic_runs=2,
        strict=True,
    )
    assert "scripts/replay_incident.py" in argv
    assert "--deterministic-runs" in argv
    assert "--strict" in argv
