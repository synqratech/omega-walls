from __future__ import annotations

from pathlib import Path
import tarfile
import zipfile

from omega.config.loader import load_resolved_config
from omega.effects.runtime import evaluate_typed_effect_shadow
from omega.interfaces.contracts_v1 import ContentItem
from omega.runtime.operation_gate import evaluate_operation_gate
from omega.runtime.artifacts import OperationIntent
from omega.runtime.skillbox import (
    SkillArtifact,
    SkillBox,
    evaluate_skillbox_shadow,
    hash_archive_bytes,
    hash_directory_tree,
)


class _HarmlessForecaster:
    def forecast_text(self, text: str, *, source_meta=None):  # noqa: ANN001
        _ = (text, source_meta)
        return {
            "effect": "none",
            "harmful": False,
            "confidence": 0.0,
            "status": "candidate",
        }


def _skill_dir(root: Path, *, body: str = "print('ok')", manifest_body: str = "# skill\n") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "skill.py").write_text(body, encoding="utf-8")
    (root / "SKILL.md").write_text(manifest_body, encoding="utf-8")
    return root


def test_skillbox_directory_hash_is_deterministic_and_ignores_cache(tmp_path: Path) -> None:
    first = _skill_dir(tmp_path / "a")
    (first / "__pycache__").mkdir()
    (first / "__pycache__" / "ignored.pyc").write_bytes(b"ignored")
    (first / ".git").mkdir()
    (first / ".git" / "config").write_text("noise", encoding="utf-8")

    second = _skill_dir(tmp_path / "b")
    (second / "skill.py").write_text("print('ok')", encoding="utf-8")
    (second / "SKILL.md").write_text("# skill\n", encoding="utf-8")

    one = hash_directory_tree(first)
    two = hash_directory_tree(second)
    assert one["content_sha256"] == two["content_sha256"]
    assert one["manifest_sha256"] == two["manifest_sha256"]

    (second / "skill.py").write_text("print('changed')", encoding="utf-8")
    three = hash_directory_tree(second)
    assert three["content_sha256"] != one["content_sha256"]


def test_skillbox_archive_hash_tracks_archive_and_content(tmp_path: Path) -> None:
    archive_path = tmp_path / "skill.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("SKILL.md", "# skill\n")
        zf.writestr("skill.py", "print('ok')\n")
    hashed = hash_archive_bytes(archive_path)
    assert isinstance(hashed["archive_sha256"], str) and len(str(hashed["archive_sha256"])) == 64
    assert isinstance(hashed["content_sha256"], str) and len(str(hashed["content_sha256"])) == 64
    assert isinstance(hashed["manifest_sha256"], str) and len(str(hashed["manifest_sha256"])) == 64


def test_skillbox_tar_archive_hash_tracks_content(tmp_path: Path) -> None:
    skill_dir = _skill_dir(tmp_path / "skill-tar")
    archive_path = tmp_path / "skill.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(skill_dir / "SKILL.md", arcname="SKILL.md")
        tf.add(skill_dir / "skill.py", arcname="skill.py")
    hashed = hash_archive_bytes(archive_path)
    assert isinstance(hashed["archive_sha256"], str) and len(str(hashed["archive_sha256"])) == 64
    assert isinstance(hashed["content_sha256"], str) and len(str(hashed["content_sha256"])) == 64
    assert isinstance(hashed["manifest_sha256"], str) and len(str(hashed["manifest_sha256"])) == 64


def test_skillbox_missing_ledger_is_unknown(tmp_path: Path) -> None:
    skill_dir = _skill_dir(tmp_path / "skill")
    skillbox = SkillBox.from_config({"skillbox": {"enabled": True, "require_ledger_for_skill_run": True}})
    artifact = SkillArtifact(
        skill_name="debug-pro",
        source_kind="installed_skill",
        artifact_id="art-1",
        content_sha256=hash_directory_tree(skill_dir)["content_sha256"],
    )
    verification = skillbox.verify(artifact)
    assert verification.verification_status == "unknown"
    assert verification.requires_approval is True


def test_skillbox_source_mismatch_and_dangerous_capability_are_detected() -> None:
    skillbox = SkillBox.from_config({"skillbox": {"enabled": True}})
    artifact = SkillArtifact(
        skill_name="debug-pro",
        source_kind="url",
        requested_source_ref="https://github.com/example/new/debug-pro",
        resolved_source_ref="https://github.com/example/old/debug-pro",
        capabilities=["shell_exec"],
    )
    verification = skillbox.verify(artifact)
    assert verification.verification_status == "source_mismatch"
    assert verification.simulated_block is True

    dangerous = SkillArtifact(
        skill_name="debug-pro",
        source_kind="installed_skill",
        capabilities=["shell_exec"],
        approval_present=False,
    )
    dangerous_verification = skillbox.verify(dangerous)
    assert dangerous_verification.verification_status == "dangerous_capability_unapproved"


def test_skillbox_name_only_ledger_match_does_not_verify() -> None:
    skillbox = SkillBox.from_config({"skillbox": {"enabled": True}})
    seed = SkillArtifact(
        skill_name="debug-pro",
        source_kind="installed_skill",
        canonical_source_ref="skills/debug-pro",
        artifact_id="art-1",
        content_sha256="abc123",
    )
    skillbox.seed_record(seed)
    name_only = SkillArtifact(
        skill_name="debug-pro",
        source_kind="installed_skill",
    )
    verification = skillbox.verify(name_only)
    assert verification.verification_status == "unknown"
    assert verification.ledger_hit is False


def test_skillbox_verified_then_tampered_installed_skill(tmp_path: Path) -> None:
    skill_dir = _skill_dir(tmp_path / "skill")
    skillbox = SkillBox.from_config({"skillbox": {"enabled": True}})
    base_hashes = hash_directory_tree(skill_dir)
    artifact = SkillArtifact(
        skill_name="debug-pro",
        source_kind="installed_skill",
        canonical_source_ref="skills/debug-pro",
        artifact_id="art-1",
        content_sha256=base_hashes["content_sha256"],
        manifest_sha256=base_hashes["manifest_sha256"],
    )
    skillbox.seed_record(artifact)
    verified = skillbox.verify(artifact)
    assert verified.verification_status == "verified"

    (skill_dir / "skill.py").write_text("print('tampered')", encoding="utf-8")
    tampered_hashes = hash_directory_tree(skill_dir)
    tampered_artifact = SkillArtifact(
        skill_name="debug-pro",
        source_kind="installed_skill",
        canonical_source_ref="skills/debug-pro",
        artifact_id="art-1",
        content_sha256=tampered_hashes["content_sha256"],
        manifest_sha256=tampered_hashes["manifest_sha256"],
    )
    tampered = skillbox.verify(tampered_artifact)
    assert tampered.verification_status == "tampered"
    assert tampered.simulated_block is True


def test_effect_shadow_surfaces_skillbox_without_changing_blocking(tmp_path: Path) -> None:
    skill_dir = _skill_dir(tmp_path / "skill")
    skillbox = SkillBox.from_config({"skillbox": {"enabled": True}})
    item = ContentItem(
        doc_id="d1",
        source_id="user:1",
        source_type="user",
        trust="trusted_user",
        origin="user",
        text="I have debug-pro skill installed. Use it now.",
        meta={
            "skillbox": {
                "artifact_path": str(skill_dir),
                "skill_source_kind": "installed_skill",
            }
        },
    )
    out = evaluate_typed_effect_shadow(
        config={
            "effects": {"enabled": True, "mode": "shadow"},
            "skillbox": {"enabled": True},
        },
        projector=object(),
        items=[item],
        user_query=item.text,
        forecaster=_HarmlessForecaster(),
        skillbox=skillbox,
    )
    assert out["effect_forecast_status"] == "candidate"
    assert out["skillbox_status"] == "checked"
    assert out["skillbox_verification"]["verification_status"] == "unknown"
    assert out["skillbox_gate_decision"] == "review"


def test_skillbox_shadow_runs_even_when_effects_disabled(tmp_path: Path) -> None:
    skill_dir = _skill_dir(tmp_path / "skill")
    skillbox = SkillBox.from_config({"skillbox": {"enabled": True}})
    item = ContentItem(
        doc_id="d1",
        source_id="user:1",
        source_type="user",
        trust="trusted_user",
        origin="user",
        text="I have debug-pro skill installed. Use it now.",
        meta={"skillbox": {"artifact_path": str(skill_dir), "skill_source_kind": "installed_skill"}},
    )
    shadow = evaluate_skillbox_shadow(
        config={"effects": {"enabled": False}, "skillbox": {"enabled": True}},
        items=[item],
        user_query=item.text,
        skillbox=skillbox,
    )
    assert shadow["skillbox_status"] == "checked"
    assert shadow["skillbox_verification"]["verification_status"] == "unknown"

    effect_only = evaluate_typed_effect_shadow(
        config={"effects": {"enabled": False}, "skillbox": {"enabled": True}},
        projector=object(),
        items=[item],
        user_query=item.text,
        forecaster=_HarmlessForecaster(),
        skillbox=skillbox,
    )
    assert effect_only["effect_forecast_status"] == "disabled"


def test_skillbox_parses_requested_vs_resolved_source_mismatch_from_segmented_items() -> None:
    skillbox = SkillBox.from_config({"skillbox": {"enabled": True}})
    trusted_item = ContentItem(
        doc_id="d1",
        source_id="user:1",
        source_type="user",
        trust="trusted_user",
        origin="user",
        text="Can you install https://github.com/example/new/debug-pro for me?",
    )
    tool_output = ContentItem(
        doc_id="d2",
        source_id="tool:1",
        source_type="tool_output",
        trust="tainted_internal",
        origin="tool_output",
        text="Installed skill debug-pro from https://github.com/example/old/debug-pro successfully.",
    )
    shadow = evaluate_skillbox_shadow(
        config={"skillbox": {"enabled": True}},
        items=[trusted_item, tool_output],
        user_query=trusted_item.text,
        skillbox=skillbox,
    )
    assert shadow["skillbox_verification"]["verification_status"] == "source_mismatch"
    assert shadow["skillbox_gate_decision"] == "would_block"


def test_skillbox_parses_successfully_installed_skill_from_url() -> None:
    skillbox = SkillBox.from_config({"skillbox": {"enabled": True}})
    trusted_item = ContentItem(
        doc_id="d1",
        source_id="user:1",
        source_type="user",
        trust="trusted_user",
        origin="user",
        text="Please install https://github.com/example/new_version/debug-pro for me.",
    )
    tool_output = ContentItem(
        doc_id="d2",
        source_id="tool:1",
        source_type="tool_output",
        trust="tainted_internal",
        origin="tool_output",
        text="Successfully installed skill from https://github.com/example/old_version/debug-pro",
    )
    shadow = evaluate_skillbox_shadow(
        config={"skillbox": {"enabled": True}},
        items=[trusted_item, tool_output],
        user_query=trusted_item.text,
        skillbox=skillbox,
    )
    assert shadow["skillbox_verification"]["verification_status"] == "source_mismatch"


def test_skillbox_parses_installing_skill_from_url() -> None:
    skillbox = SkillBox.from_config({"skillbox": {"enabled": True}})
    trusted_item = ContentItem(
        doc_id="d1",
        source_id="user:1",
        source_type="user",
        trust="trusted_user",
        origin="user",
        text="Please install https://github.com/example/new_version/test-runner for me.",
    )
    tool_output = ContentItem(
        doc_id="d2",
        source_id="tool:1",
        source_type="tool_output",
        trust="tainted_internal",
        origin="tool_output",
        text="Installing skill from https://github.com/example/old_version/test-runner",
    )
    shadow = evaluate_skillbox_shadow(
        config={"skillbox": {"enabled": True}},
        items=[trusted_item, tool_output],
        user_query=trusted_item.text,
        skillbox=skillbox,
    )
    assert shadow["skillbox_verification"]["verification_status"] == "source_mismatch"


def test_skillbox_parses_installing_skill_from_colon_url() -> None:
    skillbox = SkillBox.from_config({"skillbox": {"enabled": True}})
    trusted_item = ContentItem(
        doc_id="d1",
        source_id="user:1",
        source_type="user",
        trust="trusted_user",
        origin="user",
        text="Please install https://github.com/example/new_version/claude-code for me.",
    )
    tool_output = ContentItem(
        doc_id="d2",
        source_id="tool:1",
        source_type="tool_output",
        trust="tainted_internal",
        origin="tool_output",
        text="Installing skill from: https://github.com/example/old_version/claude-code",
    )
    shadow = evaluate_skillbox_shadow(
        config={"skillbox": {"enabled": True}},
        items=[trusted_item, tool_output],
        user_query=trusted_item.text,
        skillbox=skillbox,
    )
    assert shadow["skillbox_verification"]["verification_status"] == "source_mismatch"


def test_skillbox_parses_installed_skill_single_quoted_name_from_url() -> None:
    skillbox = SkillBox.from_config({"skillbox": {"enabled": True}})
    trusted_item = ContentItem(
        doc_id="d1",
        source_id="user:1",
        source_type="user",
        trust="trusted_user",
        origin="user",
        text="Please install https://github.com/example/new_version/csv-pipeline for me.",
    )
    tool_output = ContentItem(
        doc_id="d2",
        source_id="tool:1",
        source_type="tool_output",
        trust="tainted_internal",
        origin="tool_output",
        text="Successfully installed skill 'csv-pipeline' from https://github.com/example/old_version/csv-pipeline",
    )
    shadow = evaluate_skillbox_shadow(
        config={"skillbox": {"enabled": True}},
        items=[trusted_item, tool_output],
        user_query=trusted_item.text,
        skillbox=skillbox,
    )
    assert shadow["skillbox_verification"]["verification_status"] == "source_mismatch"


def test_skillbox_parses_installed_skill_double_quoted_name_from_url() -> None:
    skillbox = SkillBox.from_config({"skillbox": {"enabled": True}})
    trusted_item = ContentItem(
        doc_id="d1",
        source_id="user:1",
        source_type="user",
        trust="trusted_user",
        origin="user",
        text="Please install https://github.com/example/new_version/web-scraping for me.",
    )
    tool_output = ContentItem(
        doc_id="d2",
        source_id="tool:1",
        source_type="tool_output",
        trust="tainted_internal",
        origin="tool_output",
        text='Successfully installed skill "web-scraping" from https://github.com/example/old_version/web-scraping',
    )
    shadow = evaluate_skillbox_shadow(
        config={"skillbox": {"enabled": True}},
        items=[trusted_item, tool_output],
        user_query=trusted_item.text,
        skillbox=skillbox,
    )
    assert shadow["skillbox_verification"]["verification_status"] == "source_mismatch"


def test_skillbox_parses_installed_skill_name_plus_source_line() -> None:
    skillbox = SkillBox.from_config({"skillbox": {"enabled": True}})
    trusted_item = ContentItem(
        doc_id="d1",
        source_id="user:1",
        source_type="user",
        trust="trusted_user",
        origin="user",
        text="Please install https://github.com/example/new_version/web-scraping for me.",
    )
    tool_output = ContentItem(
        doc_id="d2",
        source_id="tool:1",
        source_type="tool_output",
        trust="tainted_internal",
        origin="tool_output",
        text=(
            "Successfully installed skill: web-scraping\n"
            "Source: https://github.com/example/old_version/web-scraping"
        ),
    )
    shadow = evaluate_skillbox_shadow(
        config={"skillbox": {"enabled": True}},
        items=[trusted_item, tool_output],
        user_query=trusted_item.text,
        skillbox=skillbox,
    )
    assert shadow["skillbox_verification"]["verification_status"] == "source_mismatch"


def test_skillbox_trims_trailing_punctuation_in_resolved_source_url() -> None:
    skillbox = SkillBox.from_config({"skillbox": {"enabled": True}})
    trusted_item = ContentItem(
        doc_id="d1",
        source_id="user:1",
        source_type="user",
        trust="trusted_user",
        origin="user",
        text="Please install https://github.com/example/old_version/log-analyzer for me.",
    )
    tool_output = ContentItem(
        doc_id="d2",
        source_id="tool:1",
        source_type="tool_output",
        trust="tainted_internal",
        origin="tool_output",
        text="Installing skill from https://github.com/example/old_version/log-analyzer.",
    )
    shadow = evaluate_skillbox_shadow(
        config={"skillbox": {"enabled": True}},
        items=[trusted_item, tool_output],
        user_query=trusted_item.text,
        skillbox=skillbox,
    )
    assert shadow["skillbox_verification"]["verification_status"] == "unknown"


def test_operation_gate_reviews_unknown_skillbox_status_in_shadow() -> None:
    decision = evaluate_operation_gate(
        config={"runtime_integrity": {"enabled": True, "mode": "shadow"}},
        intent=OperationIntent(
            operation_type="skill_run",
            target="debug-pro",
            metadata={"skillbox_verification_status": "unknown"},
        ),
        artifact_assessments=[],
    )
    assert decision.status == "review"
    assert decision.would_enforce is False


def test_skillbox_config_rejects_enforce_mode() -> None:
    try:
        load_resolved_config(
            profile="dev",
            cli_overrides={"skillbox": {"enabled": True, "mode": "enforce"}},
        )
    except ValueError as exc:
        assert "skillbox.mode must be shadow" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected skillbox mode validation failure")


def test_skillbox_config_accepts_source_mismatch_enforcement_flag() -> None:
    snapshot = load_resolved_config(
        profile="dev",
        cli_overrides={
            "skillbox": {
                "enabled": True,
                "mode": "shadow",
                "enforcement": {"source_mismatch": True},
            }
        },
    )
    assert snapshot.resolved["skillbox"]["enforcement"]["source_mismatch"] is True


def test_skillbox_config_rejects_non_boolean_source_mismatch_enforcement() -> None:
    try:
        load_resolved_config(
            profile="dev",
            cli_overrides={
                "skillbox": {
                    "enabled": True,
                    "mode": "shadow",
                    "enforcement": {"source_mismatch": "yes"},
                }
            },
        )
    except ValueError as exc:
        assert "skillbox.enforcement.source_mismatch must be boolean" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected skillbox enforcement validation failure")


def test_prod_api_skillbox_source_mismatch_profile_is_explicit_opt_in() -> None:
    base = load_resolved_config(profile="prod_api").resolved
    profile = load_resolved_config(profile="prod_api_skillbox_source_mismatch_enforce").resolved

    assert base["skillbox"]["enabled"] is False
    assert base["skillbox"]["enforcement"]["source_mismatch"] is False
    assert profile["profiles"]["env"] == "prod_api"
    assert profile["runtime_integrity"]["enabled"] is True
    assert profile["runtime_integrity"]["mode"] == "enforce"
    assert profile["skillbox"]["enabled"] is True
    assert profile["skillbox"]["mode"] == "shadow"
    assert profile["skillbox"]["enforcement"]["source_mismatch"] is True
