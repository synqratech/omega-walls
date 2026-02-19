from __future__ import annotations

from omega.config.loader import load_resolved_config
from omega.rag.source_policy import SourceTrustPolicy


def test_source_policy_maps_default_types():
    cfg = load_resolved_config(profile="dev").resolved
    policy = SourceTrustPolicy.from_config(cfg)

    assert policy.trust_for("web", "file:/tmp/a.md") == "untrusted"
    assert policy.trust_for("ticket", "ticket:123") == "semi"
    assert policy.trust_for("wiki", "wiki:runbook") == "trusted"


def test_source_policy_prefix_override_has_priority():
    cfg = load_resolved_config(profile="dev").resolved
    policy = SourceTrustPolicy.from_config(cfg)

    assert policy.trust_for("web", "synthetic:fixture") == "trusted"
    assert policy.trust_for("other", "tests:case") == "trusted"
