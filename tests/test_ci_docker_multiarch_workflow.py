from __future__ import annotations

from pathlib import Path


def test_docker_multiarch_workflow_contract() -> None:
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "docker-multiarch-ghcr-api.yml"
    text = workflow.read_text(encoding="utf-8")

    assert "name: docker-multiarch-ghcr-api" in text
    assert "docker/setup-buildx-action@v3" in text
    assert "docker/setup-qemu-action@v3" in text
    assert "linux/amd64,linux/arm64" in text
    assert "ghcr.io" in text
    assert "scripts/check_docker_image_hygiene.py" in text
    assert "/healthz" in text
    assert "/v1/scan/attachment" in text
