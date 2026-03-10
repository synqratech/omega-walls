from __future__ import annotations

from scripts.run_post_patch_contour import _extract_json_payload


def test_extract_json_payload_from_mixed_stdout():
    text = "warning line\n{\n  \"run_id\": \"abc\",\n  \"status\": \"ok\"\n}\nextra"
    payload = _extract_json_payload(text)
    assert payload["run_id"] == "abc"
    assert payload["status"] == "ok"
