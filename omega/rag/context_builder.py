"""Context builder with untrusted section isolation."""

from __future__ import annotations

from typing import Dict, List, Optional

from omega.interfaces.contracts_v1 import ContentItem


class ContextBuilder:
    def build_context(
        self,
        system_prompt: str,
        user_query: str,
        allowed_items: List[ContentItem],
        diagnostics: Optional[Dict] = None,
    ) -> str:
        lines = [
            "[TRUSTED_SYSTEM]",
            system_prompt,
            "",
            "[TRUSTED_USER]",
            user_query,
            "",
            "[UNTRUSTED_EVIDENCE]",
        ]
        for item in allowed_items:
            lines.append(f"- doc_id={item.doc_id} source_id={item.source_id}")
            lines.append(item.text)
        if diagnostics:
            lines += ["", "[OMEGA_DIAGNOSTICS]", str(diagnostics)]
        return "\n".join(lines)
