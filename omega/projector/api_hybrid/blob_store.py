"""Request-scoped in-memory image blob store.

Raw media bytes stay inside this adapter-near boundary. Core contracts carry only
opaque blob:// handles and integrity metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import threading
import time
import uuid
from typing import Dict, Optional


_ALLOWED_IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})


@dataclass(frozen=True)
class _BlobRecord:
    scope_id: str
    data: bytes
    mime: str
    sha256: str
    created_monotonic: float


class ImageBlobStore:
    def __init__(
        self,
        *,
        max_blob_bytes: int = 20 * 1024 * 1024,
        max_total_bytes: int = 128 * 1024 * 1024,
        max_records: int = 256,
        ttl_sec: float = 120.0,
    ) -> None:
        self.max_blob_bytes = max(1, int(max_blob_bytes))
        self.max_total_bytes = max(self.max_blob_bytes, int(max_total_bytes))
        self.max_records = max(1, int(max_records))
        self.ttl_sec = max(1.0, float(ttl_sec))
        self._lock = threading.RLock()
        self._records: Dict[str, _BlobRecord] = {}
        self._total_bytes = 0

    def _purge_expired_locked(self) -> None:
        cutoff = time.monotonic() - self.ttl_sec
        expired = [ref for ref, rec in self._records.items() if rec.created_monotonic < cutoff]
        for ref in expired:
            record = self._records.pop(ref, None)
            if record is not None:
                self._total_bytes = max(0, self._total_bytes - len(record.data))

    def put(self, *, scope_id: str, data: bytes, mime: str, expected_sha256: Optional[str] = None) -> str:
        scope = str(scope_id).strip()
        raw = bytes(data)
        normalized_mime = str(mime).strip().lower()
        if not scope:
            raise ValueError("blob scope_id must be non-empty")
        if normalized_mime not in _ALLOWED_IMAGE_MIME_TYPES:
            raise ValueError("unsupported image blob mime")
        if not raw:
            raise ValueError("image blob must be non-empty")
        if len(raw) > self.max_blob_bytes:
            raise ValueError("image blob exceeds max_blob_bytes")
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 and digest != str(expected_sha256).strip().lower():
            raise ValueError("image blob sha256 mismatch")
        # Never expose caller-controlled request/session identifiers in the handle.
        scope_token = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:32]
        ref = f"blob://{scope_token}/{uuid.uuid4().hex}"
        record = _BlobRecord(
            scope_id=scope,
            data=raw,
            mime=normalized_mime,
            sha256=digest,
            created_monotonic=time.monotonic(),
        )
        with self._lock:
            self._purge_expired_locked()
            if len(self._records) >= self.max_records:
                raise RuntimeError("image blob store record capacity exceeded")
            if self._total_bytes + len(raw) > self.max_total_bytes:
                raise RuntimeError("image blob store byte capacity exceeded")
            self._records[ref] = record
            self._total_bytes += len(raw)
        return ref

    def resolve(self, *, bytes_ref: str, expected_sha256: str, expected_mime: str) -> bytes:
        ref = str(bytes_ref).strip()
        with self._lock:
            self._purge_expired_locked()
            record = self._records.get(ref)
        if record is None:
            raise KeyError("image blob reference missing or expired")
        if record.sha256 != str(expected_sha256).strip().lower():
            raise ValueError("image blob sha256 mismatch")
        if record.mime != str(expected_mime).strip().lower():
            raise ValueError("image blob mime mismatch")
        if hashlib.sha256(record.data).hexdigest() != record.sha256:
            raise ValueError("image blob integrity check failed")
        return bytes(record.data)

    def delete_scope(self, scope_id: str) -> int:
        scope = str(scope_id).strip()
        with self._lock:
            refs = [ref for ref, rec in self._records.items() if rec.scope_id == scope]
            for ref in refs:
                record = self._records.pop(ref, None)
                if record is not None:
                    self._total_bytes = max(0, self._total_bytes - len(record.data))
        return len(refs)

    def count(self) -> int:
        with self._lock:
            self._purge_expired_locked()
            return len(self._records)

    def total_bytes(self) -> int:
        with self._lock:
            self._purge_expired_locked()
            return int(self._total_bytes)
