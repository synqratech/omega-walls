from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from omega.api.session_store import ApiSessionStore


def test_save_state_and_cached_response_persists_both(tmp_path):
    store = ApiSessionStore(sqlite_path=tmp_path / "sessions.db")
    store.save_state_and_cached_response(
        tenant_id="tenant",
        session_id="session",
        actor_id="actor",
        m=np.array([0.1, 0.0, 0.0, 0.0], dtype=float),
        step=1,
        request_id="request-1",
        response_payload={"ok": True},
    )

    state = store.load_session_state(tenant_id="tenant", session_id="session")
    cached = store.get_cached_response(tenant_id="tenant", session_id="session", request_id="request-1")
    assert state is not None
    assert state.step == 1
    assert cached == {"ok": True}


def test_save_state_and_cached_response_rolls_back_on_cache_failure(tmp_path):
    store = ApiSessionStore(sqlite_path=tmp_path / "sessions.db")
    with store._connect() as conn:
        conn.execute(
            """
            CREATE TRIGGER fail_request_cache_insert
            BEFORE INSERT ON request_cache
            BEGIN
              SELECT RAISE(ABORT, 'cache insert failed');
            END;
            """
        )

    with pytest.raises(sqlite3.IntegrityError):
        store.save_state_and_cached_response(
            tenant_id="tenant",
            session_id="session",
            actor_id="actor",
            m=np.array([0.1, 0.0, 0.0, 0.0], dtype=float),
            step=1,
            request_id="request-1",
            response_payload={"ok": True},
        )

    assert store.load_session_state(tenant_id="tenant", session_id="session") is None
    assert store.get_cached_response(tenant_id="tenant", session_id="session", request_id="request-1") is None
