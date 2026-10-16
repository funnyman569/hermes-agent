"""A clean worker exit (rc=0) without a terminal kanban call must block, not crash.

The dispatcher's dead-worker sweep counts a clean exit while the task is still
``running`` as a protocol violation, which burns retry budget and eventually trips
the breaker. Instead, the worker process should proactively call ``kanban_block``
when it exits rc=0 and the task is still running — the agent did its work but
forgot (or was unable) to call ``kanban_complete`` / ``kanban_request_review``.

This converts a silent protocol violation into a deliberate, human-readable block
that an operator can review and unblock, rather than a retry loop that never helps.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from hermes_cli import gateway_client, kanban_worker_client


def _no_sleep(_seconds):
    return None


class _StubOwner:
    """Owner that admits a claim but never calls a terminal kanban method."""

    def __init__(self, exit_code=0):
        self.exit_code = exit_code

    async def rpc(self, method, **params):
        if method == "kanban.run":
            return {
                "session_id": "s1",
                "receipt": {"status": "terminal", "admission_id": "a1"},
            }
        if method == "prompt.receipt":
            return {"status": "terminal", "admission_id": "a1"}
        return {}


def _stub_connect(owner):
    @asynccontextmanager
    async def _connect():
        yield owner
    return _connect


@pytest.fixture
def _board(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "kanban.db"
    kb.init_db(db_path=path)
    return path


@pytest.fixture
def _claimed_task(_board, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect_closing

    with connect_closing(_board) as conn:
        task_id = kb.create_task(conn, title="t", assignee="default")
        task = kb.claim_task(conn, task_id, claimer="test-host:1234")
        assert task is not None
    return {"board": "default", "db": str(_board), "task_id": task_id,
            "run_id": task.current_run_id, "claim_lock": task.claim_lock}


@pytest.mark.asyncio
async def test_clean_exit_rc0_still_running_blocks_with_reason(monkeypatch, _board, _claimed_task):
    """When the worker exits rc=0 and the task is still running, it must block
    the task with a descriptive reason rather than leaving a protocol violation."""
    from gateway import session_kanban
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect_closing

    monkeypatch.setattr(
        session_kanban, "worker_result",
        lambda board_db, params: {"exit_code": 0, "last_output": "done"},
    )

    blocked = []
    real_block = kb.block_task

    def _spy_block(conn, task_id, *, reason=None, kind=None, expected_run_id=None):
        blocked.append((task_id, reason, expected_run_id))
        return real_block(conn, task_id, reason=reason, kind=kind, expected_run_id=expected_run_id)

    monkeypatch.setattr(kb, "block_task", _spy_block)
    monkeypatch.setattr(gateway_client, "connect_gateway", _stub_connect(_StubOwner()))
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _no_sleep)

    result = await kanban_worker_client.run(
        {k: _claimed_task[k] for k in ("task_id", "run_id", "claim_lock", "board")},
        str(_board),
    )

    assert result == 0
    assert len(blocked) == 1
    tid, reason, run_id = blocked[0]
    assert tid == _claimed_task["task_id"]
    assert run_id == _claimed_task["run_id"]
    assert "rc=0" in reason
    assert "kanban_complete" in reason

    # The task should now be blocked.
    with connect_closing(Path(str(_board))) as conn:
        task = kb.get_task(conn, _claimed_task["task_id"])
        assert task.status == "blocked"


@pytest.mark.asyncio
async def test_clean_exit_rc0_not_running_is_noop(monkeypatch, _board, tmp_path):
    """If the agent already called kanban_complete before exiting, the task
    is no longer running — _block_if_still_running must be a no-op."""
    from gateway import session_kanban
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect_closing

    monkeypatch.setattr(
        session_kanban, "worker_result",
        lambda board_db, params: {"exit_code": 0, "last_output": "done"},
    )

    with connect_closing(_board) as conn:
        task_id = kb.create_task(conn, title="t2", assignee="default")
        kb.claim_task(conn, task_id, claimer="test-host:1234")
        kb.complete_task(conn, task_id, summary="worked", force=True)

    params = {
        "task_id": task_id, "run_id": None, "claim_lock": "fake",
        "board": "default",
    }

    blocked = []
    real_block = kb.block_task

    def _spy_block(conn, tid, *, reason=None, kind=None, expected_run_id=None):
        blocked.append(tid)
        return real_block(conn, tid, reason=reason, kind=kind, expected_run_id=expected_run_id)

    monkeypatch.setattr(kb, "block_task", _spy_block)
    monkeypatch.setattr(gateway_client, "connect_gateway", _stub_connect(_StubOwner()))
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _no_sleep)

    result = await kanban_worker_client.run(params, str(_board))

    assert result == 0
    assert blocked == []


@pytest.mark.asyncio
async def test_nonzero_exit_does_not_block(monkeypatch, _board, _claimed_task):
    """A non-clean exit (rc!=0) must NOT trigger the auto-block path — the
    dispatcher's normal crash protocol handles it."""
    from gateway import session_kanban
    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(
        session_kanban, "worker_result",
        lambda board_db, params: {"exit_code": 1, "last_output": "error"},
    )
    blocked = []
    real_block = kb.block_task

    def _spy_block(conn, tid, *, reason=None, kind=None, expected_run_id=None):
        blocked.append(tid)
        return real_block(conn, tid, reason=reason, kind=kind, expected_run_id=expected_run_id)

    monkeypatch.setattr(kb, "block_task", _spy_block)
    monkeypatch.setattr(gateway_client, "connect_gateway", _stub_connect(_StubOwner()))
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _no_sleep)

    result = await kanban_worker_client.run(
        {k: _claimed_task[k] for k in ("task_id", "run_id", "claim_lock", "board")},
        str(_board),
    )

    assert result == 1
    assert blocked == []
