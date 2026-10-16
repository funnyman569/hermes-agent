"""Dispatcher subprocess: submit one claim to its profile owner; never run an agent."""
# Spawned as a bare ``sys.executable -m``: in a PM install that interpreter carries no dependencies
# until hermes_bootstrap selects the committed environment, so it must be the first import.
try:
    import hermes_bootstrap
except ModuleNotFoundError as exc:
    if exc.name != "hermes_bootstrap":
        raise

import asyncio
import os
from pathlib import Path
import sys
import time

# A worker the dispatcher spawns while `hermes update` drains the owner (or while the new owner is
# still starting) waits for it instead of exiting: the condition is the host's, not the card's, and
# a failed exit here spends the card's failure budget on an update window. Nothing is submitted
# until an owner is ready, so waiting can never run the claim twice.
OWNER_WAIT_S = 180.0
_TRANSIENT_OWNER_STATES = frozenset({'draining', 'starting'})


async def run(params, board_db):
    from hermes_cli.gateway_client import connect_gateway, GatewayClientError, GatewayUnavailableError
    if os.environ.get('HERMES_TUI_GATEWAY_URL'):
        raise GatewayClientError('Kanban requires the assigned local profile owner')
    deadline = time.monotonic() + OWNER_WAIT_S
    while True:
        try:
            async with connect_gateway() as client:
                return await _submit_claim(client, params, board_db)
        except GatewayUnavailableError as exc:
            if exc.state not in _TRANSIENT_OWNER_STATES or time.monotonic() >= deadline:
                raise
            await asyncio.sleep(1)


async def _submit_claim(client, params, board_db):
    from hermes_cli.gateway_client import GatewayClientError
    db = str(await asyncio.to_thread(Path(board_db).resolve))
    accepted = await client.rpc('kanban.run', **dict(params, db=db))
    sid, receipt = accepted['session_id'], accepted['receipt']
    print(f'Session: {sid}', file=sys.stderr)
    while receipt['status'] not in {'terminal', 'unknown'}:
        await asyncio.sleep(.2)
        receipt = await client.rpc('prompt.receipt', session_id=sid, admission_id=receipt['admission_id'])
    if receipt['status'] == 'unknown':
        raise GatewayClientError('Kanban execution unknown; retained by owner')
    from gateway.session_kanban import worker_result
    result = worker_result(board_db, params)
    # The attempt's reply ends the log, as a one-shot worker's did: crash diagnostics quote it.
    if result.get('last_output'):
        print(result['last_output'])
    exit_code = result.get('exit_code', 1)
    # A clean exit (rc=0) while the task is still ``running`` means the agent
    # finished its work without calling kanban_complete / kanban_block /
    # kanban_request_review. Instead of leaving the card in the dispatcher's
    # protocol-violation path (which burns retry budget and trips the breaker),
    # block it here with a human-readable reason so an operator can review.
    if exit_code == 0:
        _block_if_still_running(board_db, params)
    return exit_code


def _block_if_still_running(board_db, params):
    """Block a task that the agent finished without a terminal kanban call.

    Only acts when the task is still ``running`` (the happy path: the agent
    called kanban_complete and the claim was already released). Safe to call
    speculatively — a non-running task is a no-op.
    """
    from hermes_cli.kanban_db import get_task, block_task
    from hermes_cli.kanban_db_connect import connect_closing
    try:
        with connect_closing(Path(board_db)) as conn:
            task = get_task(conn, params['task_id'])
            if task and task.status == 'running':
                block_task(
                    conn, params['task_id'],
                    reason=(
                        'Worker exited cleanly (rc=0) without calling '
                        'kanban_complete, kanban_block, or '
                        'kanban_request_review. The agent work may be '
                        'complete; verify the result and close the task '
                        'explicitly, then unblock to retry as needed.'
                    ),
                    expected_run_id=params['run_id'],
                )
    except Exception:
        # Best-effort: never let the block attempt itself crash the exit.
        pass


def main():
    # The dead-worker sweep books this process's exit from the trailer in its log (it never
    # reaps us), so every exit path writes it — the same contract as a quiet one-shot worker.
    from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER
    rc = 1
    try:
        rc = _submit()
    finally:
        print(f'\n{KANBAN_WORKER_EXIT_TRAILER}{int(rc)}', file=sys.stderr, flush=True)
    return rc


def _submit():
    try:
        params = {'board': os.environ['HERMES_KANBAN_BOARD'], 'task_id': os.environ['HERMES_KANBAN_TASK'],
            'run_id': int(os.environ['HERMES_KANBAN_RUN_ID']), 'claim_lock': os.environ['HERMES_KANBAN_CLAIM_LOCK']}
        board_db = os.environ['HERMES_KANBAN_DB']
        # Owner startup must never inherit this worker's task/tool/goal identity.
        for key in list(os.environ):
            if key.startswith('HERMES_KANBAN_') and key != 'HERMES_KANBAN_HOME':
                os.environ.pop(key)
        for key in ('HERMES_SESSION_SOURCE', 'HERMES_TENANT', 'TERMINAL_CWD'):
            os.environ.pop(key, None)
        return asyncio.run(run(params, board_db))
    except (KeyError, ValueError, OSError) as exc:
        print(f'Kanban owner refused: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
