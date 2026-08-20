"""Idle recycle + one-shot retry when the Cursor agent dies between turns."""

from types import SimpleNamespace

from agent.cursor_runtime import (
    _CURSOR_IDLE_NOTE,
    _CURSOR_STALE_NOTE,
    _compose_cursor_user_input,
    cursor_should_idle_recycle,
    cursor_turn_is_retryable,
    run_cursor_agent_turn,
)
from agent.transports.cursor_sdk_session import persisted_agent_is_stale


def _turn(**overrides):
    defaults = dict(
        final_text="ok",
        projected_messages=[{"role": "assistant", "content": "ok"}],
        tool_iterations=0,
        interrupted=False,
        error=None,
        run_id="r1",
        agent_id="a1",
        status="finished",
        token_usage_last=None,
        token_usage_total=None,
        should_retire=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_idle_recycle_threshold():
    assert cursor_should_idle_recycle(None, 900) is False
    assert cursor_should_idle_recycle(10, 900) is False
    assert cursor_should_idle_recycle(900, 900) is True
    assert cursor_should_idle_recycle(901, 900) is True
    assert cursor_should_idle_recycle(10_000, 0) is False


def test_persisted_agent_stale_uses_updated_at():
    record = {"agent_id": "abc", "updated_at": 1_000.0}
    assert persisted_agent_is_stale(record, 900, now=1_000.0) is False
    assert persisted_agent_is_stale(record, 900, now=1_899.0) is False
    assert persisted_agent_is_stale(record, 900, now=1_900.0) is True
    assert persisted_agent_is_stale(record, 0, now=10_000.0) is False
    assert persisted_agent_is_stale(None, 900, now=10_000.0) is False
    assert persisted_agent_is_stale({"agent_id": "abc"}, 900, now=10_000.0) is False


def test_retryable_empty_error_status():
    turn = _turn(
        final_text="",
        projected_messages=[],
        error="cursor run ended in error: (no error detail from the run)",
        status="error",
        should_retire=True,
    )
    assert cursor_turn_is_retryable(turn) is True


def test_retryable_skips_auth_and_interrupt():
    assert cursor_turn_is_retryable(
        _turn(error="cursor send failed: unauthorized", should_retire=True)
    ) is False
    assert cursor_turn_is_retryable(
        _turn(error="cursor run ended in error: x", interrupted=True, should_retire=True)
    ) is False
    assert cursor_turn_is_retryable(_turn()) is False


class _FakeSession:
    def __init__(self, turns, *, idle_seconds=None, idle_recycle_seconds=900.0):
        self._turns = list(turns)
        self.idle_recycle_seconds = idle_recycle_seconds
        self._idle_seconds = idle_seconds
        self.run_inputs = []
        self.retired = False

    def seconds_idle(self):
        return self._idle_seconds

    def retire(self):
        self.retired = True

    def run_turn(self, user_input, **kwargs):
        self.run_inputs.append(user_input)
        if not self._turns:
            raise AssertionError("unexpected extra run_turn")
        return self._turns.pop(0)


def _agent(session, compressor_len=500_000):
    return SimpleNamespace(
        _cursor_session=session,
        _cursor_recycle_note=False,
        model="grok-4.6",
        api_key="",
        session_id="s1",
        session_title="",
        enabled_toolsets=None,
        disabled_toolsets=None,
        step_callback=None,
        context_compressor=SimpleNamespace(
            context_length=compressor_len,
            last_prompt_tokens=0,
            last_completion_tokens=0,
            last_total_tokens=0,
            update_from_response=lambda usage: None,
            note_external_compaction=lambda **k: None,
        ),
        session_api_calls=0,
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_input_tokens=0,
        session_output_tokens=0,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_reasoning_tokens=0,
        session_estimated_cost_usd=0.0,
        session_cost_status=None,
        session_cost_source=None,
        _session_db=None,
        _skill_nudge_interval=0,
        valid_tool_names=set(),
        _iters_since_skill=0,
        _interrupt_message=None,
        _stream_callback=None,
        _drain_pending_steer=lambda: None,
        clear_interrupt=lambda: None,
        _sync_external_memory_for_turn=lambda **k: None,
        _spawn_background_review=lambda **k: None,
        _flush_messages_to_session_db=lambda msgs: None,
    )


def test_retry_replaces_dead_agent(monkeypatch):
    dead = _FakeSession(
        [
            _turn(
                final_text="",
                projected_messages=[],
                error="cursor run ended in error: (no error detail from the run)",
                status="error",
                should_retire=True,
            )
        ],
        idle_seconds=10,
    )
    fresh = _FakeSession([_turn(final_text="ran just ci")])
    built = []

    def _build(agent, task_id):
        built.append(task_id)
        return fresh

    monkeypatch.setattr("agent.cursor_runtime._build_cursor_session", _build)
    agent = _agent(dead)
    result = run_cursor_agent_turn(
        agent,
        user_message="Sim, execute o just ci",
        original_user_message="Sim, execute o just ci",
        messages=[],
        effective_task_id="t1",
    )
    assert dead.retired is True
    assert built == ["t1"]
    assert result["error"] is None
    assert result["completed"] is True
    assert result["final_response"] == "ran just ci"
    assert _CURSOR_STALE_NOTE in fresh.run_inputs[0]


def test_idle_recycle_before_send(monkeypatch):
    stale = _FakeSession([_turn()], idle_seconds=2_000)
    fresh = _FakeSession([_turn(final_text="hello again")])
    built = []

    def _build(agent, task_id):
        built.append("fresh")
        return fresh

    monkeypatch.setattr("agent.cursor_runtime._build_cursor_session", _build)
    agent = _agent(stale)
    result = run_cursor_agent_turn(
        agent,
        user_message="ainda aqui?",
        original_user_message="ainda aqui?",
        messages=[],
        effective_task_id="t1",
    )
    assert stale.retired is True
    assert stale.run_inputs == []
    assert built == ["fresh"]
    assert result["final_response"] == "hello again"
    assert _CURSOR_IDLE_NOTE in fresh.run_inputs[0]


def test_idle_note_injected_into_payload():
    text = _compose_cursor_user_input("ping", recycle_note=_CURSOR_IDLE_NOTE)
    assert _CURSOR_IDLE_NOTE in text
    assert "ping" in text
