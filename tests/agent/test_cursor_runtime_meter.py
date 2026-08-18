"""Cursor runtime context meter + session recycle (#1, #2)."""

from types import SimpleNamespace

from agent.cursor_runtime import (
    CURSOR_RECYCLE_RATIO,
    _CURSOR_RECYCLE_NOTE,
    _compose_cursor_user_input,
    _record_cursor_usage,
    cursor_meter_prompt_tokens,
    cursor_should_recycle_session,
    cursor_window_occupancy,
)


class _FakeCompressor:
    def __init__(self, context_length: int, last_prompt_tokens: int = 0):
        self.context_length = context_length
        self.last_prompt_tokens = last_prompt_tokens
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.updates = []
        self.parked = []

    def update_from_response(self, usage):
        self.updates.append(usage)
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def note_external_compaction(self, *, kind="completed", summary=""):
        self.parked.append((kind, summary))
        self.last_prompt_tokens = -1


def _agent(context_length=500_000):
    return SimpleNamespace(
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
        session_id=None,
        model="grok-4.6",
        provider="cursor",
        base_url="https://api.cursor.com",
        api_key="",
        context_compressor=_FakeCompressor(context_length),
    )


def test_occupancy_excludes_cache_write():
    usage = {
        "input_tokens": 200_000,
        "cache_read_tokens": 100_000,
        "cache_write_tokens": 525_000,
        "output_tokens": 10,
    }
    assert cursor_window_occupancy(usage) == 300_000
    assert cursor_meter_prompt_tokens(usage, 500_000) == 300_000


def test_occupancy_above_window_is_untrustworthy():
    usage = {
        "input_tokens": 400_000,
        "cache_read_tokens": 425_000,  # 825K > 500K
        "cache_write_tokens": 0,
    }
    assert cursor_meter_prompt_tokens(usage, 500_000) is None
    assert cursor_should_recycle_session(825_000, 500_000) is False


def test_recycle_in_high_band_only():
    assert cursor_should_recycle_session(100_000, 500_000) is False
    assert cursor_should_recycle_session(
        int(500_000 * CURSOR_RECYCLE_RATIO), 500_000
    ) is True
    assert cursor_should_recycle_session(500_000, 500_000) is True
    assert cursor_should_recycle_session(500_001, 500_000) is False


def test_record_usage_paints_occupancy_not_billing_prompt():
    agent = _agent(500_000)
    turn = SimpleNamespace(
        token_usage_last={
            "input_tokens": 200_000,
            "cache_read_tokens": 50_000,
            "cache_write_tokens": 400_000,
            "output_tokens": 20,
            "total_tokens": 670_000,
        },
        token_usage_total=None,
    )
    result = _record_cursor_usage(agent, turn)
    assert result["last_prompt_tokens"] == 250_000
    assert agent.context_compressor.last_prompt_tokens == 250_000
    assert agent.context_compressor.updates[-1]["prompt_tokens"] == 250_000
    assert "cursor_recycle" not in result


def test_record_usage_parks_meter_on_impossible_reading():
    agent = _agent(256_000)  # the grok-4 catch-all that grok-4.6 used to hit
    turn = SimpleNamespace(
        token_usage_last={
            "input_tokens": 400_000,
            "cache_read_tokens": 425_000,
            "cache_write_tokens": 0,
            "output_tokens": 10,
            "total_tokens": 825_000,
        },
        token_usage_total=None,
    )
    result = _record_cursor_usage(agent, turn)
    assert "last_prompt_tokens" not in result
    assert agent.context_compressor.last_prompt_tokens == -1
    assert agent.context_compressor.parked
    assert "cursor_recycle" not in result


def test_record_usage_flags_recycle_near_window():
    agent = _agent(500_000)
    turn = SimpleNamespace(
        token_usage_last={
            "input_tokens": 200_000,
            "cache_read_tokens": 250_000,  # 450K / 500K = 90%
            "cache_write_tokens": 0,
            "output_tokens": 10,
            "total_tokens": 460_000,
        },
        token_usage_total=None,
    )
    result = _record_cursor_usage(agent, turn)
    assert result["last_prompt_tokens"] == 450_000
    assert result["cursor_recycle"] is True


def test_recycle_note_injected_into_outbound_payload():
    text = _compose_cursor_user_input(
        "continue",
        recycle_note=_CURSOR_RECYCLE_NOTE,
    )
    assert _CURSOR_RECYCLE_NOTE in text
    assert "continue" in text
