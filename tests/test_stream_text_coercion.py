"""Regression tests for the empty-chat-bubble bug.

``databricks-gpt-oss-120b`` can return text/delta fields as a list of content
parts (reasoning + text) instead of a plain string, e.g.
``[{"type": "reasoning", "text": "..."}, {"type": "text", "text": "..."}]``.

``process_agent_stream_events`` used to assume ``delta`` was a dict wrapping
a nested ``"text"`` list (``_fix_stream_text`` checked
``isinstance(delta, dict)``), while the actual shape is a flat list. That
mismatch meant the coercion never ran, so a raw list reached
``ResponsesAgentStreamEvent.delta`` -- which the chat UI expects to be a
string -- and bubbles rendered empty.

These tests lock in the fix: reasoning parts are dropped, real text parts
are concatenated into a plain string, and the transformation is applied to
every shape it needs to run on (top-level deltas, item.text, and
item.content[*].text).
"""

import asyncio

from agent_server.utils import coerce_content_to_text, process_agent_stream_events


def test_coerce_content_to_text_drops_pure_reasoning():
    assert coerce_content_to_text([{"type": "reasoning", "text": "internal thought"}]) == ""


def test_coerce_content_to_text_keeps_real_text_and_strips_reasoning():
    mixed = [
        {"type": "reasoning", "text": "internal thought"},
        {"type": "text", "text": "Hello "},
        {"type": "text", "text": "world"},
    ]
    assert coerce_content_to_text(mixed) == "Hello world"


def test_coerce_content_to_text_passes_through_plain_strings():
    assert coerce_content_to_text("already a string") == "already a string"


def test_coerce_content_to_text_handles_nested_list_text():
    assert coerce_content_to_text([{"type": "text", "text": ["Hel", "lo"]}]) == "Hello"


class _FakeData:
    """Mimics a pydantic model instance: .model_dump() returns the raw dict."""

    def __init__(self, d):
        self._d = d

    def model_dump(self):
        return self._d


class _FakeEvent:
    def __init__(self, type_, data):
        self.type = type_
        self.data = data


async def _fake_gpt_oss_stream():
    yield _FakeEvent(
        "raw_response_event",
        _FakeData({"type": "response.output_item.added", "item": {"type": "message", "content": []}}),
    )
    yield _FakeEvent(
        "raw_response_event",
        _FakeData(
            {
                "type": "response.output_text.delta",
                "item_id": "placeholder",
                "delta": [
                    {"type": "reasoning", "text": "thinking about the answer..."},
                    {"type": "text", "text": "Hallo, "},
                ],
            }
        ),
    )
    yield _FakeEvent(
        "raw_response_event",
        _FakeData(
            {
                "type": "response.output_text.delta",
                "item_id": "placeholder",
                "delta": [{"type": "text", "text": "wie kann ich helfen?"}],
            }
        ),
    )
    yield _FakeEvent(
        "raw_response_event",
        _FakeData(
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "content": [
                        {"type": "reasoning", "text": "thinking..."},
                        {"type": "text", "text": "Hallo, wie kann ich helfen?"},
                    ],
                },
            }
        ),
    )


def test_process_agent_stream_events_reconstructs_bubble_text():
    async def collect():
        deltas = []
        done_item = None
        async for event_data in process_agent_stream_events(_fake_gpt_oss_stream()):
            assert isinstance(event_data, dict)
            if event_data["type"] == "response.output_text.delta":
                deltas.append(event_data["delta"])
            elif event_data["type"] == "response.output_item.done":
                done_item = event_data["item"]
        return deltas, done_item

    deltas, done_item = asyncio.run(collect())

    assert "".join(deltas) == "Hallo, wie kann ich helfen?"
    assert done_item["content"] == [{"type": "text", "text": "Hallo, wie kann ich helfen?"}]
