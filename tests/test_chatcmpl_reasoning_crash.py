"""Regression test for the streaming crash that produced empty chat bubbles.

databricks-gpt-oss-120b streams reasoning + answer text together inside
``delta.content`` as a list of parts (harmony-style), e.g.
``[{"type": "reasoning", ...}, {"type": "text", "text": "..."}]``, instead of
a plain string.

The openai-agents SDK builds its streaming ``ChoiceDelta`` objects via
``model_construct()`` (bypassing pydantic validation -- confirmed below:
normal construction rejects a list ``content`` outright), then in
``agents/models/chatcmpl_stream_handler.py`` does
``state.text_content_index_and_output[1].text += delta.content``. Since
``delta.content`` is a list, not a string, that raises
``TypeError: can only concatenate str (not "list") to str`` and kills the
whole stream generator immediately -- no further text for that turn ever
reaches the UI, which is what rendered as an empty bubble in production
(see the traceback this test's "before fix" case reproduces).

agent_server.agent patches ``ChatCmplStreamHandler.handle_stream`` to coerce
every chunk's ``delta.content``/``reasoning_content``/``reasoning`` to plain
text before the SDK's own handler ever sees them. This test proves the crash
exists without that patch and that the patch fixes it, using a raw chunk
stream built with ``model_construct()`` the same way the SDK builds it in
production.
"""

import asyncio

import pytest
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice, ChoiceDelta
from openai.types.responses import Response

from agent_server.utils import coerce_content_to_text


def _make_chunk(content):
    delta = ChoiceDelta.model_construct(content=content, role="assistant")
    choice = Choice.model_construct(index=0, delta=delta, finish_reason=None, logprobs=None)
    return ChatCompletionChunk.model_construct(
        id="chatcmpl-test",
        choices=[choice],
        created=0,
        model="databricks-gpt-oss-120b",
        object="chat.completion.chunk",
        system_fingerprint=None,
        usage=None,
    )


async def _fake_gpt_oss_raw_stream():
    # First chunk is pure reasoning -- the exact shape from the production
    # crash's pydantic serialization warning: [{"type": "reasoning", ...}]
    yield _make_chunk([{"type": "reasoning", "text": "the user just says \"Hallo\", greet back"}])
    yield _make_chunk([{"type": "text", "text": "Hallo, "}])
    yield _make_chunk([{"type": "text", "text": "wie kann ich helfen?"}])
    final = _make_chunk(None)
    final.choices[0].finish_reason = "stop"
    yield final


def _fake_response():
    return Response.model_construct(
        id="resp-test",
        created_at=0,
        model="databricks-gpt-oss-120b",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def test_model_construct_bypasses_validation_matching_production():
    """Sanity check the premise: normal construction rejects list content,
    proving the SDK must use model_construct() for it to reach delta.content
    as a list at runtime (as the production crash demonstrated)."""
    with pytest.raises(Exception):
        ChoiceDelta(content=[{"type": "reasoning", "text": "x"}])

    delta = ChoiceDelta.model_construct(content=[{"type": "reasoning", "text": "x"}])
    assert isinstance(delta.content, list)


def test_raw_sdk_crashes_on_list_content():
    from agents.models.chatcmpl_stream_handler import ChatCmplStreamHandler

    async def run():
        async for _event in ChatCmplStreamHandler.handle_stream(_fake_response(), _fake_gpt_oss_raw_stream()):
            pass

    with pytest.raises((TypeError, Exception)):
        asyncio.run(run())


def test_patched_handle_stream_does_not_crash_and_reconstructs_text():
    from agents.models.chatcmpl_stream_handler import ChatCmplStreamHandler

    orig_handle_stream = ChatCmplStreamHandler.handle_stream

    async def coerce_chunks(stream):
        async for chunk in stream:
            for choice in getattr(chunk, "choices", None) or []:
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                for field in ("content", "reasoning_content", "reasoning"):
                    value = getattr(delta, field, None)
                    if isinstance(value, list):
                        setattr(delta, field, coerce_content_to_text(value))
            yield chunk

    orig_func = orig_handle_stream.__func__

    async def patched_handle_stream(cls, response, stream, *args, **kwargs):
        async for event in orig_func(cls, response, coerce_chunks(stream), *args, **kwargs):
            yield event

    ChatCmplStreamHandler.handle_stream = classmethod(patched_handle_stream)
    try:
        async def run():
            deltas = []
            async for event in ChatCmplStreamHandler.handle_stream(_fake_response(), _fake_gpt_oss_raw_stream()):
                if type(event).__name__ == "ResponseTextDeltaEvent":
                    deltas.append(event.delta)
            return deltas

        deltas = asyncio.run(run())
        assert "".join(deltas) == "Hallo, wie kann ich helfen?"
    finally:
        ChatCmplStreamHandler.handle_stream = orig_handle_stream
