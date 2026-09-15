"""Compatibility shim for databricks-gpt-oss-120b's harmony-style content.

Databricks' gpt-oss models can return `content` / `reasoning_content` /
`reasoning` on a chat-completions delta as a list of parts (e.g.
``[{"type": "reasoning", "text": "..."}, {"type": "text", "text": "..."}]``)
instead of a plain string, and the openai-agents SDK does not expect that:

- ``agents/models/chatcmpl_stream_handler.py`` accumulates the raw chunk
  into a plain ``str`` field via ``+=`` (``state.text_content_index_and_
  output[1].text += delta.content``). If ``delta.content`` is a list, this
  raises ``TypeError: can only concatenate str (not "list") to str`` and
  kills the whole stream generator immediately - no further text for that
  turn ever reaches the client, which is what rendered as an empty chat
  bubble in production.
- Even once that's fixed, the SDK also builds ``ResponseOutputText``/
  ``ResponseTextDeltaEvent`` with the same list-shaped fields when
  constructing the non-streaming response, which fails Pydantic validation.

``GptOssReasoningCompat.apply()`` patches all three sites to coerce these
fields to plain text before the SDK ever touches them.
"""

from agent_server.utils import coerce_content_to_text


class GptOssReasoningCompat:
    """Groups the openai-agents SDK patches needed for gpt-oss's list-shaped
    reasoning/content fields. Call `apply()` once at process startup."""

    _applied = False

    @classmethod
    def apply(cls) -> None:
        """Idempotently install all patches. Each patch is independently
        best-effort: if the SDK's internals have changed shape, that one
        patch is skipped rather than crashing startup."""
        if cls._applied:
            return
        cls._patch_response_output_text()
        cls._patch_response_text_delta_event()
        cls._patch_chatcmpl_stream_handler()
        cls._applied = True

    @staticmethod
    def _patch_response_output_text() -> None:
        """Coerce ResponseOutputText.text before Pydantic validates it.

        Only takes effect when the SDK builds this object via
        `ResponseOutputText(**data)` directly (calls `__init__`). Belt-and-
        suspenders alongside `_patch_chatcmpl_stream_handler`, which fixes
        the data at its source and makes this mostly a no-op in practice.
        """
        try:
            from openai.types.responses import ResponseOutputText
        except ImportError:
            return

        original_init = ResponseOutputText.__init__

        def patched_init(self, /, **data):
            if isinstance(data.get("text"), list):
                data["text"] = coerce_content_to_text(data["text"])
            original_init(self, **data)

        ResponseOutputText.__init__ = patched_init

    @staticmethod
    def _patch_response_text_delta_event() -> None:
        """Coerce ResponseTextDeltaEvent.delta before Pydantic validates it.

        Same caveat as `_patch_response_output_text` above.
        """
        try:
            from openai.types.responses import ResponseTextDeltaEvent
        except ImportError:
            return

        original_init = ResponseTextDeltaEvent.__init__

        def patched_init(self, /, **data):
            if isinstance(data.get("delta"), list):
                data["delta"] = coerce_content_to_text(data["delta"])
            original_init(self, **data)

        ResponseTextDeltaEvent.__init__ = patched_init

    @staticmethod
    def _patch_chatcmpl_stream_handler() -> None:
        """Coerce every chat-completion chunk's delta fields to plain text
        before ChatCmplStreamHandler.handle_stream ever sees them.

        This is the fix that actually matters for streaming: it's upstream
        of both the `+=` crash and the two Pydantic-validation patches
        above, so by the time either of those run, the data is already
        clean.
        """
        try:
            from agents.models.chatcmpl_stream_handler import ChatCmplStreamHandler
        except ImportError:
            return

        original_handle_stream = ChatCmplStreamHandler.handle_stream.__func__

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

        async def patched_handle_stream(cls, response, stream, *args, **kwargs):
            async for event in original_handle_stream(
                cls, response, coerce_chunks(stream), *args, **kwargs
            ):
                yield event

        ChatCmplStreamHandler.handle_stream = classmethod(patched_handle_stream)
