import logging
from typing import AsyncGenerator, AsyncIterator, Optional
from uuid import uuid4

from agents.result import StreamEvent
from databricks.sdk import WorkspaceClient
from mlflow.genai.agent_server import get_request_headers
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentStreamEvent


def get_session_id(request: ResponsesAgentRequest) -> str | None:
    if request.context and request.context.conversation_id:
        return request.context.conversation_id
    if request.custom_inputs and isinstance(request.custom_inputs, dict):
        return request.custom_inputs.get("session_id")
    return None


def get_databricks_host(workspace_client: WorkspaceClient | None = None) -> Optional[str]:
    workspace_client = workspace_client or WorkspaceClient()
    try:
        return workspace_client.config.host
    except Exception as e:
        logging.exception(f"Error getting databricks host from env: {e}")
        return None


def build_mcp_url(path: str, workspace_client: WorkspaceClient | None = None) -> str:
    if not path.startswith("/"):
        return path
    hostname = get_databricks_host(workspace_client)
    return f"{hostname}{path}"


def get_user_workspace_client() -> WorkspaceClient:
    token = get_request_headers().get("x-forwarded-access-token")
    return WorkspaceClient(token=token, auth_type="pat")


def coerce_content_to_text(value):
    """Coerce a text/delta field that may be a list of content parts into a plain string.

    databricks-gpt-oss-120b returns content as a list of parts (e.g.
    ``[{"type": "reasoning", "text": "..."}, {"type": "text", "text": "..."}]``)
    instead of a plain string. Reasoning parts are dropped; everything else is
    concatenated. Non-list values are returned unchanged.
    """
    if not isinstance(value, list):
        return value
    parts = []
    for p in value:
        if isinstance(p, str):
            parts.append(p)
        elif isinstance(p, dict) and p.get("type") != "reasoning":
            t = coerce_content_to_text(p.get("text", ""))
            parts.append(t if isinstance(t, str) else str(t))
    return "".join(parts)


def _is_reasoning_event(event_data: dict) -> bool:
    """Check if a stream event is a reasoning/thinking event that should be filtered."""
    item = event_data.get("item", {})
    if isinstance(item, dict) and item.get("type") == "reasoning":
        return True
    if event_data.get("type", "").startswith("response.reasoning"):
        return True
    return False


def _fix_stream_text(event_data: dict) -> dict:
    """Ensure text fields in stream events are strings, not content-part lists."""
    if "delta" in event_data:
        event_data["delta"] = coerce_content_to_text(event_data["delta"])
    item = event_data.get("item")
    if isinstance(item, dict):
        if isinstance(item.get("text"), list):
            item["text"] = coerce_content_to_text(item["text"])
        if isinstance(item.get("content"), list):
            new_content = []
            for part in item["content"]:
                if isinstance(part, dict) and part.get("type") == "reasoning":
                    continue
                if isinstance(part, dict) and isinstance(part.get("text"), list):
                    part = {**part, "text": coerce_content_to_text(part["text"])}
                new_content.append(part)
            item["content"] = new_content
    return event_data


async def process_agent_stream_events(
    async_stream: AsyncIterator[StreamEvent],
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    curr_item_id = str(uuid4())
    async for event in async_stream:
        if event.type == "raw_response_event":
            event_data = event.data.model_dump()
            # Skip reasoning events
            if _is_reasoning_event(event_data):
                continue
            # Fix any text fields that are lists instead of strings
            event_data = _fix_stream_text(event_data)
            if event_data["type"] == "response.output_item.added":
                curr_item_id = str(uuid4())
                event_data["item"]["id"] = curr_item_id
            elif event_data.get("item") is not None and event_data["item"].get("id") is not None:
                event_data["item"]["id"] = curr_item_id
            elif event_data.get("item_id") is not None:
                event_data["item_id"] = curr_item_id
            yield event_data
        elif event.type == "run_item_stream_event" and event.item.type == "tool_call_output_item":
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=event.item.to_input_item(),
            )
