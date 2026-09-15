"""Tests for SapFileUploader. No real network - httpx.MockTransport
intercepts the request to assert on the JSON payload actually sent."""

import asyncio
import base64
import json
from unittest.mock import patch

import httpx

from agent_server.sap_file_upload import UPLOAD_CONNECTION_NAME, SapFileUploader
from agent_server.uc_connection_proxy import UcConnectionProxyClient


class _FakeConfig:
    host = "https://adb-123.cloud.databricks.com"

    def authenticate(self):
        return {"Authorization": "Bearer fake-databricks-token"}


class _FakeWorkspaceClient:
    config = _FakeConfig()


def test_upload_sends_base64_content_and_file_name_as_json():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "ok"})

    proxy_client = UcConnectionProxyClient(UPLOAD_CONNECTION_NAME, _FakeWorkspaceClient())
    uploader = SapFileUploader(proxy_client)
    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = asyncio.run(uploader.upload("receipt.pdf", b"pdf-bytes-here"))

    assert response.status_code == 200
    assert (
        captured["url"]
        == "https://adb-123.cloud.databricks.com/api/2.0/unity-catalog/connections/"
        f"{UPLOAD_CONNECTION_NAME}/proxy"
    )
    assert captured["body"]["file_name"] == "receipt.pdf"
    assert base64.b64decode(captured["body"]["file_content"]) == b"pdf-bytes-here"


def test_for_default_connection_uses_upload_connection_name():
    uploader = SapFileUploader.for_default_connection(_FakeWorkspaceClient())

    assert uploader._proxy_client._connection_name == UPLOAD_CONNECTION_NAME
