"""Tests for UcConnectionProxyClient. No real Databricks/network calls -
httpx.MockTransport intercepts the request so we can assert on the URL and
headers actually sent."""

import asyncio
from unittest.mock import patch

import httpx

from agent_server.uc_connection_proxy import UcConnectionProxyClient


class _FakeConfig:
    host = "https://adb-123.cloud.databricks.com/"

    def authenticate(self):
        return {"Authorization": "Bearer fake-databricks-token"}


class _FakeWorkspaceClient:
    config = _FakeConfig()


def test_proxy_url_combines_host_and_sub_path():
    client = UcConnectionProxyClient("my-connection", _FakeWorkspaceClient())

    assert (
        client._proxy_url()
        == "https://adb-123.cloud.databricks.com/api/2.0/unity-catalog/connections/my-connection/proxy"
    )
    assert (
        client._proxy_url("/extra")
        == "https://adb-123.cloud.databricks.com/api/2.0/unity-catalog/connections/my-connection/proxy/extra"
    )


def test_request_sends_databricks_auth_header_to_proxy_url():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = request.content
        return httpx.Response(200, json={"status": "ok"})

    client = UcConnectionProxyClient("my-connection", _FakeWorkspaceClient())
    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = asyncio.run(client.request("POST", content=b"payload"))

    assert response.status_code == 200
    assert (
        captured["url"]
        == "https://adb-123.cloud.databricks.com/api/2.0/unity-catalog/connections/my-connection/proxy"
    )
    assert captured["authorization"] == "Bearer fake-databricks-token"
    assert captured["body"] == b"payload"
