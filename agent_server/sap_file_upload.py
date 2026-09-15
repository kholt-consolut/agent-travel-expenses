"""Uploads files to SAP's temporary file staging endpoint.

Goes through the `travel-expense-mcp-dev-file-upload` UC Connection's proxy
(see uc_connection_proxy.py) - Databricks injects the SAP credentials, this
code never handles them directly.

SAP's endpoint expects a JSON body: {"file_name": str, "file_content": str}
where file_content is base64-encoded.
"""

import base64

import httpx
from databricks.sdk import WorkspaceClient

from agent_server.uc_connection_proxy import UcConnectionProxyClient

UPLOAD_CONNECTION_NAME = "travel-expense-mcp-dev-file-upload"


class SapFileUploader:
    def __init__(self, proxy_client: UcConnectionProxyClient):
        self._proxy_client = proxy_client

    @classmethod
    def for_default_connection(cls, workspace_client: WorkspaceClient) -> "SapFileUploader":
        return cls(UcConnectionProxyClient(UPLOAD_CONNECTION_NAME, workspace_client))

    async def upload(self, file_name: str, content: bytes) -> httpx.Response:
        payload = {
            "file_name": file_name,
            "file_content": base64.b64encode(content).decode("ascii"),
        }
        return await self._proxy_client.request("POST", json=payload)
