"""Client for calling an external HTTP endpoint through a Unity Catalog
Connection's Databricks-managed proxy.

Databricks holds the connection's external-service credentials and injects
them into the forwarded request - this client only ever needs a Databricks
token, never the external service's own credentials.

URL shape: `{databricks_host}/api/2.0/unity-catalog/connections/{name}/proxy[/<sub-path>]`,
which Databricks forwards to `{connection.host}{connection.base_path}{sub-path}`.
See: https://docs.databricks.com/aws/en/query-federation/http#proxy
"""

import httpx
from databricks.sdk import WorkspaceClient


class UcConnectionProxyClient:
    """Calls an external HTTP endpoint through a named Unity Catalog
    Connection's proxy."""

    def __init__(self, connection_name: str, workspace_client: WorkspaceClient):
        self._connection_name = connection_name
        self._workspace_client = workspace_client

    def _proxy_url(self, sub_path: str = "") -> str:
        host = self._workspace_client.config.host.rstrip("/")
        return f"{host}/api/2.0/unity-catalog/connections/{self._connection_name}/proxy{sub_path}"

    async def request(self, method: str, sub_path: str = "", **kwargs) -> httpx.Response:
        headers = self._workspace_client.config.authenticate()
        async with httpx.AsyncClient() as client:
            return await client.request(method, self._proxy_url(sub_path), headers=headers, **kwargs)
