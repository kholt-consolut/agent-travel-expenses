"""Dry-run the agent + UC-connection MCP tool directly, without MLflow's
AgentServer/tracking machinery (mlflow.genai.agent_server.AgentServer and
setup_mlflow_git_based_version_tracking() aren't needed to exercise the
agent - they're what start_server.py adds on top for the deployed HTTP
server). Not part of the deployed app; for local verification only.

Reuses the real agent config (SYSTEM_PROMPT, MODEL, MCP_SERVERS,
create_agent, init_mcp_servers) from agent_server.agent so this stays in
sync with production - only the MLflow server/tracking wrapper is skipped.

Run from the project root:
    .venv\\Scripts\\python.exe dryrun\\run_agent.py "Hallo, wer bist du?"
"""

import asyncio
import sys
from pathlib import Path

from agents import Runner
from agents.mcp import MCPServerManager
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

# Local-only workaround, not part of the deployed app: this machine's network
# can't reach the Databricks SDK's `/.well-known/databricks-config` discovery
# probe, which otherwise blocks every WorkspaceClient() construction for
# minutes. See databricks_sdk_compat.py.
from databricks_sdk_compat import NoDiscoveryProbeCompat  # noqa: E402

NoDiscoveryProbeCompat.apply()

from agent_server.agent import create_agent, init_mcp_servers  # noqa: E402


async def main(message: str) -> None:
    mcp_servers = init_mcp_servers()
    async with MCPServerManager(servers=mcp_servers, connect_in_parallel=True) as manager:
        print(f"MCP servers active: {[s.name for s in manager.active_servers]}")

        agent = create_agent(manager.active_servers)
        result = await Runner.run(agent, [{"role": "user", "content": message}])

        for item in result.new_items:
            print(item.to_input_item())


if __name__ == "__main__":
    message = sys.argv[1] if len(sys.argv) > 1 else "Hallo, wer bist du?"
    asyncio.run(main(message))
