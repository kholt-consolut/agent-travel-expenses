"""Compatibility shim that skips the Databricks SDK's discovery probe.

`databricks.sdk.config.Config.__init__` always calls `_resolve_host_metadata`,
which probes the workspace's `/.well-known/databricks-config` endpoint to
auto-discover `account_id` / `workspace_id` / `discovery_url` (used for OIDC
token exchange). It's best-effort and falls back to explicit configuration
on failure - but on this project's workspace that probe never gets a
response for unauthenticated automated clients (works fine in a browser,
likely a WAF/bot-protection rule upstream - not something fixable here), and
the SDK's own default retry budget is 300s with at least two attempts, so a
single unresponsive probe blocks for up to ~10 minutes.

This project authenticates with an explicit host + PAT (see `.env`:
`DATABRICKS_HOST` / `DATABRICKS_TOKEN`) and never needs the discovered
fields - PAT auth doesn't use OIDC token exchange. So instead of merely
bounding the probe's timeout, skip it outright.

This blocks every `Config()`/`WorkspaceClient()` construction across the
process - including ones this project doesn't call directly, such as
MLflow's own Databricks auth resolution during `mlflow.openai.autolog()` /
`setup_mlflow_git_based_version_tracking()`. Patching only our own call
sites wouldn't cover those.
"""

from databricks.sdk.config import Config


class NoDiscoveryProbeCompat:
    """Disables Config's `/.well-known/databricks-config` auto-discovery
    process-wide. Call `apply()` once, as early as possible - before
    anything else constructs a WorkspaceClient or Config, including
    third-party code."""

    _applied = False

    @classmethod
    def apply(cls) -> None:
        if cls._applied:
            return
        Config._resolve_host_metadata = lambda self: None
        cls._applied = True
