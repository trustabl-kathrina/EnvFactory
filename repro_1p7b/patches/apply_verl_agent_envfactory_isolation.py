"""Make generated EnvFactory rollouts process-isolated in verl-agent.

FastMCP ``base_client.new()`` creates a new protocol session over the same
Python server process.  EnvFactory tools store scenario state in module
globals, so sessions are not an isolation boundary.  This adapter-only patch
gives every GRPO rollout/server pair its own FastMCP Client transport/process.
"""

from pathlib import Path

PATH = Path(
    "/home/u2024311031/verl-agent/agent_system/environments/"
    "env_package/envfactory/official_envs.py"
)


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match, got {count}: {old[:100]!r}")
    return text.replace(old, new, 1)


text = PATH.read_text()
text = replace_once(
    text,
    '''        self.states = []
        self.registered = set()
        self.reset_serial = 0
''',
    '''        self.states = []
        self.registered = set()
        self.isolated_clients = {}
        self.reset_serial = 0
''',
)
text = replace_once(
    text,
    '''    def _save(self, state):
        result = {}
        for server, client_id in state["clients"].items():
            value = self.MCPManager.call_tool(client_id, "save_scenario", {})
            result[server] = _json(value)
        return result

    def _call_typed(self, client_id, tool_name, arguments):
        """Call at the FastMCP boundary before core code flattens content."""
        async def invoke():
            client, _ = self.MCPManager.get_client(client_id)
            short_name = tool_name.split("-", 1)[-1]
            server = client_id.split("-", 1)[0]
            if server in self.MCPManager.stateless_clients:
                async with self.MCPManager._stateless_lock:
                    return await client.call_tool(short_name, arguments)
            async with client:
                return await client.call_tool(short_name, arguments)

        future = asyncio.run_coroutine_threadsafe(invoke(), self.MCPManager._loop)
        return future.result(timeout=30)
''',
    '''    @staticmethod
    def _scenario_preserves_explicit(expected, actual):
        """Require every explicit input field while allowing schema defaults."""
        if isinstance(expected, dict):
            return (
                isinstance(actual, dict)
                and all(
                    key in actual
                    and EnvFactoryBatchEnv._scenario_preserves_explicit(value, actual[key])
                    for key, value in expected.items()
                )
            )
        if isinstance(expected, list):
            return isinstance(actual, list) and len(expected) == len(actual) and all(
                EnvFactoryBatchEnv._scenario_preserves_explicit(left, right)
                for left, right in zip(expected, actual)
            )
        return expected == actual

    def _open_isolated(self, server, client_id, scenario):
        """Start a dedicated tool-server process for one rollout/server pair."""
        from fastmcp import Client

        tool_path = self.root / "envs/tools" / f"{server}.py"

        async def open_and_load():
            client = Client(str(tool_path))
            await client._connect()
            await client.call_tool("load_scenario", {"scenario": scenario})
            return client

        future = asyncio.run_coroutine_threadsafe(open_and_load(), self.MCPManager._loop)
        client = future.result(timeout=120)
        self.isolated_clients[client_id] = client
        saved = self._save_one(client_id)
        if not self._scenario_preserves_explicit(scenario, saved):
            from repro_1p7b.graph_frontier.fastmcp_adapter import fastmcp_execution_success

            loaded = self._call_typed(
                client_id, "load_scenario", {"scenario": saved}
            )
            canonical = self._save_one(client_id)
            if fastmcp_execution_success(loaded) is False or canonical != saved:
                self._close_isolated(client_id)
                raise RuntimeError(
                    f"isolated scenario canonicalization failed: {client_id}"
                )

    def _close_isolated(self, client_id):
        client = self.isolated_clients.pop(client_id, None)
        if client is None:
            return

        async def close_client():
            await client.close()

        future = asyncio.run_coroutine_threadsafe(close_client(), self.MCPManager._loop)
        future.result(timeout=30)

    def _save_one(self, client_id):
        from repro_1p7b.graph_frontier.fastmcp_adapter import adapt_fastmcp_result

        raw = self._call_typed(client_id, "save_scenario", {})
        fields, success = adapt_fastmcp_result(raw)
        if success is False or fields == UNKNOWN:
            raise RuntimeError(f"typed save_scenario failed: {client_id}")
        if isinstance(fields, dict) and set(fields) == {"result"}:
            fields = fields["result"]
        return _json(fields)

    def _save(self, state):
        return {
            server: self._save_one(client_id)
            for server, client_id in state["clients"].items()
        }

    def _call_typed(self, client_id, tool_name, arguments):
        """Call a rollout-private FastMCP process before text flattening."""
        async def invoke():
            client = self.isolated_clients[client_id]
            short_name = tool_name.split("-", 1)[-1]
            return await client.call_tool(short_name, arguments)

        future = asyncio.run_coroutine_threadsafe(invoke(), self.MCPManager._loop)
        return future.result(timeout=30)
''',
)
text = replace_once(
    text,
    '''                scenario = item["initial_config"].get(server, {})
                self.MCPManager.load_scenario(client_id, scenario, check=True)
                clients[server] = client_id
''',
    '''                scenario = item["initial_config"].get(server, {})
                self._open_isolated(server, client_id, scenario)
                clients[server] = client_id
''',
)
text = replace_once(
    text,
    '''        for client_id in state["clients"].values():
            self.MCPManager.close_client(client_id)
''',
    '''        for client_id in state["clients"].values():
            self._close_isolated(client_id)
''',
)
text = replace_once(
    text,
    '''                try:
                    self.MCPManager.close_client(client_id)
                except Exception:
                    pass
''',
    '''                try:
                    self._close_isolated(client_id)
                except Exception:
                    pass
''',
)
temporary = PATH.with_suffix(".py.tmp")
temporary.write_text(text)
temporary.replace(PATH)
print(PATH)
