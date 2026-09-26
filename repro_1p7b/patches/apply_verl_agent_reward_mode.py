"""Align the adapter reward-mode default with the frozen reward API."""

from pathlib import Path

path = Path(
    "/home/u2024311031/verl-agent/agent_system/environments/"
    "env_package/envfactory/official_envs.py"
)
text = path.read_text()
old = 'self.reward_mode = os.getenv("ENVFACTORY_REWARD_MODE", "graph")'
new = 'self.reward_mode = os.getenv("ENVFACTORY_REWARD_MODE", "graph_frontier")'
if text.count(old) != 1:
    raise RuntimeError(f"expected exactly one reward-mode default, got {text.count(old)}")
temporary = path.with_suffix(".py.tmp")
temporary.write_text(text.replace(old, new, 1))
temporary.replace(path)
print(path)
