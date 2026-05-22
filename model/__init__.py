from .heuristic_agent import heuristic_agent
from .mcts_agent import mcts_agent, MCTSAgent

__all__ = [
    "heuristic_agent", "mcts_agent", "ppo_agent", "strategic_agent",
    "opponent_model", "value_network",
]


def __getattr__(name):
    if name == "strategic_agent":
        from .strategic_agent import strategic_agent as _mod
        return _mod
    if name == "opponent_model":
        from .opponent_model import OpponentModel as _mod
        return _mod
    if name == "value_network":
        from .value_network import ValueNetwork as _mod
        return _mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
