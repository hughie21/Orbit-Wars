"""
Kaggle submission entry point for Orbit Wars.
Uses MCTS agent with fast simulation and time-budgeted search.
"""

from model.mcts_agent import mcts_agent as agent

# The `agent` function is the entry point for Kaggle.
# `mcts_agent` is already compatible with the Kaggle environment interface.
