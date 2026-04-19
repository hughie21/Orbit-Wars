"""
Orbit Wars - Heuristic Agent
Implements the five strategies described in the requirements.

Strategy:
1. Prioritize capturing high-production neutral planets (production 3-5) with minimal ships
2. Defense mechanism: detect incoming enemy fleets and recall ships from nearby planets
3. Resource management: keep at least 20% ships in home planet
4. Comet utilization: capture comets immediately, withdraw ships before they leave
5. Orbit prediction: compute future positions of orbiting planets to avoid missed fleets

This agent uses the heuristic implementation from model.heuristic_agent.
"""

import sys
import os

# Add model directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import heuristic_agent


def agent(obs):
    """
    Agent function compatible with Kaggle interface.
    Simply delegates to the heuristic agent implementation.
    """
    return heuristic_agent(obs)


# For local testing
if __name__ == "__main__":
    # Simple test to verify the agent loads
    print("Heuristic agent loaded successfully")
    print("Agent function signature:", agent)

