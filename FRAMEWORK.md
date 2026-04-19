# Orbit Wars Training Framework

This framework provides tools for training, evaluating, and benchmarking agents for the Orbit Wars Kaggle competition.

## Structure

```
├── model/                    # Model implementations
│   ├── heuristic_agent.py   # Heuristic agent with five strategies
│   └── (future RL models)
├── framework/               # Framework core
│   ├── __init__.py
│   ├── logger.py           # Logging system
│   ├── environment.py      # Environment wrapper
│   ├── agent.py            # Agent base classes
│   ├── trainer.py          # Training loop
│   └── evaluator.py        # Evaluation utilities
├── train.py                # Training script
├── test.py                 # Evaluation script
├── main.py                 # Kaggle submission entry point
├── requirements.txt        # Dependencies
└── log/                    # Log directory (auto-generated)
```

## Heuristic Agent Strategies

The heuristic agent implements five strategies:

1. **Priority Capture**: Targets nearest high-production neutral planets (production 3-5) with minimal ships (garrison + 1-2 ships)
2. **Defense Mechanism**: Detects incoming enemy fleets and recalls ships from nearby planets for defense
3. **Resource Management**: Keeps at least 20% of ships in home planet, never empties it completely
4. **Comet Utilization**: Captures comets immediately and withdraws ships before they leave the board
5. **Orbit Prediction**: Computes future positions of orbiting planets to avoid missed fleet attacks

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Test the Heuristic Agent

```bash
python train.py --agent heuristic --opponent random --num_episodes 5 --render_every 1
```

### Evaluate Multiple Agents

```bash
python test.py --agents heuristic random --num_episodes 10
```

### Kaggle Submission

The `main.py` file is ready for Kaggle submission. It imports and uses the heuristic agent from `model/heuristic_agent.py`.

To submit to Kaggle:

```bash
kaggle competitions submit orbit-wars -f main.py -m "Heuristic agent v1"
```

## Framework Components

### Logger

The logging system (`framework/logger.py`) provides:

- Console and file logging
- Timestamped log files in `log/` directory
- Separate `latest.log` for easy tailing
- Configurable log levels

Usage:

```python
from framework.logger import setup_logger, get_logger

logger = setup_logger(name="my_agent", log_dir="./logs", level=logging.INFO)
# or
logger = get_logger()  # Get default logger
```

### Environment Wrapper

`framework/environment.py` wraps the Kaggle environment for easier use:

```python
from framework.environment import OrbitWarsEnvironment

env = OrbitWarsEnvironment(debug=True)
results = env.run([agent1, agent2])
```

### Agent Base Class

`framework/agent.py` defines the `BaseAgent` abstract class:

```python
from framework.agent import BaseAgent

class MyAgent(BaseAgent):
    def act(self, observation):
        # Implement your strategy
        return moves
```

Pre-built agents:
- `HeuristicAgent`: The five-strategy heuristic agent
- `RandomAgent`: Random baseline agent

### Trainer

`framework/trainer.py` manages training episodes:

```python
from framework.trainer import Trainer

trainer = Trainer(env=env, agent=my_agent, opponent_agents=[opponent])
results = trainer.train(num_episodes=100)
```

### Evaluator

`framework/evaluator.py` benchmarks agents:

```python
from framework.evaluator import Evaluator

evaluator = Evaluator(env=env, agents=[agent1, agent2, agent3])
results = evaluator.evaluate_agent(agent_index=0, opponents=[1, 2])
```

## Adding New Agents

1. Create your agent in `model/` directory:

```python
# model/my_agent.py
from framework.agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self, player_id=0):
        super().__init__(player_id)
        self.name = "MyAgent"

    def act(self, observation):
        # Your implementation
        return []
```

2. Register it in `framework/agent.py`:

```python
from model.my_agent import MyAgent

def load_agent(agent_type="my_agent", **kwargs):
    if agent_type == "my_agent":
        return MyAgent(**kwargs)
    # ... existing code
```

3. Test it:

```bash
python train.py --agent my_agent --opponent random
```

## Logging and Debugging

- Logs are saved to `log/` directory with timestamps
- Use `--verbose` flag for detailed logging
- Use `--debug` flag for environment debug mode
- Check `log/latest.log` for the most recent run

## Performance Metrics

The framework tracks:
- Win rates against different opponents
- Average rewards per episode
- Number of steps per episode
- Agent-specific metrics (to be extended)

## Next Steps

1. Implement reinforcement learning agents (DQN, PPO, etc.)
2. Add neural network models in `model/` directory
3. Extend logging with TensorBoard support
4. Add hyperparameter tuning utilities
5. Implement self-play training

## Troubleshooting

**Import errors**: Make sure you're in the project root directory and have installed dependencies.

**Kaggle environment not found**: Install kaggle_environments from the competition repository.

**Agent crashes**: Check logs in `log/` directory for error details.

**Performance issues**: Reduce `--num_episodes` or disable rendering with `--render_every 0`.