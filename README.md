# JAX-only Dopamine

This is a fork of [Google Dopamine](https://github.com/google/dopamine) that has been modified to use only the JAX implementation of the agents. All TensorFlow dependencies have been removed, making it compatible with Python 3.13.

## Installation

```bash
pip install -e .
```

## What's Changed

- Removed all TensorFlow dependencies
- Created compatibility layers for TensorFlow and TensorFlow Probability functionality
- Focused only on the JAX implementation of agents
- Updated requirements.txt and setup.py to include only the necessary dependencies

## Usage

You can use the JAX agents directly:

```python
from dopamine.jax.agents.dqn import dqn_agent

# Create a DQN agent
agent = dqn_agent.JaxDQNAgent(num_actions=4)
```

## Available Agents

The following JAX-based agents are available:

- DQN
- Rainbow
- Quantile
- Implicit Quantile
- Full Rainbow
- SAC
- PPO
- EDQN

## Dependencies

The main dependencies are:

- JAX and JAXlib
- Flax
- Optax
- Gymnasium

## Notes

This fork is designed to be minimal and focused only on the JAX implementation. If you need the TensorFlow-based agents or additional functionality from the original Dopamine library, please use the [official version](https://github.com/google/dopamine) instead. 