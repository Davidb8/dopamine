#!/usr/bin/env python
# coding=utf-8
"""Simple test script for JAX-only DQN agent."""

import os
import gymnasium as gym
import jax
import numpy as np
from dopamine.jax.agents.dqn import dqn_agent


def main():
    """Run a simple test of the DQN agent."""
    print("JAX devices:", jax.devices())
    
    # Create a simple environment
    env = gym.make('CartPole-v1')
    
    # Create the DQN agent
    agent = dqn_agent.JaxDQNAgent(
        num_actions=env.action_space.n,
        observation_shape=env.observation_space.shape,
        observation_dtype=jax.numpy.float32,
        stack_size=1,
        epsilon_train=0.1,
        epsilon_eval=0.05,
        epsilon_decay_period=1000,
        min_replay_history=500,
        update_period=4,
        target_update_period=100,
        summary_writer=os.path.join(os.getcwd(), 'logs')
    )
    
    print("Created DQN agent")
    
    # Run a simple test episode
    observation, info = env.reset()
    action = agent.begin_episode(observation)
    
    for _ in range(100):
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Episode ended")
            break
        action = agent.step(reward, observation)
    
    agent.end_episode(reward)
    
    print("Test completed successfully!")


if __name__ == '__main__':
    main() 