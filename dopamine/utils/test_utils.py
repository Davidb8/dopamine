# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test utilities for Dopamine."""

import numpy as np


def update_terminal_episodes(episodes_list, terminal_list):
  """Updates the list of episodes with terminated episodes."""
  for i, terminal_value in enumerate(terminal_list):
    if not terminal_value:
      episodes_list[i] = []


def get_mock_replay_elements(replay_capacity, observation_shape, stack_size,
                             batch_size, update_horizon):
  """Creates mock replay elements for testing."""
  states = np.random.randint(
      0, 256, size=[replay_capacity] + list(observation_shape))
  next_states = np.random.randint(
      0, 256, size=[replay_capacity] + list(observation_shape))
  actions = np.random.randint(0, 4, size=replay_capacity)
  rewards = np.random.random(size=replay_capacity)
  terminals = np.random.random(size=replay_capacity) > 0.9
  
  return states, actions, rewards, next_states, terminals 