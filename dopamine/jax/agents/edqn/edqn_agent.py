# coding=utf-8
# Copyright 2023 The Dopamine Authors.
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
"""Compact implementation of an Enhanced DQN agent (N-step + PER) in Jax."""

import functools

from dopamine.jax import losses
from dopamine.jax import networks
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.replay_memory import accumulator
from dopamine.jax.replay_memory import replay_buffer
from dopamine.jax.replay_memory import samplers
from dopamine.metrics import statistics_instance
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tensorflow as tf


@functools.partial(jax.jit, static_argnums=(0, 3, 11, 12))
def train(
    network_def,
    online_params,
    target_params,
    optimizer,
    optimizer_state,
    states,
    actions,
    next_states,
    rewards,
    terminals,
    loss_weights,  # New argument for PER
    cumulative_gamma,
    loss_type='mse',
):
  """Run the training step with support for Prioritized Replay weights."""

  def loss_fn(params, target, weights):
    def q_online(state):
      return network_def.apply(params, state)

    q_values = jax.vmap(q_online)(states).q_values
    q_values = jnp.squeeze(q_values)
    replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)

    # Calculate element-wise loss (needed for priority updates)
    if loss_type == 'huber':
      elementwise_loss = jax.vmap(losses.huber_loss)(target, replay_chosen_q)
    else:  # Default to mse_loss
      elementwise_loss = jax.vmap(losses.mse_loss)(target, replay_chosen_q)

    # Weight the loss by PER weights for the gradient calculation
    mean_loss = jnp.mean(weights * elementwise_loss)
    return mean_loss, elementwise_loss  # Return element-wise loss too

  def q_target(state):
    return network_def.apply(target_params, state)

  target = dqn_agent.target_q(
      q_target, next_states, rewards, terminals, cumulative_gamma
  )
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

  # Get the weighted mean loss for gradient, and unweighted element-wise loss
  (mean_loss, elementwise_loss), grad = grad_fn(
      online_params, target, loss_weights
  )
  updates, optimizer_state = optimizer.update(
      grad, optimizer_state, params=online_params
  )
  online_params = optax.apply_updates(online_params, updates)

  # Return elementwise_loss for priority updates
  return optimizer_state, online_params, elementwise_loss, mean_loss


@gin.configurable
class JaxEnhancedDQNAgent(dqn_agent.JaxDQNAgent):
  """A JAX implementation of DQN with N-step updates and Prioritized Replay."""

  def __init__(
      self,
      num_actions,
      observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
      observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
      stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
      network=networks.NatureDQNNetwork,  # Standard DQN network
      gamma=0.99,
      update_horizon=3,  # N-step update horizon, default > 1
      min_replay_history=20000,
      update_period=4,
      target_update_period=8000,
      epsilon_fn=dqn_agent.linearly_decaying_epsilon,
      epsilon_train=0.01,
      epsilon_eval=0.001,
      epsilon_decay_period=250000,
      replay_scheme='prioritized',  # Default to prioritized replay
      optimizer='adam',
      seed=None,
      summary_writer=None,
      summary_writing_frequency=500,
      allow_partial_reload=False,
      loss_type='huber',  # Huber loss is common with PER
  ):
    """Initializes the agent and constructs the necessary components.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints or an int. If single int, the observation
        is assumed to be a 2D square.
      observation_dtype: DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to jnp.float32.
      stack_size: int, number of frames to use in state stack.
      network: flax.linen Module to use for the Q-network.
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters: (decay_period, step,
        warmup_steps, epsilon). This function should return the epsilon value
        used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      replay_scheme: str, 'prioritized' or 'uniform'. Defaults to prioritized.
      optimizer: str, name of optimizer to use.
      seed: int, a seed for the agent's internal RNG. If None, will use the
        current time in nanoseconds.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      allow_partial_reload: bool, whether we allow reloading a partial agent
        (for instance, only the network parameters).
      loss_type: str, 'mse' or 'huber', loss function to use.
    """
    # Set replay scheme before calling super init to build the correct buffer
    self._replay_scheme = replay_scheme
    self.loss_type = loss_type  # Store loss type

    # Use update_horizon > 1 for N-step learning.
    print(f'Creating JaxEnhancedDQNAgent with N-step = {update_horizon} '
          f'and replay scheme = {replay_scheme}')

    super().__init__(
        num_actions=num_actions,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        network=network,  # Pass the standard network
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        optimizer=optimizer,
        loss_type=loss_type,  # Pass loss type to super
        seed=seed,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency,
        allow_partial_reload=allow_partial_reload,
        # Note: _replay_scheme is set above and used in _build_replay_buffer
    )

  def _build_replay_buffer(self):
    """Creates the replay buffer used by the agent."""
    if self._replay_scheme not in ['uniform', 'prioritized']:
      raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))

    transition_accumulator = accumulator.TransitionAccumulator(
        stack_size=self.stack_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
    )

    if self._replay_scheme == 'prioritized':
      sampling_distribution = samplers.PrioritizedSamplingDistribution(
          seed=self._seed
      )
    else:  # Uniform sampling
      sampling_distribution = samplers.UniformSamplingDistribution(
          seed=self._seed
      )

    return replay_buffer.ReplayBuffer(
        transition_accumulator=transition_accumulator,
        sampling_distribution=sampling_distribution,
    )

  def _train_step(self):
    """Runs a single training step."""
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()

        if self._replay_scheme == 'prioritized':
          probs = self.replay_elements['sampling_probabilities']
          # Importance sampling weights (beta=0.5 in Rainbow paper, fixed here)
          # Note: Original PER paper uses beta annealed from 0.4 to 1.0.
          # We use a fixed beta=0.5 similar to the Rainbow simplification.
          loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
          loss_weights /= jnp.max(loss_weights)  # Normalize weights
        else:
          # Uniform sampling weights are all 1.
          batch_size = self.replay_elements['state'].shape[0]
          loss_weights = jnp.ones(batch_size)

        states = self.preprocess_fn(self.replay_elements['state'])
        next_states = self.preprocess_fn(self.replay_elements['next_state'])

        # Use the modified train function that handles loss_weights
        # and returns element-wise loss
        (
            self.optimizer_state,
            self.online_params,
            elementwise_loss,  # Unweighted element-wise losses
            mean_loss,  # Weighted mean loss for logging
        ) = train(
            self.network_def,
            self.online_params,
            self.target_network_params,
            self.optimizer,
            self.optimizer_state,
            states,
            self.replay_elements['action'],
            next_states,
            self.replay_elements['reward'],
            self.replay_elements['terminal'],
            loss_weights,  # Pass PER weights
            self.cumulative_gamma,
            self.loss_type,  # Pass loss type
        )

        if self._replay_scheme == 'prioritized':
          # Update priorities based on TD error magnitude (approximated by loss)
          # Using sqrt(loss) as in Rainbow for priority calculation (alpha=0.5)
          # Add epsilon to prevent 0 priorities
          priorities = jnp.sqrt(jnp.abs(elementwise_loss) + 1e-10)
          self._replay.update(self.replay_elements['indices'], priorities=priorities)

        if (
            self.summary_writer is not None
            and self.training_steps > 0
            and self.training_steps % self.summary_writing_frequency == 0
        ):
          loss_name = (
              'HuberLoss' if self.loss_type == 'huber' else 'MSELoss'
          )
          with self.summary_writer.as_default():
            tf.summary.scalar(
                loss_name, mean_loss, step=self.training_steps
            )
          self.summary_writer.flush()
          if hasattr(self, 'collector_dispatcher'):
            self.collector_dispatcher.write(
                [
                    statistics_instance.StatisticsInstance(
                        'Loss', onp.asarray(mean_loss), step=self.training_steps
                    ),
                ],
                collector_allowlist=self._collector_allowlist,
            )

      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1

  # We inherit begin_episode, step, end_episode, _store_transition etc. from
  # JaxDQNAgent. The _store_transition method in JaxDQNAgent already handles
  # the priority argument needed for PER if the replay buffer supports it.