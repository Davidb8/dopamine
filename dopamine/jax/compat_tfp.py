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
"""Compatibility layer for tensorflow_probability functionality.

This module provides replacements for tensorflow_probability functionality
used in the JAX implementation, making it compatible with Python 3.13 which
doesn't support TensorFlow.

Currently, it only implements the minimal subset of functionality needed for
the JAX agents.
"""

import jax
import jax.numpy as jnp


class Categorical:
    """A simple categorical distribution implementation using JAX."""
    
    def __init__(self, logits=None, probs=None):
        """Initialize the categorical distribution.
        
        Args:
            logits: The unnormalized log probabilities.
            probs: The probabilities. Either logits or probs must be provided.
        """
        if logits is not None:
            self._logits = logits
            self._probs = jax.nn.softmax(logits, axis=-1)
        elif probs is not None:
            self._probs = probs
            self._logits = jnp.log(probs + 1e-8)  # Add small epsilon for numerical stability
        else:
            raise ValueError("Either 'logits' or 'probs' must be specified.")
    
    def sample(self, seed=None):
        """Sample from the distribution.
        
        Args:
            seed: Random seed.
            
        Returns:
            A sample from the categorical distribution.
        """
        if seed is None:
            seed = jax.random.PRNGKey(0)
        return jax.random.categorical(seed, self._logits)
    
    def log_prob(self, value):
        """Compute the log probability of a value.
        
        Args:
            value: The value to compute the log probability for.
            
        Returns:
            The log probability.
        """
        return jnp.take_along_axis(jnp.log(self._probs + 1e-8), value[..., None], axis=-1)[..., 0]
    
    def entropy(self):
        """Compute the entropy of the distribution.
        
        Returns:
            The entropy.
        """
        return -jnp.sum(self._probs * jnp.log(self._probs + 1e-8), axis=-1)


class Normal:
    """A simple normal distribution implementation using JAX."""
    
    def __init__(self, loc, scale):
        """Initialize the normal distribution.
        
        Args:
            loc: The mean of the distribution.
            scale: The standard deviation of the distribution.
        """
        self._loc = loc
        self._scale = scale
    
    def sample(self, seed=None):
        """Sample from the distribution.
        
        Args:
            seed: Random seed.
            
        Returns:
            A sample from the normal distribution.
        """
        if seed is None:
            seed = jax.random.PRNGKey(0)
        return self._loc + self._scale * jax.random.normal(seed, self._loc.shape)
    
    def log_prob(self, value):
        """Compute the log probability of a value.
        
        Args:
            value: The value to compute the log probability for.
            
        Returns:
            The log probability.
        """
        normalized = (value - self._loc) / self._scale
        return -0.5 * jnp.log(2 * jnp.pi) - jnp.log(self._scale) - 0.5 * normalized**2


class MultivariateNormalDiag:
    """A simple multivariate normal distribution with diagonal covariance implementation using JAX."""
    
    def __init__(self, loc, scale_diag):
        """Initialize the normal distribution.
        
        Args:
            loc: The mean of the distribution.
            scale_diag: The diagonal of the scale matrix.
        """
        self._loc = loc
        self._scale_diag = scale_diag
    
    def sample(self, seed=None):
        """Sample from the distribution.
        
        Args:
            seed: Random seed.
            
        Returns:
            A sample from the multivariate normal distribution.
        """
        if seed is None:
            seed = jax.random.PRNGKey(0)
        return self._loc + self._scale_diag * jax.random.normal(seed, self._loc.shape)
    
    def log_prob(self, value):
        """Compute the log probability of a value.
        
        Args:
            value: The value to compute the log probability for.
            
        Returns:
            The log probability.
        """
        normalized = (value - self._loc) / self._scale_diag
        return -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + 2 * jnp.log(self._scale_diag) + normalized**2, axis=-1)
    
    def entropy(self):
        """Compute the entropy of the distribution.
        
        Returns:
            The entropy.
        """
        # The entropy of a multivariate normal is:
        # 0.5 * log(det(2 * pi * e * Σ))
        # For diagonal covariance, det(Σ) = product(diag(Σ))
        k = self._loc.shape[-1]
        return 0.5 * (k * (1.0 + jnp.log(2 * jnp.pi)) + jnp.sum(jnp.log(self._scale_diag**2)))


class TransformedDistribution:
    """A stub implementation of TransformedDistribution - only minimally implemented."""
    
    def __init__(self, distribution, bijectors):
        """Initialize the transformed distribution.
        
        Args:
            distribution: The base distribution.
            bijectors: The bijectors to apply.
        """
        self._distribution = distribution
        self._bijectors = bijectors
    
    def sample(self, seed=None):
        """Sample from the distribution.
        
        Args:
            seed: Random seed.
            
        Returns:
            A sample from the transformed distribution.
        """
        # Simple passthrough in this minimal implementation
        return self._distribution.sample(seed)


# TensorFlow Probability compatibility namespace
class TFPDistributions:
    """A namespace for TensorFlow Probability distributions."""
    
    Categorical = Categorical
    Normal = Normal
    MultivariateNormalDiag = MultivariateNormalDiag
    TransformedDistribution = TransformedDistribution


# Export the compatibility layer as tfp
class TFPCompatJax:
    """A namespace for TensorFlow Probability JAX compatibility."""
    
    distributions = TFPDistributions()


# Public API for use as:
# from dopamine.jax.compat_tfp import jax as tfp
jax = TFPCompatJax() 