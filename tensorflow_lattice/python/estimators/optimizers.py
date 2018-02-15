# Copyright 2018 The TensorFlow Lattice Authors.
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
# ==============================================================================
"""Optimizer helper functions."""
from tensorflow.python.training import adagrad
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import learning_rate_decay
from tensorflow.python.training import training_util


def gradient_descent_polynomial_decay(
        decay_steps=10000,
        end_learning_rate=0.0001,
        power=0.5,
        cycle=False,
        name=None):
  """Returns a gradient descent optimizer function with polynomial_decay.

  See tesnorflow.training.learning_rate_decay.polynomial_decay how to set the
  argument. This function returns a python callable that sets gradient descent
  optimizer with a polynomialy decaying learning rate for tensorflow lattice
  estimator training.

  Args:
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
      Must be positive.  See the decay computation above.
    end_learning_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The minimal end learning rate.
    power: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The power of the polynomial. Defaults to sqrt, 0.5.
    cycle: A boolean, whether or not it should cycle beyond decay_steps.
    name: String.  Optional name of the operation. Defaults to
      'PolynomialDecay'.

  Returns:
    A python callable that accepts learning_rate and sets a gradient descent
    optimizer with a polynomialy decaying learning rate for tensorflow lattice
    estimator training.
  """
  def optimizer_fn(learning_rate=0.01):
      global_step_tensor = training_util.get_or_create_global_step()
      learning_rate = learning_rate_decay.polynomial_decay(
              learning_rate=learning_rate,
              global_step=global_step_tensor,
              decay_steps=decay_steps,
              end_learning_rate=end_learning_rate,
              power=power,
              cycle=cycle,
              name=name)
      return gradient_descent.GradientDescentOptimizer(learning_rate)
  return optimizer_fn


def adagrad_polynomial_decay(
        decay_steps=10000,
        end_learning_rate=0.0001,
        power=0.5,
        cycle=False,
        name=None):
  """Returns a adagrad optimizer function with polynomial_decay.

  See tesnorflow.training.learning_rate_decay.polynomial_decay how to set the
  argument. This function returns a python callable that sets adagrad optimizer
  with a polynomialy decaying learning rate for tensorflow lattice estimator
  training.

  Args:
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
      Must be positive.  See the decay computation above.
    end_learning_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The minimal end learning rate.
    power: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The power of the polynomial. Defaults to sqrt, 0.5.
    cycle: A boolean, whether or not it should cycle beyond decay_steps.
    name: String.  Optional name of the operation. Defaults to
      'PolynomialDecay'.

  Returns:
    A python callable that accepts learning_rate and sets a gradient descent
    optimizer with a polynomialy decaying learning rate for tensorflow lattice
    estimator training.
  """
  def optimizer_fn(learning_rate=0.01):
      global_step_tensor = training_util.get_or_create_global_step()
      learning_rate = learning_rate_decay.polynomial_decay(
              learning_rate=learning_rate,
              global_step=global_step_tensor,
              decay_steps=decay_steps,
              end_learning_rate=end_learning_rate,
              power=power,
              cycle=cycle,
              name=name)
      return adagrad.AdagradOptimizer(learning_rate)
  return optimizer_fn
