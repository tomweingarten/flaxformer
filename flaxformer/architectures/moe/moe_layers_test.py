# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for moe_layers."""

import functools
from typing import Any, Dict, Mapping
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
from jax import numpy as jnp
import numpy as np

from flaxformer.architectures.moe import moe_layers
from flaxformer.architectures.moe import routing
from flaxformer.components import dense

# Type Stubs
FrozenDict = flax.core.frozen_dict.FrozenDict
MoeLayer = moe_layers.MoeLayer
PRNGKey = Any

NUM_CLASSES = 2


def init_layer_variables(
    key: PRNGKey, module: MoeLayer,
    init_batch: Mapping[str, jnp.ndarray]) -> Dict[str, Any]:
  """Initialize layer parameters."""
  params_key, dropout_key, jitter_key = jax.random.split(key, num=3)

  return module.init(
      {
          'params': params_key,
          'dropout': dropout_key,
          'jitter': jitter_key
      }, **init_batch)


class MoeLayerTest(parameterized.TestCase):

  @parameterized.parameters(dict(dispatch='scatter'), dict(dispatch='mask'))
  def test_moe_layer_runs(self, dispatch: str):
    batch_size = 3
    max_seq_length = 4
    num_tokens = batch_size * max_seq_length
    hidden_dim = 2
    num_experts = 4
    rng = jax.random.PRNGKey(0)

    if dispatch == 'mask':
      router = routing.TokensChooseMaskedRouter(
          router_weights=routing.RouterWeights(name='router_weights'),
          jitter_noise=0.,
          num_selected_experts=2,
          batch_prioritized_routing=True,
          ignore_padding_tokens=True,
          dtype=jnp.float32)
    else:
      router = routing.TokensChooseScatterRouter(
          router_weights=routing.RouterWeights(name='router_weights'),
          jitter_noise=0.,
          num_selected_experts=2,
          batch_prioritized_routing=True,
          ignore_padding_tokens=True,
          dtype=jnp.float32)

    expert = dense.MlpBlock(
        use_bias=False,
        intermediate_dim=2,
        activations=('gelu',),
        intermediate_dropout_rate=0.1)
    moe_layer = moe_layers.MoeLayer(
        num_experts=num_experts,
        max_group_size=num_tokens,
        router=router,
        train_capacity_factor=1.5,
        eval_capacity_factor=1.5,
        expert=expert,
        num_model_partitions=1,
        split_params=False)  # Ensures all experts start with same params
    init_batch = {
        'inputs':
            jnp.ones((batch_size, max_seq_length, hidden_dim), jnp.float32)
    }
    params = init_layer_variables(rng, moe_layer, init_batch)['params']

    expected_keys = {'router', 'expert'}
    self.assertEqual(params.keys(), expected_keys)

    dropout_rng, jitter_rng, init_rng = jax.random.split(rng, num=3)
    inputs = jax.random.uniform(
        init_rng, (batch_size, max_seq_length, hidden_dim),
        minval=-10,
        maxval=10)
    actual_outputs, state = moe_layer.apply({'params': params},
                                            rngs={
                                                'dropout': dropout_rng,
                                                'jitter': jitter_rng
                                            },
                                            mutable=['intermediates'],
                                            inputs=inputs)
    self.assertEqual(actual_outputs.shape,
                     (batch_size, max_seq_length, hidden_dim))

    for metric in [
        'auxiliary_loss', 'router_z_loss', 'fraction_tokens_left_behind',
        'expert_usage', 'router_confidence'
    ]:
      self.assertIn(metric, state['intermediates'])

  def test_scatter_mask_dispatch_equal(self):
    batch_size = 4
    max_seq_length = 4
    hidden_dim = 2
    num_experts = 2
    tokens_per_group = 8
    num_groups = batch_size * max_seq_length // tokens_per_group

    rng = jax.random.PRNGKey(0)

    expert = dense.MlpBlock(
        use_bias=True,
        intermediate_dropout_rate=0.,
        final_dropout_rate=0.,
        intermediate_dim=2,
        name='feed_forward_expert')
    moe_layer_factory = functools.partial(
        moe_layers.MoeLayer,
        num_experts=num_experts,
        dropout_rate=0.,
        max_group_size=tokens_per_group,
        train_capacity_factor=1.,
        eval_capacity_factor=1.,
        expert=expert,
        num_model_partitions=1,
        split_params=False)  # Ensures all experts start with same params

    router_weights = routing.RouterWeights(name='router_weights')
    masked_router = routing.TokensChooseMaskedRouter(
        router_weights=router_weights,
        jitter_noise=0.,
        num_selected_experts=2,
        batch_prioritized_routing=True,
        dtype=jnp.float32,
        ignore_padding_tokens=False)
    masked_moe_layer = moe_layer_factory(router=masked_router)
    scatter_router = routing.TokensChooseScatterRouter(
        router_weights=router_weights,
        jitter_noise=0.,
        num_selected_experts=2,
        batch_prioritized_routing=True,
        dtype=jnp.float32,
        ignore_padding_tokens=False)
    scatter_moe_layer = moe_layer_factory(router=scatter_router)

    inputs = jax.random.uniform(
        rng, (batch_size, max_seq_length, hidden_dim), minval=-10, maxval=10)

    # Mock the router weights to ensure both layers compute with the same
    # logits.
    mock_router_logits = jax.random.uniform(
        rng, (num_groups, tokens_per_group, num_experts), minval=-1, maxval=1)
    with mock.patch.object(
        masked_router, 'router_weights', return_value=mock_router_logits):
      masked_outputs, _ = masked_moe_layer.init_with_output(
          rng, inputs, enable_dropout=False)
    with mock.patch.object(
        scatter_router, 'router_weights', return_value=mock_router_logits):
      scatter_outputs, _ = scatter_moe_layer.init_with_output(
          rng, inputs, enable_dropout=False)

    expected_outputs = jnp.array([
        [
            [-5.4286949e-07, -9.7972497e-07],
            [-2.8485384e+00, -2.0157995e+00],
            [-3.3071041e+00, -2.3111360e+00],
            [0.0000000e+00, 0.0000000e+00],
        ],
        [
            [-4.8432793e-07, -8.7407409e-07],
            [-5.3204980e-07, -9.6019858e-07],
            [-7.4125074e-07, -1.3377468e-06],
            [1.3946553e+00, 1.0524274e+00],
        ],
        [
            [-1.5633354e+00, -1.0680735e+00],
            [-6.9210348e+00, -4.8458524e+00],
            [4.6841961e-01, 3.9556113e-01],
            [-4.6012778e-07, -8.3039981e-07],
        ],
        [
            [0.0000000e+00, 0.0000000e+00],
            [8.1172025e-01, 5.8894420e-01],
            [-1.7119075e+00, -1.1470584e+00],
            [-5.5547832e-07, -1.0024803e-06],
        ],
    ],
                                 dtype=jnp.float32)

    np.testing.assert_allclose(
        masked_outputs, expected_outputs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        scatter_outputs, expected_outputs, rtol=1e-6, atol=1e-6)

  @parameterized.parameters(
      dict(
          max_group_size=8,
          num_tokens=32,
          num_experts=2,
          num_expert_replicas=1,
          expected_num_groups=4),
      dict(
          max_group_size=9,
          num_tokens=32,
          num_experts=2,
          num_expert_replicas=1,
          expected_num_groups=4),
      dict(
          max_group_size=16,
          num_tokens=32,
          num_experts=4,
          num_expert_replicas=2,
          expected_num_groups=8),
      dict(
          max_group_size=32,
          num_tokens=32,
          num_experts=2,
          num_expert_replicas=1,
          expected_num_groups=2),
      dict(
          max_group_size=64,
          num_tokens=32,
          num_experts=2,
          num_expert_replicas=1,
          expected_num_groups=2))
  def test_num_groups(self, max_group_size: int, num_tokens: int,
                      num_experts: int, num_expert_replicas: int,
                      expected_num_groups: int):

    self.assertEqual(
        moe_layers._num_groups(
            num_tokens,
            max_group_size,
            num_experts,
            num_expert_replicas,
            strict_group_size=False), expected_num_groups)

  def test_strict_group_size(self):
    with self.assertRaisesRegex(
        ValueError,
        'Selected group_size=8 is less than the max_group_size=16.'):
      moe_layers._num_groups(
          num_tokens=16,
          max_group_size=16,
          num_experts=2,
          num_expert_replicas=1,
          strict_group_size=True)

  @parameterized.parameters(
      dict(num_model_partitions=-1), dict(num_model_partitions=0),
      dict(num_model_partitions=1))
  def test_num_partitions_incorrect(self, num_model_partitions: int):
    expert = dense.MlpBlock(
        use_bias=False, intermediate_dim=2, name='feed_forward_expert')
    router = routing.ExpertsChooseMaskedRouter(
        router_weights=routing.RouterWeights(name='router_weights'),
        jitter_noise=0.,
        dtype=jnp.float32,
        ignore_padding_tokens=True)
    moe_layer = moe_layers.MoeLayer(
        num_experts=2,
        router=router,
        max_group_size=16,
        train_capacity_factor=1.,
        eval_capacity_factor=1.,
        expert=expert,
        num_model_partitions=num_model_partitions,
        optimize_model_parallel_communications=True)
    init_batch = {'inputs': jnp.ones((4, 4, 3), jnp.float32)}

    with self.assertRaisesRegex(
        ValueError, f'optimize_model_parallel_communications=True with '
        f'num_model_partitions={num_model_partitions} has no effect; '
        f'please set optimize_model_parallel_communications=False'):
      init_layer_variables(jax.random.PRNGKey(0), moe_layer, init_batch)


if __name__ == '__main__':
  absltest.main()
