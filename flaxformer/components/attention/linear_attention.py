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

"""Linear attention classes."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic

import abc
import functools
from typing import Callable, Optional, Tuple, Union

import flax
from flax import linen as nn
import flax.core.variables as variables
from flax.linen import initializers
from flax.linen.linear import default_kernel_init
import jax
from jax import lax
import jax.numpy as jnp

from flaxformer import activation_partitioning
from flaxformer.components import dense
from flaxformer.components import embedding
from flaxformer.types import Array
from flaxformer.types import DType
from flaxformer.types import Initializer


class LinearAttention(metaclass=abc.ABCMeta):
  """API for attention classes that compute a linear approximation of the attention matrix.

  This allows for 1D vectors masking the key/value part of the attention
  """

  @abc.abstractmethod
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               mask: Optional[Array] = None,
               *,
               precomputed_qkv: Optional[Array] = None,
               decode: bool = False,
               enable_dropout: bool = True) -> Array:
    """Applies attention on the input data.

    Args:
      inputs_q: input queries of shape `[batch_sizes..., q_length, q_features]`.
      inputs_kv: key/values of shape `[batch_sizes..., kv_length, kv_features]`.
      mask: attention mask of shape `[batch_sizes..., num_heads, kv_length]`.
      precomputed_qkv: when using fused implementations QKVO are defined outside
        this module and we only use the module to run computations.
      decode: Whether to prepare and use an autoregressive cache.
      enable_dropout: Enables dropout if set to True.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    
    

class MultiHeadLinearAttention(nn.Module, LinearAttention):
  """Multi-head linear attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      dtype: the dtype of the computation (default: float32)
      qkv_features: dimension of the key, query, and value.
      head_dim: dimension of each head. If unspecified, it defaults to
        qkv_features // num_heads.
      out_features: dimension of the last projection
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      attention_fn: dot_product_attention or compatible function. Accepts query,
        key, value, and returns output of shape `[bs, dim1, dim2, ..., dimN,,
        num_heads, value_channels]``
      use_extra_logit: whether to include a virtual extra logit equal to zero.
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.
      output_projection: Project the output of `attention_fn` to `out_features`.
        If False, returns the output of `attention_fn` without a projection.
      split_head_kernel: whether to store QKVO variables with a split head
        dimension.
      kernels_to_fuse: Which kernels to fuse, if any.
      use_rotary_embedding: whether to use rotary embeddings.
  """
  num_heads: int
  use_bias: bool
  dtype: DType = jnp.float32
  qkv_features: Optional[int] = None
  head_dim: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.
  precision: Optional[lax.Precision] = None
  kernel_init: Initializer = default_kernel_init
  bias_init: Initializer = initializers.zeros
  rescale_logits: bool = False
  attention_fn: Callable[..., Array] = None
  use_extra_logit: bool = False
  float32_logits: bool = False
  output_projection: bool = True
  # TODO: Remove out_features and output_projection.
  split_head_kernel: bool = False
  kernels_to_fuse: Optional[str] = None
  use_rotary_embedding: bool = False
  rotary_embedding_max_timescale: float = 1e4
  # Whether to shard over the head dimension, setting this to False when the
  # number of heads is not divisible your activation num_partitions
  sharding_over_head_dimension: bool = True

  def update_cache_prefill(
      self, key: Array, value: Array, cached_key: variables.Variable,
      cached_value: variables.Variable, cache_index: variables.Variable,
      prefill_lengths: Array
  ) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """Update the autoregressive cache for multiple timesteps at once.

    This is useful for things like a prefix-lm where the encoder section of the
    input is visible bidirectionally. The key and value for this section need to
    be computed in a single shot, as a step by step approach would result in
    causal attention.

    Args:
      key: The calculated key used in attention. [batch..., length, num_heads,
        features_per_head]
      value: The calculated value used in attention. [batch..., length,
        num_heads, features_per_head]
      cached_key: The cache of previous keys. [batch..., num_heads,
        features_per_head, length]
      cached_value: The cache of previous values. [batch..., num_heads,
        features_per_head, length]
      cache_index: The timestep that we are currently calculating the key and
        value for. [batch]
      prefill_lengths: The number of timesteps we should fill in the cache.
        [batch]

    Returns:
      The key, value, and the last timestep we just filled in the cache.
      We also return the new cache values for now because assigning to a
      variable inside of a method doesn't work. These returns will be removed
      eventually.
    """
    # Make a reference to the data underlaying the variable for ease of
    # use.
    cache_index.value = prefill_lengths
    # Note, the cache index is now a vector
    # of batch size so that each example can start just after it's
    # prefix which can be different lengths for different examples.
    cur_index = cache_index.value
    # Move the sequence dimension to the end to match the cache shapes.
    key_cached = jnp.moveaxis(key, -3, -1)
    value_cached = jnp.moveaxis(value, -3, -1)
    # Reshape the index so the batch is at the beginning, default
    # broadcasting behavior is to add singleton dims to the front but
    # we need them at the end.
    batch_first_index = jnp.reshape(
        cur_index, (-1,) + tuple(1 for _ in range(cached_key.value.ndim - 1)))
    # Calculate a mask that will set any position past the prefix to zero
    # when applied to the key.
    key_mask = (
        lax.broadcasted_iota(jnp.int32, cached_key.value.shape,
                             cached_key.value.ndim - 1) < batch_first_index)
    value_mask = (
        lax.broadcasted_iota(jnp.int32, cached_value.value.shape,
                             cached_value.value.ndim - 1) < batch_first_index)
    # Set the caches with the calculated key and values but hide anything
    # past the prefix.
    cached_key_value = key_cached * key_mask
    cached_value_value = value_cached * value_mask
    return (key, value, cur_index, cached_key_value, cached_value_value,
            prefill_lengths)

  def update_cache_decode(
      self, key: Array, value: Array, cached_key: variables.Variable,
      cached_value: variables.Variable, cache_index: variables.Variable
  ) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """Update the next timestep in the autoregressive cache.

    This is used during step by step decoding where each key and value we get
    are a single (the next) timestep.

    Args:
      key: The calculated key used in attention. [batch..., 1, num_heads,
        features_per_head]
      value: The calculated value used in attention. [batch..., 1, num_heads,
        features_per_head]
      cached_key: The cache of previous keys. [batch..., num_heads,
        features_per_head, length]
      cached_value: The cache of previous values. [batch..., num_heads,
        features_per_head, length]
      cache_index: The timestep that we are currently calculating the key and
        value for. [batch] if we are decoding after doing a prefill or [1] if we
        are starting with step-by-step decoding.

    Returns:
      The key, value, and the last timestep we just filled in the cache. Note:
      this index is the last timestep we just fill, the actual value of the
      `cache_index` is already increased to point to the next timestep to fill.
      We also return the new cache values for now because assigning to a
      variable inside of a method doesn't work. These returns will be removed
      eventually.
    """
    cache_length = cached_key.value.shape[-1]
    # Create a OHE of the current index. NOTE: the index is increased
    # below.
    # Note: We reshape the index into a column vector so that it will work
    # if the index is a scalar or a vector with different cache positions
    # from different elements in a batch.
    cur_index = jnp.reshape(cache_index.value, (-1,))
    one_hot_indices = jax.nn.one_hot(cur_index, cache_length, dtype=key.dtype)
    # In order to update the key, value caches with the current key and
    # value, we move the length axis to the back, similar to what we did
    # for the cached ones above.
    # Note these are currently the key and value of a single position,
    # since we feed one position at a time.
    one_token_key = jnp.moveaxis(key, -3, -1)
    one_token_value = jnp.moveaxis(value, -3, -1)
    # The one hot indices are now either [1, length] for a scalar index or
    # [batch size, length] for examples where there are different lengths
    # of prefixes. We need to add dims for num_heads and num_features as
    # broadcasting doesn't work for the batched version.
    one_hot_indices = jnp.expand_dims(
        jnp.expand_dims(one_hot_indices, axis=1), axis=1)
    # Update key, value caches with our new 1d spatial slices.
    # We implement an efficient scatter into the cache via one-hot
    # broadcast and addition.
    # Key/Value have seq lengths of 1 while one_hot has a seq_length
    # of length. key/value will broadcast their value to each timestep
    # and the onehot will mask all but the correct timesteps.
    key = cached_key.value + one_token_key * one_hot_indices
    value = cached_value.value + one_token_value * one_hot_indices
    cached_key_value = key
    cached_value_value = value
    cache_index_value = cache_index.value + 1
    # Move the keys and values back to their original shapes.
    key = jnp.moveaxis(key, -1, -3)
    value = jnp.moveaxis(value, -1, -3)
    return (key, value, cur_index, cached_key_value, cached_value_value,
            cache_index_value)

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               mask: Optional[Array] = None,
               bias: Optional[Array] = None,
               *,
               precomputed_qkv: Optional[Array] = None,
               decode: bool = False,
               enable_dropout: bool = True,
               prefill: bool = False,
               prefill_lengths: Optional[Array] = None) -> Array:
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    There are two modes: decoding and non-decoding (e.g., training). The mode is
    determined by `decode`.

    During decoding mode, this method is called twice, by `init` and
    `apply`. In the former, inputs_q: [batch..., length, qkv_features] and
    inputs_kv: [batch..., length, qkv_features]

    During apply, query, key and value all have the shape: [batch * beam, 1,
    qkv_features] where the batch dimension is added to include multiple beams.
    Note that the batch dimension is different during the init and apply calls.
    This is because the cached variables are directly passed-in during `apply`
    method. In other words, the cache variables such as `cached_key` are
    initialized with `batch` dim, expanded by tiling in the beam search function
    to `batch * beam` dimension, and passed to the `apply` method as part of a
    variable dict.

    Args:
      inputs_q: input queries of shape `[batch_sizes..., q_length, q_features]`.
      inputs_kv: key/values of shape `[batch_sizes..., kv_length, kv_features]`.
      mask: attention mask of shape `[batch_sizes..., {1, num_heads}, q_length,
        kv_length]`.
      bias: attention bias of shape `[batch_sizes..., num_heads, q_length,
        kv_length]`.
      precomputed_qkv: when using fused implementations QKVO are defined outside
        this module and we only use the module to run computations.
      decode: Whether to prepare and use an autoregressive cache.
      enable_dropout: Enables dropout if set to True.
      prefill: Whether to run a partial sequence to prefill the cache.
      prefill_lengths: The length of each partial sequence we are filling in the
        cache, lengths are inferred from the mask if not provided.

    Returns:
      If output_projection is True, then output of shape
      `[batch_sizes..., length, out_features]`, where out_features is set to
      features if not provided. If output_projection is False, then output of
      shape `[batch_sizes..., length, qkv_features]`, where qkv_features is set
      to features if not provided.
    """
    validate_linear_attention_call_parameter_shapes(inputs_q, inputs_kv, mask,
                                                   bias, self.num_heads)
    if precomputed_qkv is not None:
      raise ValueError('Support for precomputed QKVO not implemented.')

    rotary_index = None
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    if self.head_dim is None:
      head_dim = qkv_features // self.num_heads
    else:
      head_dim = self.head_dim

    if self.kernels_to_fuse and not self.split_head_kernel:
      raise ValueError('Un-reshaped kernels are required when using QKV fused '
                       'kernel optimization.')

    # Is attention logit rescaling explicit or folded into initializer?
    if self.rescale_logits:
      query_init = self.kernel_init
    else:
      if self.kernels_to_fuse:
        raise ValueError('Cannot fold in logit normalization to query '
                         'initializer when using fused kernels.')
      depth_scaling = jnp.sqrt(head_dim).astype(self.dtype)
      query_init = lambda *args: self.kernel_init(*args) / depth_scaling

    make_dense = functools.partial(
        dense.DenseGeneral,
        axis=-1,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        precision=self.precision,
        reshape_kernel=not self.split_head_kernel,
    )

    # Project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, num_heads, features_per_head]
    if self.kernels_to_fuse is None:
      query = make_dense(
          kernel_init=query_init,
          features=(self.num_heads, head_dim),
          kernel_axis_names=['embed', 'heads', 'head_dim'],
          name='query')(
              inputs_q)
      key = make_dense(
          kernel_init=self.kernel_init,
          features=(self.num_heads, head_dim),
          kernel_axis_names=['embed', 'heads', 'head_dim'],
          name='key')(
              inputs_kv)
      value = make_dense(
          kernel_init=self.kernel_init,
          features=(self.num_heads, head_dim),
          kernel_axis_names=['embed', 'heads', 'head_dim'],
          name='value')(
              inputs_kv)
    # TODO: should we fuse/slice along depth or head dim?
    elif self.kernels_to_fuse == 'qkv':
      if inputs_q is not inputs_kv:
        raise ValueError('qkv fusion is only supported in self-attention mode '
                         '(when inputs_q is inputs_kv).')
      # 'qkv' fusion mode implies self-attention
      qkv = make_dense(
          kernel_init=self.kernel_init,
          features=(3, self.num_heads, head_dim),
          kernel_axis_names=['embed', 'unmodeled', 'heads', 'head_dim'],
          name='qkv_fused')(
              inputs_q)
      query = jnp.squeeze(lax.dynamic_slice_in_dim(qkv, 0, 1, -3), -3)
      key = jnp.squeeze(lax.dynamic_slice_in_dim(qkv, 1, 1, -3), -3)
      value = jnp.squeeze(lax.dynamic_slice_in_dim(qkv, 2, 1, -3), -3)
    elif self.kernels_to_fuse == 'kv':
      query = make_dense(
          kernel_init=query_init,
          features=(self.num_heads, head_dim),
          kernel_axis_names=['embed', 'heads', 'head_dim'],
          name='query')(
              inputs_q)
      kv = make_dense(
          kernel_init=self.kernel_init,
          features=(2, self.num_heads, head_dim),
          kernel_axis_names=['embed', 'unmodeled', 'heads', 'head_dim'],
          name='kv_fused')(
              inputs_kv)
      key = jnp.squeeze(lax.dynamic_slice_in_dim(kv, 0, 1, -3), -3)
      value = jnp.squeeze(lax.dynamic_slice_in_dim(kv, 1, 1, -3), -3)
    else:
      raise ValueError('Incorrect kernel fusion mode specified.')

    if self.sharding_over_head_dimension:
      query = activation_partitioning.with_sharding(query, 2)
      key = activation_partitioning.with_sharding(key, 2)
      value = activation_partitioning.with_sharding(value, 2)

    query: Array = query  # hint to quiet pytype.
    key: Array = key
    value: Array = value

    if prefill and decode:
      raise ValueError('prefill and decode cannot both be true at the same'
                       'time. If you are using a prefix LM with bidirectional '
                       'attention on the inputs, please make a call with '
                       'prefill=True that includes an attention mask that '
                       'covers your inputs first and then make your decoding '
                       'calls.')
    if prefill or decode:
      # Detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      # The key and value have dimension
      # [batch..., length, num_heads, features_per_head], but we cache them as
      # [batch..., num_heads, features_per_head, length] as a TPU fusion
      # optimization. This also enable the "scatter via one-hot broadcast"
      # trick, which means we do a one-hot broadcast instead of a scatter/gather
      # operations, which gives a 3-4x speedup in practice.
      swap_dims = lambda x: x[:-3] + tuple(x[i] for i in [-2, -1, -3])
      cached_key = self.variable('cache', 'cached_key', jnp.zeros,
                                 swap_dims(key.shape), key.dtype)
      cached_value = self.variable('cache', 'cached_value', jnp.zeros,
                                   swap_dims(value.shape), value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      rotary_index = cache_index.value
      if is_initialized:
        # Here we are in "apply()".
        *batch_dims, num_heads, features_per_head, length = (
            cached_key.value.shape)
        if prefill:
          if prefill_lengths is None:
            # Figure out how far each element in the batch fills the cache based
            # on the mask. We index each element in the batch, the first head
            # dim (because this is always set to one), and the first query
            # vector. If there is any prefix at all, the first element in the
            # prefix would be part of it.
            raise NotImplementedError  # TODO(tomprom)
            prefill_lengths = jnp.sum(
                mask[:, 0, 0, :], axis=-1).astype(cache_index.value.dtype)
          (key, value, cur_index, cached_key_value, cached_value_value,
           cache_index_value) = self.update_cache_prefill(
               key, value, cached_key, cached_value, cache_index,
               prefill_lengths)
        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        elif decode:
          # Check the shape of the cached key against the input query.
          expected_shape = tuple(batch_dims) + (1, num_heads, features_per_head)
          if expected_shape != query.shape:
            raise ValueError('Autoregressive cache shape error, '
                             'expected query shape %s instead got %s.' %
                             (expected_shape, query.shape))
          (key, value, cur_index, cached_key_value, cached_value_value,
           cache_index_value) = self.update_cache_decode(
               key, value, cached_key, cached_value, cache_index)
          
          # NB: While decoding, we rely on a causal mask implementation in the
          # linear attention_fn

        # Currently, updating a variable inside of a method is not handled
        # in flax, so we return the actual values and assign them in the main
        # compacted call for now.
        # TODO: Move variable assignment inside of the
        # cache update functions once variable references are tracked across
        # transform boundaries.
        cache_index.value = cache_index_value
        cached_key.value = cached_key_value
        cached_value.value = cached_value_value

    # Mask the key and value with the attention mask
    if mask is not None:
      if mask.shape[1] > 1:
          key = jnp.einsum('...hl,...lhd->...lhd', mask, key)
          value = jnp.einsum('...hl,...lhd->...lhd', mask, value)
      else:
          key = jnp.einsum('...xl,...lhd->...lhd', mask, key)
          value = jnp.einsum('...xl,...lhd->...lhd', mask, value)

    dropout_rng = None
    if enable_dropout and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    if self.use_rotary_embedding:
      # use rotary embeddings before attention
      # https://arxiv.org/abs/2104.09864
      # TODO: Put it in a new class
      dim = query.shape[-1]
      max_length = max(query.shape[1], key.shape[1])
      sin, cos = embedding.generate_fixed_pos_embedding(
          dim, max_length, max_timescale=self.rotary_embedding_max_timescale)
      query, key = embedding.apply_rotary_embedding(
          query,
          key,
          cos,
          sin,
          batch_size=inputs_q.shape[0],
          num_heads=self.num_heads,
          decode=decode,
          rotary_index=rotary_index)

    # Compute and apply attention (at the same time).
    if self.rescale_logits or self.use_extra_logit or self.float32_logits:
        # TODO: Implement these in FAVOR+ so they can be used here
        raise NotImplementedError
    if enable_dropout and self.dropout_rate > 0.:
        raise NotImplementedError
    x = self.attention_fn(
        query,
        key,
        value,
        broadcast_dropout=self.broadcast_dropout,
        # rescale_logits=self.rescale_logits,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        # enable_dropout=enable_dropout,
        dtype=self.dtype,
        precision=self.precision,
        # use_extra_logit=self.use_extra_logit,
        # float32_logits=self.float32_logits
    )  # pytype: disable=wrong-keyword-args

    if not self.output_projection:
      return x

    # Back to the original inputs dimensions.
    out = dense.DenseGeneral(
        features=features,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        precision=self.precision,
        reshape_kernel=not self.split_head_kernel,
        kernel_axis_names=['heads', 'head_dim', 'embed'],
        name='out')(  # pytype: disable=wrong-arg-types
            x)
    return out
        
    
class FactoredDense(nn.Module):
  n_modules: int
  d_out: Optional[int] = None
  use_bias: bool = True
  use_bfloat16 = False

  @nn.compact
  def call(self, x,
           decode: bool = False,
           enable_dropout: bool = True):
    r"""Returns a Dense-like layer, internally factored to use fewer parameters.

    This layer treats an activation vector as if divided into :math:`M`
    subvectors (``n_modules`` 'modules'). It uses this factored view to compute
    a :py:class:`Dense`-like mapping with high mixing/connectivity, but using
    approximately :math:`1/M` the number of weights of a similarly dimensioned
    :py:class:`Dense` layer.

    More specifically, each activation vector of dimensionality ``n_in`` is
    multiplied element-wise (a generalized form of gating) with ``n_modules``
    vectors also of dimensionality ``n_in``. The resulting vectors are projected
    to the subvector/module dimensionality ``d_out / n_modules`` via a matrix
    multiply, and finally reshaped back to a single vector of dimensionality
    ``d_out``. Optionally, a bias vector of dimensionality ``d_out`` is added at
    the end. All the above-mentioned non-input objects -- gating vectors,
    projection matrix, and optional bias -- are trainable weights.

    Args:
      n_modules: Number by which an activation vector is divided into subvectors
          (modules) for the factored computation.
      d_in: Last/innermost dimension of input array.
      d_out: Last/innermost dimension of output array.
      use_bias: If True, add bias vectors at the end of the layer; else end the
          layer with the matrix multiply.
      use_bfloat16: If True, use bfloat16 weights; else use float32 weights.
    """
    d_in = x.shape[-1]
    if self.d_out is None:
      d_out = d_in
    else:
      d_out = self.d_out
      
    if d_out % self.n_modules != 0:
      raise ValueError(f'Value d_out ({d_out}) must be a multiple of arg '
                       f'n_modules ({self.n_modules}).')
    d_module = d_out // self.n_modules
    
    gating = self.param('gating', flax.linen.initializers.normal(0.5), [self.n_modules, d_in], x.dtype)
    projection = self.param('projection', flax.linen.initializers.glorot_uniform(), [self.n_modules, d_in], x.dtype)

    x = jnp.einsum('...d,nd,dm->...nm', x, gating, projection)
    x = jnp.reshape(x, tuple(x.shape)[:-2] + (-1,))
    if self.use_bias:
      bias = self.param('bias', flax.linen.initializers.normal(1e-6), [self.n_modules, d_in], x.dtype)
      x += bias

    return x
  
  
class RememberPad(nn.Module):
  n_items_to_remember: int
  
  @nn.compact
  def call(self, x,
           decode: bool = False,
           enable_dropout: bool = True):
    if self._n_items_to_remember == 0:
      return x
    if decode:
      raise NotImplementedError
    else:
      pad_widths = [[0, 0] for _ in range(len(x.shape))]
      pad_widths[1][0] = self.n_items_to_remember
      x = jnp.pad(x, pad_width=pad_widths, mode='constant')
    return x


class LocallyConvDense(nn.Module):
  
  n_modules: int
  n_units: int
  kernel_size: int = 1
  length_kernel_size: int = 1

  @nn.compact
  def call(self, x,
           decode: bool = False,
           enable_dropout: bool = True,):
    """Layer using local convolutions for approximation of Dense layer.

    The layer splits the last axis of a tensor into `n_modules`, then runs
    a convolution on all those modules, and concatenates their results.
    It is similar to LocallyConnectedDense above, but shares weights.

    Args:
      n_modules: Indicates how many modules (pixels) should be input and output
          split into for processing.
      n_units: how many outputs (filters) should each module generate.
      mode: One of `'train'`, `'eval'`, or `'predict'`.
      kernel_size: The size of the kernel to be used.
      length_kernel_size: If > 1, also do causal convolution on the previous axis,
        which is often the sentence length in sequence models.

    Returns:
        LocallyConvDense tl.Layer.
    """
    if decode:
      # Prediction mode is not yet implemented for the convolution layer
      # It required "remembering" the last few tokens
      raise NotImplementedError
    if self.n_modules == 1:
      return dense.DenseGeneral(self.n_units)
    if self.kernel_size % 2 != 1:
      raise ValueError('Currently we only handle odd kernel sizes.')
    half = (self.kernel_size - 1) // 2
    pad_widths = [[0, 0], [0, 0], [half, half], [0, 0]]
    x = jnp.reshape(x, tuple(x.shape)[:-1] + (self.n_modules, -1))
    x = jnp.pad(x, pad_width=pad_widths, mode='constant')
    x = RememberPad(n_items_to_remember=self.length_kernel_size - 1)(x)
    x = nn.Conv(self.n_units, kernel_size=(self.length_kernel_size, self.kernel_size))(x)
    x = jnp.reshape(x, tuple(x.shape)[:-2] + (-1,))
    return x
  
    

class MultiHeadSparseLinearAttention(MultiHeadLinearAttention):
  """Multi-head linear attention with sparse QKV calculations.
  
  See: https://arxiv.org/abs/2111.12763

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      dtype: the dtype of the computation (default: float32)
      qkv_features: dimension of the key, query, and value.
      head_dim: dimension of each head. If unspecified, it defaults to
        qkv_features // num_heads.
      out_features: dimension of the last projection
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      attention_fn: dot_product_attention or compatible function. Accepts query,
        key, value, and returns output of shape `[bs, dim1, dim2, ..., dimN,,
        num_heads, value_channels]``
      use_extra_logit: whether to include a virtual extra logit equal to zero.
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.
      output_projection: Project the output of `attention_fn` to `out_features`.
        If False, returns the output of `attention_fn` without a projection.
      split_head_kernel: whether to store QKVO variables with a split head
        dimension.
      kernels_to_fuse: Which kernels to fuse, if any.
      use_rotary_embedding: whether to use rotary embeddings.
  """

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               mask: Optional[Array] = None,
               bias: Optional[Array] = None,
               *,
               precomputed_qkv: Optional[Array] = None,
               decode: bool = False,
               enable_dropout: bool = True,
               prefill: bool = False,
               prefill_lengths: Optional[Array] = None) -> Array:
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    There are two modes: decoding and non-decoding (e.g., training). The mode is
    determined by `decode`.

    During decoding mode, this method is called twice, by `init` and
    `apply`. In the former, inputs_q: [batch..., length, qkv_features] and
    inputs_kv: [batch..., length, qkv_features]

    During apply, query, key and value all have the shape: [batch * beam, 1,
    qkv_features] where the batch dimension is added to include multiple beams.
    Note that the batch dimension is different during the init and apply calls.
    This is because the cached variables are directly passed-in during `apply`
    method. In other words, the cache variables such as `cached_key` are
    initialized with `batch` dim, expanded by tiling in the beam search function
    to `batch * beam` dimension, and passed to the `apply` method as part of a
    variable dict.

    Args:
      inputs_q: input queries of shape `[batch_sizes..., q_length, q_features]`.
      inputs_kv: key/values of shape `[batch_sizes..., kv_length, kv_features]`.
      mask: attention mask of shape `[batch_sizes..., {1, num_heads}, q_length,
        kv_length]`.
      bias: attention bias of shape `[batch_sizes..., num_heads, q_length,
        kv_length]`.
      precomputed_qkv: when using fused implementations QKVO are defined outside
        this module and we only use the module to run computations.
      decode: Whether to prepare and use an autoregressive cache.
      enable_dropout: Enables dropout if set to True.
      prefill: Whether to run a partial sequence to prefill the cache.
      prefill_lengths: The length of each partial sequence we are filling in the
        cache, lengths are inferred from the mask if not provided.

    Returns:
      If output_projection is True, then output of shape
      `[batch_sizes..., length, out_features]`, where out_features is set to
      features if not provided. If output_projection is False, then output of
      shape `[batch_sizes..., length, qkv_features]`, where qkv_features is set
      to features if not provided.
    """
    validate_linear_attention_call_parameter_shapes(inputs_q, inputs_kv, mask,
                                                   bias, self.num_heads)
    if precomputed_qkv is not None:
      raise ValueError('Support for precomputed QKVO not implemented.')

    rotary_index = None
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    if self.head_dim is None:
      head_dim = qkv_features // self.num_heads
    else:
      head_dim = self.head_dim
    
    if self.kernels_to_fuse:
        raise ValueError('Fused kernels are not supported with sparse attention')

    if self.kernels_to_fuse and not self.split_head_kernel:
      raise ValueError('Un-reshaped kernels are required when using QKV fused '
                       'kernel optimization.')

    # Is attention logit rescaling explicit or folded into initializer?
    if self.rescale_logits:
      query_init = self.kernel_init
    else:
      if self.kernels_to_fuse:
        raise ValueError('Cannot fold in logit normalization to query '
                         'initializer when using fused kernels.')
      depth_scaling = jnp.sqrt(head_dim).astype(self.dtype)
      query_init = lambda *args: self.kernel_init(*args) / depth_scaling

    make_dense = functools.partial(
        dense.DenseGeneral,
        axis=-1,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        precision=self.precision,
        reshape_kernel=not self.split_head_kernel,
    )

    # Project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, num_heads, features_per_head]
    query = make_dense(
      kernel_init=query_init,
      features=(self.num_heads, head_dim),
      kernel_axis_names=['embed', 'heads', 'head_dim'],
      name='query')(
          inputs_q)
    key = make_dense(
      kernel_init=self.kernel_init,
      features=(self.num_heads, head_dim),
      kernel_axis_names=['embed', 'heads', 'head_dim'],
      name='key')(
          inputs_kv)
    value = make_dense(
      kernel_init=self.kernel_init,
      features=(self.num_heads, head_dim),
      kernel_axis_names=['embed', 'heads', 'head_dim'],
      name='value')(
          inputs_kv)

    if self.sharding_over_head_dimension:
      query = activation_partitioning.with_sharding(query, 2)
      key = activation_partitioning.with_sharding(key, 2)
      value = activation_partitioning.with_sharding(value, 2)

    query: Array = query  # hint to quiet pytype.
    key: Array = key
    value: Array = value

    if prefill and decode:
      raise ValueError('prefill and decode cannot both be true at the same'
                       'time. If you are using a prefix LM with bidirectional '
                       'attention on the inputs, please make a call with '
                       'prefill=True that includes an attention mask that '
                       'covers your inputs first and then make your decoding '
                       'calls.')
    if prefill or decode:
      # Detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      # The key and value have dimension
      # [batch..., length, num_heads, features_per_head], but we cache them as
      # [batch..., num_heads, features_per_head, length] as a TPU fusion
      # optimization. This also enable the "scatter via one-hot broadcast"
      # trick, which means we do a one-hot broadcast instead of a scatter/gather
      # operations, which gives a 3-4x speedup in practice.
      swap_dims = lambda x: x[:-3] + tuple(x[i] for i in [-2, -1, -3])
      cached_key = self.variable('cache', 'cached_key', jnp.zeros,
                                 swap_dims(key.shape), key.dtype)
      cached_value = self.variable('cache', 'cached_value', jnp.zeros,
                                   swap_dims(value.shape), value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      rotary_index = cache_index.value
      if is_initialized:
        # Here we are in "apply()".
        *batch_dims, num_heads, features_per_head, length = (
            cached_key.value.shape)
        if prefill:
          if prefill_lengths is None:
            # Figure out how far each element in the batch fills the cache based
            # on the mask. We index each element in the batch, the first head
            # dim (because this is always set to one), and the first query
            # vector. If there is any prefix at all, the first element in the
            # prefix would be part of it.
            raise NotImplementedError  # TODO(tomprom)
            prefill_lengths = jnp.sum(
                mask[:, 0, 0, :], axis=-1).astype(cache_index.value.dtype)
          (key, value, cur_index, cached_key_value, cached_value_value,
           cache_index_value) = self.update_cache_prefill(
               key, value, cached_key, cached_value, cache_index,
               prefill_lengths)
        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        elif decode:
          # Check the shape of the cached key against the input query.
          expected_shape = tuple(batch_dims) + (1, num_heads, features_per_head)
          if expected_shape != query.shape:
            raise ValueError('Autoregressive cache shape error, '
                             'expected query shape %s instead got %s.' %
                             (expected_shape, query.shape))
          (key, value, cur_index, cached_key_value, cached_value_value,
           cache_index_value) = self.update_cache_decode(
               key, value, cached_key, cached_value, cache_index)
          
          # NB: While decoding, we rely on a causal mask implementation in the
          # linear attention_fn

        # Currently, updating a variable inside of a method is not handled
        # in flax, so we return the actual values and assign them in the main
        # compacted call for now.
        # TODO: Move variable assignment inside of the
        # cache update functions once variable references are tracked across
        # transform boundaries.
        cache_index.value = cache_index_value
        cached_key.value = cached_key_value
        cached_value.value = cached_value_value

    # Mask the key and value with the attention mask
    if mask is not None:
      if mask.shape[1] > 1:
          key = jnp.einsum('...hl,...lhd->...lhd', mask, key)
          value = jnp.einsum('...hl,...lhd->...lhd', mask, value)
      else:
          key = jnp.einsum('...xl,...lhd->...lhd', mask, key)
          value = jnp.einsum('...xl,...lhd->...lhd', mask, value)

    dropout_rng = None
    if enable_dropout and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    if self.use_rotary_embedding:
      # use rotary embeddings before attention
      # https://arxiv.org/abs/2104.09864
      # TODO: Put it in a new class
      dim = query.shape[-1]
      max_length = max(query.shape[1], key.shape[1])
      sin, cos = embedding.generate_fixed_pos_embedding(
          dim, max_length, max_timescale=self.rotary_embedding_max_timescale)
      query, key = embedding.apply_rotary_embedding(
          query,
          key,
          cos,
          sin,
          batch_size=inputs_q.shape[0],
          num_heads=self.num_heads,
          decode=decode,
          rotary_index=rotary_index)

    # Compute and apply attention (at the same time).
    if self.rescale_logits or self.use_extra_logit or self.float32_logits:
        # TODO: Implement these in FAVOR+ so they can be used here
        raise NotImplementedError
    if enable_dropout and self.dropout_rate > 0.:
        raise NotImplementedError
    x = self.attention_fn(
        query,
        key,
        value,
        broadcast_dropout=self.broadcast_dropout,
        # rescale_logits=self.rescale_logits,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        # enable_dropout=enable_dropout,
        dtype=self.dtype,
        precision=self.precision,
        # use_extra_logit=self.use_extra_logit,
        # float32_logits=self.float32_logits
    )  # pytype: disable=wrong-keyword-args

    if not self.output_projection:
      return x

    # Back to the original inputs dimensions.
    out = dense.DenseGeneral(
        features=features,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        precision=self.precision,
        reshape_kernel=not self.split_head_kernel,
        kernel_axis_names=['heads', 'head_dim', 'embed'],
        name='out')(  # pytype: disable=wrong-arg-types
            x)
    return out
    
    

def validate_linear_attention_call_parameter_shapes(inputs_q: Array,
                                                    inputs_kv: Array,
                                                    mask: Optional[Array],
                                                    bias: Optional[Array],
                                                    num_heads: Optional[int]):
  """Validates the shapes of parameters to DenseAttention call methods."""
  if inputs_q.ndim != inputs_kv.ndim:
    raise ValueError(f'Mismatched inputs rank: expected '
                     f'inputs_q.ndim ({inputs_q.ndim}) == '
                     f'inputs_kv.ndim ({inputs_kv.ndim})')
  if inputs_q.ndim < 3:
    raise ValueError(f'Expected rank of inputs >= 3, was {inputs_q.ndim}')
  if inputs_q.shape[:-3] != inputs_kv.shape[:-3]:
    raise ValueError(f'Mismatched batch dims: expected '
                     f'inputs_q.shape[:-3] ({inputs_q.shape[:-3]}) == '
                     f'inputs_kv.shape[:-3] ({inputs_kv.shape[:-3]})')
  if mask is not None:
    if mask.ndim != inputs_kv.ndim:
      raise ValueError(f'Mismatched ranks: expected '
                       f'mask.ndim ({mask.ndim}) to be equal to '
                       f'inputs_kv.ndim ({inputs_kv.ndim})')
    if num_heads is not None:
      if mask.shape[-2] not in (1, num_heads):
        raise ValueError(f'Mismatched num_heads: expected '
                         f'mask.shape[-2] ({mask.shape[-2]}) == '
                         f'num_heads ({num_heads}), or 1')
    else:
      num_heads = mask.shape[-2]
    if mask.shape[-1] != inputs_kv.shape[-2]:
      raise ValueError(f'Mismatched kv_length: expected '
                       f'mask.shape[-1] ({mask.shape[-1]}) == '
                       f'inputs_kv.shape[-2] ({inputs_kv.shape[-2]})')
  if bias is not None:
    raise ValueError('Bias must be None in linear attention.')
