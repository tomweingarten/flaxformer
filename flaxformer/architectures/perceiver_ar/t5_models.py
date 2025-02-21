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

"""This file contains "model" classes for T5 models."""

import enum
import functools
from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple, Union

from absl import logging
import flax
from flax import linen as nn
from flax import traverse_util
import jax
from jax import lax
import jax.numpy as jnp
import seqio
from t5x import decoding
from t5x import losses
from t5x import models
from t5x import optimizers


class CroppingMethod(enum.Enum):
  """Perceiver AR training cropping methods.

  NONE: Cropping will be done in the data pipeline, so no online cropping
    is needed.

  FULL_LATENTS: Random placement of latents between the beginning and end
    of the sequence where as many latents as possible are allocated positions.
    Advantage: Loss over as many tokens as possible, better use of compute.
    Disadvantage: May bias against learning to generate positions toward the
      beginning or end of sequences because they will be selected less
      frequently.

  EQUAL_POSITION_LIKELIHOOD: Random placement of latents such that every
    sequence position is equally likely to have loss calculated on it. Achieved
    by letting the latent "window" extend beyond the edges of the sequence and
    then cropping/masking any invalid positions.
    Advantage: Every position is equally likely to be trained.
    Disadvantage: Loss over fewer positions, wasted compute. For example, with
      a sequence length of 8192 and 2048 latent positions, each training batch
      will be only 80% non-padding tokens.
  """
  NONE = 1
  FULL_LATENTS = 2
  EQUAL_POSITION_LIKELIHOOD = 3


def crop_train_batch(
    rng: Optional[jax.random.KeyArray],
    batch: Mapping[str, jnp.ndarray],
    cropping_method: CroppingMethod,
    num_latents: int,
) -> Mapping[str, jnp.ndarray]:
  """Apply random cropping to a training batch.

  Perceiver AR can utilize a longer input sequence than the number of latents
  and therefore outputs positions for loss. In order to train the model to be
  able to generate outputs with a variety of input context lengths, random
  cropping of the input sequence is used.

  Args:
    rng: PRNG key.
    batch: T5X training batch.
    cropping_method: Type of cropping method to use.
    num_latents: Number of latents in the Perceiver AR model.

  Returns:
    A cropped batch.
  """
  first_loss_idx = jnp.argmax(batch['decoder_loss_weights'] == 1, axis=-1)
  last_loss_idx = batch['decoder_loss_weights'].shape[-1] - 1 - jnp.argmax(
      jnp.flip(batch['decoder_loss_weights'] == 1, axis=-1), axis=-1)

  logging.info('Using cropping method "%s".', cropping_method)
  if cropping_method == CroppingMethod.NONE:
    return batch
  if cropping_method == CroppingMethod.FULL_LATENTS:
    # "naive" crop selection. always results in a full batch.
    min_crop_start = first_loss_idx
    max_crop_start = last_loss_idx - num_latents + 1
  elif cropping_method == CroppingMethod.EQUAL_POSITION_LIKELIHOOD:
    # "fair" crop selection. all positions equally likely.
    min_crop_start = first_loss_idx - num_latents + 1
    max_crop_start = last_loss_idx
  else:
    raise ValueError(f'Unknown cropping method: {cropping_method}')

  seq_crop_first_idx = jax.random.randint(
      rng, [batch['decoder_loss_weights'].shape[0]], min_crop_start,
      max_crop_start + 1)

  seq_crop_end = jnp.minimum(seq_crop_first_idx + num_latents,
                             last_loss_idx + 1)
  seq_crop_start = jnp.maximum(seq_crop_first_idx, 0)

  def crop_seq(x):
    return jnp.where(
        jnp.arange(x.shape[-1])[jnp.newaxis, :] < seq_crop_end[:, jnp.newaxis],
        x, 0)

  batch = jax.tree_map(crop_seq, batch)

  # Handle the loss weights specifically to ensure that loss isn't
  # calculated for positions before seq_crop_start. This ensures that all
  # token positions have an equal likelihood of being counted in the loss.
  # Specifically, it handles cases where the crop over a sequence of length
  # 8192 is something like [8000:8192]. Even if there are 2048 latents
  # allocated to [6144:8192], loss is only calculated on [8000:8192].
  batch['decoder_loss_weights'] = jnp.where(
      jnp.arange(batch['decoder_loss_weights'].shape[-1])[jnp.newaxis, :] >=
      seq_crop_start[:, jnp.newaxis], batch['decoder_loss_weights'], 0)

  return batch


class PerceiverARModel(models.DecoderOnlyModel):
  """Model class for Perceiver AR decoder-only model.

  Implements Perceiver AR as described in https://arxiv.org/abs/2202.07765.

  Decouples input length from most of the compute requirements by utilizing
  an initial cross-attention layer over the inputs to a smaller number of
  latents for processing with the self-attention stack.
  """

  def __init__(
      self,
      module: nn.Module,
      vocabulary: seqio.Vocabulary,
      optimizer_def: optimizers.OptimizerDefType,
      num_latents: int,
      decoding_latent_reset_fill: Optional[int] = None,
      decode_fn: models.DecodeFnCallable = decoding.temperature_sample,
      inputs_bidirectional_attention: bool = False,
      feature_converter_cls: Optional[Callable[...,
                                               seqio.FeatureConverter]] = None,
      label_smoothing: float = 0.0,
      z_loss: float = 0.0,
      loss_normalizing_factor: Optional[float] = None,
      train_cropping_method: CroppingMethod = CroppingMethod.FULL_LATENTS,
  ):
    self._num_latents = num_latents
    self._decoding_latent_reset_fill = decoding_latent_reset_fill
    self._train_cropping_method = train_cropping_method
    super().__init__(
        module=module,
        vocabulary=vocabulary,
        optimizer_def=optimizer_def,
        decode_fn=decode_fn,
        inputs_bidirectional_attention=inputs_bidirectional_attention,
        feature_converter_cls=feature_converter_cls,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=loss_normalizing_factor,
    )

  def loss_fn(
      self,
      params: models.PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.random.KeyArray],
  ) -> Tuple[jnp.ndarray, models.MetricsMap]:
    """Loss function used for training with a cross-entropy loss."""
    if dropout_rng is None:
      # TODO: Either find a way to verify that this happens only in
      # eval mode or add RNG ability to T5X during eval.
      logging.info(
          'No RNG key present, so cropping method of "%s" will not occur. '
          'Should happen only during eval.', self._train_cropping_method)
    else:
      crop_train_rng, dropout_rng = jax.random.split(dropout_rng)
      batch = crop_train_batch(
          crop_train_rng,
          batch=batch,
          cropping_method=self._train_cropping_method,
          num_latents=self._num_latents)

    logits = self._compute_logits(params, batch, dropout_rng)

    loss_normalizing_factor: Optional[Union[
        float, int, str, losses.SpecialLossNormalizingFactor]]
    (loss_normalizing_factor,
     weights) = losses.get_loss_normalizing_factor_and_weights(
         self._loss_normalizing_factor, batch)

    sequence_lengths = (batch['decoder_target_tokens'] > 0).astype(
        jnp.int32).sum(axis=-1)
    assert self._num_latents == logits.shape[-2]
    q_end = jnp.maximum(self._num_latents, sequence_lengths)
    q_start = q_end - self._num_latents

    targets = jax.vmap(
        functools.partial(
            lax.dynamic_slice_in_dim, slice_size=self._num_latents,
            axis=-1))(batch['decoder_target_tokens'], q_start)

    weights = jax.vmap(
        functools.partial(
            lax.dynamic_slice_in_dim, slice_size=self._num_latents,
            axis=-1))(weights, q_start)

    loss, z_loss, _ = losses.compute_weighted_cross_entropy(
        logits,
        targets=targets,
        weights=weights,
        label_smoothing=self._label_smoothing,
        z_loss=self._z_loss,
        loss_normalizing_factor=loss_normalizing_factor)
    metrics = self._compute_metrics(
        logits=logits, targets=targets, mask=weights, loss=loss, z_loss=z_loss)
    return loss, metrics

  def _compute_logits_from_slice(
      self,
      decoding_state: decoding.DecodingState,
      params: models.PyTreeDef,
      decoder_causal_attention: jnp.ndarray,
      max_decode_length: int,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Token slice to logits from decoder model."""
    # Implement a cache reset step as described in Appendix E.3 of the Perceiver
    # AR paper (https://arxiv.org/pdf/2202.07765.pdf)
    if self._decoding_latent_reset_fill:
      decoding_latent_reset_fill = self._decoding_latent_reset_fill
    else:
      decoding_latent_reset_fill = max(self._num_latents - 128,
                                       self._num_latents // 2)

    logging.info('Using a reset step latent fill of %d positions',
                 decoding_latent_reset_fill)

    def get_cache_by_layers(cache):
      return traverse_util.flatten_dict(
          cache, is_leaf=lambda k, x: 'cache_index' in x)

    def tree_map_self_att_cache(map_fn, cache):
      """Map a function over just the self-attention cache layers."""
      cache_by_layers = get_cache_by_layers(cache)
      new_cache_by_layers = {}
      for layer_name, layer_cache in cache_by_layers.items():
        # The first layer is cross-attention, so don't modify it.
        if 'layers_0' not in layer_name:
          layer_cache = jax.tree_map(map_fn, layer_cache)
        new_cache_by_layers[layer_name] = layer_cache
      return flax.core.freeze(traverse_util.unflatten_dict(new_cache_by_layers))

    def reset_step():
      assert self._num_latents > decoding_latent_reset_fill

      # Create a version of the kv cache that has
      # decoding_latent_reset_fill positions instead of self._num_latents
      # positions.
      def prepare_reset_cache(x):
        # Modify key and value, but not index.
        if x.ndim > 1 and x.shape[-1] == self._num_latents:
          return x[..., :decoding_latent_reset_fill] * 0
        else:
          return x

      reset_cache = tree_map_self_att_cache(prepare_reset_cache,
                                            decoding_state.cache)

      # Note that it's possible to reuse the cached activations for the
      # cross-attention layer, but that would be fairly difficult to do with
      # the current cache API.

      # To ensure masking is calculated correctly, construct target_ids by
      # shifting inputs left and adding a placeholder value for the current
      # position.
      target_ids = jnp.pad(decoding_state.sequences[:, 1:], [[0, 0], [0, 1]])
      target_ids = jax.vmap(lambda x, y: x.at[y].set(1))(
          target_ids, decoding_state.cur_index)

      # Do a full forward pass of the model to predict the next tokens, filling
      # in the partial cache with the smaller number of latents as wel do.
      logits, new_vars = self.module.apply(
          {
              'params': params,
              'cache': reset_cache,
          },
          decoder_input_tokens=decoding_state.sequences,
          decoder_target_tokens=target_ids,
          enable_dropout=False,
          decoder_causal_attention=decoder_causal_attention,
          decode=False,
          max_decode_length=max_decode_length,
          prefill=True,
          prefill_lengths=decoding_state.cur_index + 1,
          mutable=['cache'],
          num_latents=decoding_latent_reset_fill)

      # Now expand the kv cache size back to self._num_latents.
      def expand_reset_cache(x):
        # Modify key and value, but not index.
        if x.ndim > 1 and x.shape[-1] == decoding_latent_reset_fill:
          padding = [(0, 0)] * x.ndim
          padding[-1] = (0, self._num_latents - decoding_latent_reset_fill)
          return jnp.pad(x, padding)
        else:
          return x

      new_cache = tree_map_self_att_cache(expand_reset_cache, new_vars['cache'])

      logits_idx = jnp.minimum(logits.shape[-2] - 1, decoding_state.cur_index)
      flat_logits = jax.vmap(
          functools.partial(lax.dynamic_slice_in_dim, slice_size=1,
                            axis=-2))(logits, logits_idx)
      return flat_logits, new_cache

    def regular_step():
      flat_logits, new_vars = self.module.apply(
          {
              'params': params,
              'cache': decoding_state.cache
          },
          decoding_state.cur_token,
          decoding_state.cur_token,
          enable_dropout=False,
          decode=True,
          max_decode_length=max_decode_length,
          mutable=['cache'])
      return flat_logits, new_vars['cache']

    # Determine if a reset step is needed based on whether the kv cache in
    # a self-attention layer is full.
    needs_reset = False
    for layer_name, layer_cache in get_cache_by_layers(
        decoding_state.cache).items():
      if 'layers_0' in layer_name:
        continue
      needs_reset |= (layer_cache['cache_index'] >=
                      layer_cache['cached_key'].shape[-1]).any()

    flat_logits, new_flat_cache = lax.cond(needs_reset, reset_step,
                                           regular_step)

    # Remove sequence length dimension since it's always 1 during decoding.
    flat_logits = jnp.squeeze(flat_logits, axis=1)

    return flat_logits, new_flat_cache

  def predict_batch_with_aux(
      self,
      params: models.PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      rng: Optional[jax.random.KeyArray] = None,
      *,
      return_all_decodes: bool = False,
      num_decodes: int = 1,
      decoder_params: Optional[MutableMapping[str, Any]] = None,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Predict with prefix.

    Mostly copied from DecoderOnlyModel with minor modifications for preparing
    the tokens_ids_to_logits function.

    `decoder_params` can be used to pass dynamic configurations to
    `self.decode_fn`. An example usage is to pass different random seed (i.e.,
    `jax.random.PRNGKey(seed)` with different `seed` value). This can be done by
    setting `decoder_params['decode_rng'] = jax.random.PRNGKey(seed)`.

    Although this method is short, there are a few subtle points that. We use a
    running example to make these points clear.

    ```
    Example
      inputs = [9, 4, 6, 1]
      targets = [3, 9, 1]

      seqio.DecoderFeatureConverter will generate these set of features

         decoder_target_tokens = [9, 4, 6, 1, 3, 9, 1, 0, 0]
          decoder_input_tokens = [0, 9, 4, 6, 1, 3, 9, 1, 0]
      decoder_causal_attention = [1, 1, 1, 1, 1, 0, 0, 0, 0]

      The output of this function is (`a` through `e` are the sampled token
      ids):

             sampled_sequences = [9, 4, 6, 1, a, b, c, d, e].
    ```

    Given these set of features, we make a few important observation.

    1) When a decoder-only model is used for a supervised learning with "inputs"
       and "targets", one way to handle this is to concatenate the "inputs" and
       "targets". For training, we use teacher forcing for the entire
       concatenated sequence. For inference, on the other hand, we don't have
       the targets. This requires that we use teacher forcing on the "inputs"
       portion while using the generated token as the input token for the next
       decoding step. For evaluation, we do have "targets" but we only want to
       use them for computing metrics, i.e., by comparing to the sequence
       generated by the model.

       This function is currently used for evaluation mode, but by ignoring
       "targets", it can be extended for the inference mode.

    2) During evaluation mode, the targets portion is zeroed out and they are
       filled with the sampled token ids. The inputs portion is kept intact.

    3) Note that `decoder_causal_attention` has an additional 1 after the final
       "inputs" token. This is because the position where the last "inputs"
       token (in this case 1) is input and the output is the first "target"
       token (in this case 3) can be included in the non-causal attention
       region.

       This results in an alignment between `decoder_input_tokens` and
       `decoder_causal_attention` because the former is shifted to the right by
       one position. So we use `decoder_causal_attention` as a binary mask to
       zero out the target tokens in `decoder_input_tokens`.

    Note:
      In order to use a custom self._decode_fn with this model it must support:

      1) Decoding from a partially decoded state by accepting a vector of
         `initial_indices` that specify where in the input to start decoding
         from.
      2) Using a vector as the loop counter to support different examples being
         a different number of steps into their decoding loop.
      3) Be able to handle one batch element reaching `max_decode_length`
         before the others without it causing the model to prematurely stop
         decoding.

    Args:
      params: Model parameters.
      batch: Batch element with the model features specified in
        seqio.DecoderFeatureConverter.
      rng: An optional RNG key to use during prediction, which is passed as
        'decode_rng' to the decoding function.
      return_all_decodes: If True, will return all batch_size * num_decodes
        samples from the model as an array of shape [batch_size, num_decodes,
        sequence_length]. Otherwise returns only the most likely samples as an
        array of shape [batch_size, sequence_length].
      num_decodes: Number of decoded sequences to be returned.
      decoder_params: Additional (model-independent) parameters for the decoder.

    Returns:
      Sampled sequences, an array of shape [batch, max_decode_length].
    """
    if 'decoder_causal_attention' not in batch:
      raise ValueError(
          'Batch does not have the right format for text generation: probably '
          'because `task_feature_lengths` passed to the feature converter does '
          'not have both `inputs` and `targets`.')
    # We can use the decoder causal attention mask to tell how long the inputs
    # are. The causal mask has a 1 for all the input tokens (and one more to
    # cover the original BOS token, created by shifting the inputs one to the
    # right) so we need to delete one.
    inputs_lengths = jnp.sum(batch['decoder_causal_attention'], axis=1) - 1

    # since decoder_input_tokens is shifted to the right and
    # `decoder_causal_attention` has one more 1 than the number of inputs
    # tokens, this masks out targets portion of the decoder_input_tokens.
    inputs = batch['decoder_input_tokens'] * batch['decoder_causal_attention']

    # TODO: Minor decoding performance improvement: Ideally
    # _compute_kv_cache would prefill the cache with enough space left over to
    # not immediately trigger a cache reset step if the sequence length is
    # already longer than self._num_latents.

    prefilled_cache = self._compute_kv_cache(params, inputs, inputs_lengths,
                                             batch['decoder_causal_attention'])

    target_shape = batch['decoder_input_tokens'].shape
    max_decode_length = target_shape[1]

    tokens_ids_to_logits = functools.partial(
        self._compute_logits_from_slice,
        params=params,
        decoder_causal_attention=batch['decoder_causal_attention'],
        max_decode_length=max_decode_length)

    if decoder_params is None:
      decoder_params = {}
    if rng is not None:
      if decoder_params.get('decode_rng') is not None:
        raise ValueError(
            f'Got RNG both from the `rng` argument ({rng}) and '
            f"`decoder_params['decode_rng']` ({decoder_params['decode_rng']}). "
            'Please specify one or the other.')
      decoder_params['decode_rng'] = rng

    # Using the above-defined single-step decoder function, run temperature
    # sampling with the prefix.
    # [batch, max_decode_length]
    scanned = hasattr(self.module, 'scan_layers') and self.module.scan_layers

    if 'eos_id' not in decoder_params:
      decoder_params['eos_id'] = self.output_vocabulary.eos_id
    decoded_sequences, scores = self._decode_fn(
        inputs=inputs,
        cache=prefilled_cache,
        tokens_to_logits=tokens_ids_to_logits,
        num_decodes=num_decodes,
        initial_index=inputs_lengths,
        cache_offset=1 if scanned else 0,
        **decoder_params)

    if not return_all_decodes:
      # Search returns [n_batch, n_beam/decodes, n_length] with the beam/decode
      # dimension sorted in increasing order of log-probability.
      # `scores` is [batch, beam/decode_size]
      # We take the highest scoring sequence (-1) and its score
      decoded_sequences = decoded_sequences[:, -1, :]
      # Beam search returns []
      aux = {'scores': scores[:, -1]}
    else:
      # We return all samples and scores, rather than just the top ones.
      aux = {'scores': scores}

    return models.remove_prefix(decoded_sequences, inputs_lengths), aux
