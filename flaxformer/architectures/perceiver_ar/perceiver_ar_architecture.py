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

"""Perceiver AR Architecture implementation.

As described in:
"General-purpose, long-context autoregressive modeling with Perceiver AR"
https://arxiv.org/abs/2202.07765
"""

import dataclasses
from typing import List, Optional, Tuple

from flax import linen as nn
import jax.numpy as jnp

from flaxformer.architectures.perceiver_ar import slicing
from flaxformer.architectures.t5 import parallel_fused_decoder
from flaxformer.architectures.t5 import t5_architecture
from flaxformer.types import Array


@dataclasses.dataclass(frozen=True)
class PerceiverARTransparentLayerSequence:
  """Perceiver AR version of TransparentLayerSequence that manages slicing.

  The decoder_mask is different for the first layer vs. the remaining layers.
  Similar for the logit mask and prefill lengths. It's better to do the change
  outside of the scan-over-layers so that it is done only once.

  Attributes:
    layers: List of nn.Modules, which should be owned by a parent Flax module.
    num_latents: Number of latents and outputs.
  """
  layers: List[nn.Module]
  num_latents: int

  def __call__(self,
               inputs: Array,
               encoded,
               decoder_mask=None,
               encoder_decoder_mask=None,
               *,
               logit_mask=None,
               enable_dropout: bool = True,
               decode: bool = False,
               max_decode_length: Optional[int] = None,
               prefill: bool = False,
               prefill_lengths: Optional[Array] = None,
               num_latents: Optional[int] = None,
               sequence_lengths: Optional[Array] = None) -> Array:
    """Applies all Transformer layers to the inputs sequentially.

    Args:
      inputs: Input data for decoder with shape [batch_size, decoder_seq_length,
        decoder_hidden_size].
      encoded: required to be None, block is Decoder only, only kept for
        __call__ signature uniformity.
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: required to be None, block is Decoder only, only
        kept for __call__ signature uniformity.
      logit_mask: a mask (e.g., padding logit mask) to be applied to the
        attention logits.
      enable_dropout: Enables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.
      max_decode_length: An optional integer specifying the maximum decoding
        length. Note that this is only used for defining the relative position
        embedding parameters.
      prefill: Whether to run a partial sequence to prefill the cache.
      prefill_lengths: The length of each partial sequence we are filling in the
        cache, lengths are inferred from the mask if not provided.
      num_latents: Used to override the number of output Perceiver AR latents
        during decoding.
      sequence_lengths: Lengths of all target sequences. Required for Perceiver
        AR operation.

    Returns:
      The encoded inputs <float>[..., seq_len, hidden_size].
    """

    current_activations = inputs
    for layer in self.layers:
      layer_decoder_mask = decoder_mask

      num_latents = num_latents or self.num_latents

      if layer_decoder_mask is not None:
        if layer_decoder_mask.shape[-2] != num_latents:
          # Slice queries to match number of latents.
          layer_decoder_mask = slicing.slice_sequences_vmap(
              layer_decoder_mask,
              sequence_lengths,
              num_latents,
              axis_within_vmap=-2)

        if layer_decoder_mask.shape[-1] != current_activations.shape[-2]:
          # If we're in the self-attention stack, then kv should also be sliced.
          layer_decoder_mask = slicing.slice_sequences_vmap(
              layer_decoder_mask,
              sequence_lengths,
              num_latents,
              axis_within_vmap=-1)

      layer_prefill_lengths = prefill_lengths
      if prefill:
        if layer_prefill_lengths is None:
          layer_prefill_lengths = sequence_lengths

        # Ensure prefill_lengths isn't longer than the input length.
        # For Perceiver AR, this can happen in the self-attention stack, which
        # is narrower than the actual sequence length.
        layer_prefill_lengths = jnp.minimum(current_activations.shape[-2],
                                            layer_prefill_lengths)

      layer_logit_mask = logit_mask
      if (layer_logit_mask is not None and
          layer_logit_mask.shape[-2] != current_activations.shape[-2]):
        layer_logit_mask = slicing.slice_sequences_vmap(
            layer_logit_mask,
            sequence_lengths,
            current_activations.shape[-2],
            axis_within_vmap=0)

      current_activations = layer(
          current_activations,
          encoded,
          layer_decoder_mask,
          encoder_decoder_mask,
          logit_mask=layer_logit_mask,
          enable_dropout=enable_dropout,
          decode=decode,
          max_decode_length=max_decode_length,
          prefill=prefill,
          prefill_lengths=layer_prefill_lengths,
          num_latents=num_latents,
          sequence_lengths=sequence_lengths)
    return current_activations


class Decoder(t5_architecture.Decoder):
  """Perceiver AR Decoder.

  Attributes:
    num_latents: Number of latents for queries and number of output latents.
  """
  # num_latents is actually required, but has to be marked as optional because
  # we don't yet require Python 3.10, which provides keyword-only dataclasses.
  num_latents: Optional[int] = None

  def setup(self):
    if self.num_latents is None:
      raise ValueError('num_latents must be specified.')

    super().setup()

  def _setup_layer_sequence(self):
    lyrf = lambda: self.layer_factory(  # pylint: disable=g-long-lambda
        shared_relative_position_bias=self.relpos_bias)
    lyrf = t5_architecture.maybe_remat(
        lyrf,
        self.layer_remat,
        self.scan_layers,
        static_argnums=(5, 6, 7, 8, 9, 10))

    if not self.scan_layers:
      self.layers = [lyrf() for _ in range(self.num_layers)]
      return PerceiverARTransparentLayerSequence(self.layers, self.num_latents)
    else:
      # Create a non-scanned version of lyrf to use for the first layer.
      lyrf_notscanned = lambda: self.layer_factory(  # pylint: disable=g-long-lambda
          shared_relative_position_bias=self.relpos_bias,
          scanned=False)
      lyrf_notscanned = t5_architecture.maybe_remat(
          lyrf_notscanned,
          self.layer_remat,
          self.scan_layers,
          static_argnums=(5, 6, 7, 8, 9, 10))

      self.layers = [
          lyrf_notscanned(),
          self._construct_scanned_decoder(
              lyrf, self.num_layers - 1, num_broadcast_args=11)
      ]
      return PerceiverARTransparentLayerSequence(self.layers, self.num_latents)

  def decode_from_continuous_inputs(self,
                                    embedded_inputs,
                                    encoder_outputs,
                                    decoder_positions=None,
                                    decoder_mask=None,
                                    encoder_decoder_mask=None,
                                    logit_mask=None,
                                    *,
                                    enable_dropout: bool = True,
                                    decode: bool = False,
                                    max_decode_length: Optional[int] = None,
                                    prefill: bool = False,
                                    prefill_lengths: Optional[Array] = None,
                                    num_latents: Optional[int] = None,
                                    sequence_lengths: Optional[Array] = None):
    """Applies the decoder on the continuous (embedded) inputs."""
    if decoder_positions is not None:
      raise NotImplementedError('Perceiver AR does not yet support packing.')

    # sequence_lengths is required, but has to be defined as optional to
    # maintain API compatibility.
    if sequence_lengths is None:
      raise ValueError('sequence_lengths must be supplied fo Perceiver AR.')

    num_latents = num_latents or self.num_latents

    # If encoded is not given, this block is decoder only and does not contain
    # attention from decoder to encoder.
    if encoder_outputs is not None:
      assert encoder_outputs.ndim == 3  # (batch, len, depth)

    # Apply the decoder layers, attending to the encoder outputs (if provided),
    # and attending to previous decoder inputs (by masking future inputs).
    decoder_outputs = self.decoder(
        embedded_inputs,
        encoder_outputs,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        logit_mask=logit_mask,
        enable_dropout=enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length,
        prefill=prefill,
        prefill_lengths=prefill_lengths,
        num_latents=num_latents,
        sequence_lengths=sequence_lengths)

    if self.scan_layers:
      decoder_outputs = decoder_outputs[0]

    # Output length should always be <= the number of latents regardless of
    # input length or configured number of latents. During training it will be
    # the same. During fast decoding, it may just be 1.
    assert decoder_outputs.shape[-2] <= num_latents

    # Post-process final decoder layer outputs.
    decoder_outputs = self.decoder_norm(decoder_outputs)
    decoder_outputs = self.output_dropout(
        decoder_outputs, deterministic=not enable_dropout)

    # Slice logit_mask to match output positions.
    if logit_mask is not None:
      if logit_mask.shape[-2] != decoder_outputs.shape[-2]:
        logit_mask = slicing.slice_sequences_vmap(
            logit_mask,
            sequence_lengths,
            decoder_outputs.shape[-2],
            axis_within_vmap=-2)
      decoder_outputs = logit_mask * decoder_outputs

    if self.sow_intermediates:
      self.sow('intermediates', 'pre_logits_layer', decoder_outputs)

    # Decoded Logits
    if self.logits_dense is not None:
      logits = self.logits_dense(decoder_outputs)
    else:
      # Use the transpose of embedding matrix for logit transform.
      logits = self.embedder.embedders['token_ids'].attend(decoder_outputs)  # pytype: disable=attribute-error
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(decoder_outputs.shape[-1])

    if self.sow_intermediates:
      self.sow('intermediates', 'logits', logits)
    return logits


class DecoderOnly(t5_architecture.DecoderOnly):
  """Perceiver AR Decoder-only model."""

  def __call__(self,
               decoder_input_tokens,
               decoder_target_tokens,
               decoder_segment_ids=None,
               decoder_positions=None,
               decoder_causal_attention=None,
               **kwargs):
    """Applies Perceiver AR Decoder-only model on the inputs."""
    # Perceiver AR operation does not support packing.
    if decoder_positions is not None:
      raise NotImplementedError(
          'decoder_positions is provided, but Perceiver AR does not yet '
          'support packing.')
    if decoder_segment_ids is not None:
      raise NotImplementedError(
          'decoder_segment_ids is provided, but Perceiver AR does not yet '
          'support packing.')

    # Calculate sequence lengths based on target tokens.
    sequence_lengths = (decoder_target_tokens > 0).sum(axis=-1).astype(
        jnp.int32)

    return super().__call__(
        decoder_input_tokens,
        decoder_target_tokens,
        decoder_segment_ids,
        decoder_positions,
        decoder_causal_attention,
        sequence_lengths=sequence_lengths,
        **kwargs)


def _create_residuals_and_queries(
    layer_input: Array, x: Array, logit_mask, *, num_latents: Optional[Array],
    sequence_lengths: Array) -> Tuple[Array, Array, Array]:
  """Slice layer inputs to get versions to use as queries."""
  if x.shape[-2] > num_latents:
    layer_input_residuals = slicing.slice_sequences_xmap(
        layer_input, sequence_lengths, num_latents, axis_within_xmap=0)
    x_queries = slicing.slice_sequences_xmap(
        x, sequence_lengths, num_latents, axis_within_xmap=0)
  else:
    layer_input_residuals = layer_input
    x_queries = x

  if logit_mask.shape[-2] > num_latents:
    logit_mask_queries = slicing.slice_sequences_vmap(
        logit_mask, sequence_lengths, num_latents, axis_within_vmap=0)
  else:
    logit_mask_queries = logit_mask

  return layer_input_residuals, x_queries, logit_mask_queries


class ParallelFusedDecoderLayer(parallel_fused_decoder.ParallelFusedDecoderLayer
                               ):
  """Decoder layer subclass that includes Perceiver AR slicing."""
  # num_latents is actually required, but has to be marked as optional because
  # we don't yet require Python 3.10, which provides keyword-only dataclasses.
  num_latents: Optional[int] = None

  def setup(self):
    if self.num_latents is None:
      raise ValueError('num_latents must be specified.')

    super().setup()

    if self.relpos_bias is not None:
      raise NotImplementedError(
          'Relative position bias support not yet implemented for Perceiver AR.'
      )

  def _create_residuals_and_queries(
      self, layer_input: Array, x: Array, logit_mask: Array, *,
      num_latents: Optional[Array],
      sequence_lengths: Array) -> Tuple[Array, Array, Array]:
    num_latents = num_latents or self.num_latents

    return _create_residuals_and_queries(
        layer_input=layer_input,
        x=x,
        logit_mask=logit_mask,
        num_latents=num_latents,
        sequence_lengths=sequence_lengths)

  @nn.nowrap
  def __call__(self,
               targets,
               encoded,
               decoder_mask=None,
               encoder_decoder_mask=None,
               *,
               logit_mask=None,
               enable_dropout: bool = True,
               decode: bool = False,
               max_decode_length: Optional[int] = None,
               prefill: bool = False,
               prefill_lengths: Optional[Array] = None,
               num_latents: Optional[int] = None,
               sequence_lengths: Optional[Array] = None) -> Array:
    """Applies ParallelFusedDecoder1DBlock module.

    Redefined here to spell out the arguments that will be passed as **kwargs
    to the superclass. This allows flaxformer to know the order of arguments
    for static_argnums.

    Args:
      targets: Input data for decoder with shape [batch_size,
        decoder_seq_length, decoder_hidden_size].
      encoded: required to be None, block is Decoder only, only kept for
        __call__ signature uniformity.
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: required to be None, block is Decoder only, only
        kept for __call__ signature uniformity.
      logit_mask: a mask (e.g., padding logit mask) to be applied to the
        attention logits.
      enable_dropout: Enables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.
      max_decode_length: An optional integer specifying the maximum decoding
        length. Note that this is only used for defining the relative position
        embedding parameters.
      prefill: Whether to run a partial sequence to prefill the cache.
      prefill_lengths: The length of each partial sequence we are filling in the
        cache, lengths are inferred from the mask if not provided.
      num_latents: Used to override the number of output Perceiver AR latents
        during decoding.
      sequence_lengths: Lengths of all target sequences. Required for Perceiver
        AR operation.

    Returns:
      Output after transformer encoder-decoder block.
    """
    return super().__call__(
        targets,
        encoded,
        decoder_mask,
        encoder_decoder_mask,
        logit_mask=logit_mask,
        enable_dropout=enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length,
        prefill=prefill,
        prefill_lengths=prefill_lengths,
        num_latents=num_latents,
        sequence_lengths=sequence_lengths)


class DecoderLayer(t5_architecture.DecoderLayer):
  """Decoder layer subclass that includes Perceiver AR slicing."""
  # num_latents is actually required, but has to be marked as optional because
  # we don't yet require Python 3.10, which provides keyword-only dataclasses.
  num_latents: Optional[int] = None

  def setup(self):
    if self.num_latents is None:
      raise ValueError('num_latents must be specified.')

    super().setup()

    if self.relpos_bias is not None:
      raise NotImplementedError(
          'Relative position bias support not yet implemented for Perceiver AR.'
      )

  def _create_residuals_and_queries(
      self, layer_input: Array, x: Array, logit_mask: Array, *,
      num_latents: Optional[Array],
      sequence_lengths: Array) -> Tuple[Array, Array, Array]:
    num_latents = num_latents or self.num_latents

    return _create_residuals_and_queries(
        layer_input=layer_input,
        x=x,
        logit_mask=logit_mask,
        num_latents=num_latents,
        sequence_lengths=sequence_lengths)

  @nn.nowrap
  def __call__(self,
               targets,
               encoded,
               decoder_mask=None,
               encoder_decoder_mask=None,
               *,
               logit_mask=None,
               enable_dropout: bool = True,
               decode: bool = False,
               max_decode_length: Optional[int] = None,
               prefill: bool = False,
               prefill_lengths: Optional[Array] = None,
               num_latents: Optional[int] = None,
               sequence_lengths: Optional[Array] = None) -> Array:
    """Applies EncoderDecoder1DBlock module.

    Redefined here to spell out the arguments that will be passed as **kwargs
    to the superclass. This allows flaxformer to know the order of arguments
    for static_argnums.

    Args:
      targets: Input data for decoder with shape [batch_size,
        decoder_seq_length, decoder_hidden_size].
      encoded: Input data from encoder with shape [batch_size,
        encoder_seq_length, decoder_hidden_size]. If None, block is Decoder
        only.
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask with shape [
        batch_size, 1, decoder_seq_length, encoder_seq_length].
      logit_mask: a mask (e.g., padding logit mask) to be applied to the
        attention logits.
      enable_dropout: Enables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.
      max_decode_length: An optional integer specifying the maximum decoding
        length. Note that this is only used for defining the relative position
        embedding parameters.
      prefill: Whether to run a partial sequence to prefill the cache.
      prefill_lengths: The length of each partial sequence we are filling in the
        cache, lengths are inferred from the mask if not provided.
      num_latents: Used to override the number of output Perceiver AR latents
        during decoding.
      sequence_lengths: Lengths of all target sequences. Required for Perceiver
        AR operation.

    Returns:
      Output after transformer encoder-decoder block.
    """
    return super().__call__(
        targets,
        encoded,
        decoder_mask,
        encoder_decoder_mask,
        logit_mask=logit_mask,
        enable_dropout=enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length,
        prefill=prefill,
        prefill_lengths=prefill_lengths,
        num_latents=num_latents,
        sequence_lengths=sequence_lengths)
