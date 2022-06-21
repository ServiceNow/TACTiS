"""
Copyright 2022 ServiceNow
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

>> The highest level model for TACTiS, which contains both its encoder and decoder.
"""


import torch
from torch import nn
from typing import Optional, Dict, Any, Tuple
import numpy

from .encoder import Encoder, TemporalEncoder
from .decoder import CopulaDecoder, GaussianDecoder


class PositionalEncoding(nn.Module):
    """
    A class implementing the positional encoding for Transformers described in Vaswani et al. (2017).
    Somewhat generalized to allow unaligned or unordered time steps, as long as the time steps are integers.

    Adapted from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_length: int = 5000):
        """
        Parameters:
        -----------
        embedding_dim: int
            The dimension of the input and output embeddings for this encoding.
        dropout: float, default to 0.1
            Dropout parameter for this encoding.
        max_length: int, default to 5000
            The maximum time steps difference which will have to be handled by this encoding.
        """
        super().__init__()

        assert embedding_dim % 2 == 0, "PositionEncoding needs an even embedding dimension"

        self.dropout = nn.Dropout(p=dropout)

        pos_encoding = torch.zeros(max_length, embedding_dim)
        possible_pos = torch.arange(0, max_length, dtype=torch.float)[:, None]
        factor = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float) * (-numpy.log(10000.0) / embedding_dim))

        # Alternate between using sine and cosine
        pos_encoding[:, 0::2] = torch.sin(possible_pos * factor)
        pos_encoding[:, 1::2] = torch.cos(possible_pos * factor)

        # Register as a buffer, to automatically be sent to another device if the model is sent there
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, input_encoded: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        input_encoded: torch.Tensor [batch, series, time steps, embedding dimension]
            An embedding which will be modified by the position encoding.
        timesteps: torch.IntTensor [batch, series, time steps] or [batch, 1, time steps]
            The time step for each entry in the input.

        Returns:
        --------
        output_encoded: torch.Tensor [batch, series, time steps, embedding dimension]
            The modified embedding.
        """
        # Use the time difference between the first time step of each batch and the other time steps.
        # min returns two outputs, we only keep the first.
        min_t = timesteps.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        delta_t = timesteps - min_t

        output_encoded = input_encoded + self.pos_encoding[delta_t]
        return self.dropout(output_encoded)


class TACTiS(nn.Module):
    """
    The top-level module for TACTiS.

    The role of this module is to handle everything outside of the encoder and decoder.
    This consists mainly the data manipulation ahead of the encoder and after the decoder.
    """

    def __init__(
        self,
        num_series: int,
        series_embedding_dim: int,
        input_encoder_layers: int,
        bagging_size: Optional[int] = None,
        input_encoding_normalization: bool = True,
        positional_encoding: Optional[Dict[str, Any]] = None,
        encoder: Optional[Dict[str, Any]] = None,
        temporal_encoder: Optional[Dict[str, Any]] = None,
        copula_decoder: Optional[Dict[str, Any]] = None,
        gaussian_decoder: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        assert (encoder is not None) + (temporal_encoder is not None) == 1, "Must select exactly one type of encoder"
        assert (copula_decoder is not None) + (
            gaussian_decoder is not None
        ) == 1, "Must select exactly one type of decoder"

        assert (not bagging_size) or bagging_size <= num_series, "Bagging size must not be above number of series"

        self.num_series = num_series
        self.bagging_size = bagging_size
        self.series_embedding_dim = series_embedding_dim
        self.input_encoder_layers = input_encoder_layers
        self.input_encoding_normalization = input_encoding_normalization

        self.series_encoder = nn.Embedding(num_embeddings=num_series, embedding_dim=self.series_embedding_dim)

        if encoder is not None:
            self.encoder = Encoder(**encoder)
        if temporal_encoder is not None:
            self.encoder = TemporalEncoder(**temporal_encoder)

        if copula_decoder is not None:
            self.decoder = CopulaDecoder(input_dim=self.encoder.embedding_dim, **copula_decoder)
        if gaussian_decoder is not None:
            self.decoder = GaussianDecoder(input_dim=self.encoder.embedding_dim, **gaussian_decoder)

        if positional_encoding is not None:
            self.time_encoding = PositionalEncoding(self.encoder.embedding_dim, **positional_encoding)
        else:
            self.time_encoding = None

        elayers = nn.ModuleList([])
        for i in range(self.input_encoder_layers):
            if i == 0:
                elayers.append(
                    nn.Linear(self.series_embedding_dim + 1, self.encoder.embedding_dim)
                )  # +1 is for the mask
            else:
                elayers.append(nn.Linear(self.encoder.embedding_dim, self.encoder.embedding_dim))
            elayers.append(nn.ReLU())
        self.input_encoder = nn.Sequential(*elayers)

    @staticmethod
    def _apply_bagging(
        bagging_size,
        hist_time: torch.Tensor,
        hist_value: torch.Tensor,
        pred_time: torch.Tensor,
        pred_value: torch.Tensor,
        series_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Only keep a small number of series for each of the input tensors.
        Which series will be kept is randomly selected for each batch. The order is not preserved.

        Parameters:
        -----------
        bagging_size: int
            How many series to keep
        hist_time: Tensor [batch, series, time steps]
            A tensor containing the time steps associated with the values of hist_value.
        hist_value: Tensor [batch, series, time steps]
            A tensor containing the values that will be available at inference time.
        pred_time: Tensor [batch, series, time steps]
            A tensor containing the time steps associated with the values of pred_value.
        pred_value: Tensor [batch, series, time steps]
            A tensor containing the values that the model should learn to forecast at inference time.
        series_emb: Tensor [batch, series, embedding size]
            An embedding for each series, expanded over the batches.

        Returns:
        --------
        hist_time: Tensor [batch, series, time steps]
        hist_value: Tensor [batch, series, time steps]
        pred_time: Tensor [batch, series, time steps]
        pred_value: Tensor [batch, series, time steps]
        series_emb: Tensor [batch, series, embedding size]
            A subset of the input data, where only the given number of series is kept for each batch.
        """
        num_batches = hist_time.shape[0]
        num_series = hist_time.shape[1]

        # Make sure to have the exact same bag for all series
        bags = [torch.randperm(num_series)[0:bagging_size] for _ in range(num_batches)]

        hist_time = torch.stack([hist_time[i, bags[i], :] for i in range(num_batches)], dim=0)
        hist_value = torch.stack([hist_value[i, bags[i], :] for i in range(num_batches)], dim=0)
        pred_time = torch.stack([pred_time[i, bags[i], :] for i in range(num_batches)], dim=0)
        pred_value = torch.stack([pred_value[i, bags[i], :] for i in range(num_batches)], dim=0)
        series_emb = torch.stack([series_emb[i, bags[i], :] for i in range(num_batches)], dim=0)

        return hist_time, hist_value, pred_time, pred_value, series_emb

    def forward(
        self, hist_time: torch.Tensor, hist_value: torch.Tensor, pred_time: torch.Tensor, pred_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss function of the model.

        Parameters:
        -----------
        hist_time: Tensor [batch, series, time steps] or [batch, 1, time steps] or [batch, time steps]
            A tensor containing the time steps associated with the values of hist_value.
            If the series dimension is singleton or missing, then the time steps are taken as constant across all series.
        hist_value: Tensor [batch, series, time steps]
            A tensor containing the values that will be available at inference time.
        pred_time: Tensor [batch, series, time steps] or [batch, 1, time steps] or [batch, time steps]
            A tensor containing the time steps associated with the values of pred_value.
            If the series dimension is singleton or missing, then the time steps are taken as constant across all series.
        pred_value: Tensor [batch, series, time steps]
            A tensor containing the values that the model should learn to forecast at inference time.

        Returns:
        --------
        loss: torch.Tensor [batch]
            The loss function of TACTiS, with lower values being better.
        """
        num_batches = hist_value.shape[0]
        num_series = hist_value.shape[1]
        num_hist_timesteps = hist_value.shape[2]
        num_pred_timesteps = pred_value.shape[2]
        device = hist_value.device

        # Gets the embedding for each series [batch, series, embedding size]
        # Expand over batches to be compatible with the bagging procedure, which select different series for each batch
        series_emb = self.series_encoder(torch.arange(hist_value.shape[1], device=device))
        series_emb = series_emb[None, :, :].expand(num_batches, -1, -1)

        # Make sure that both time tensors are in the correct format
        if len(hist_time.shape) == 2:
            hist_time = hist_time[:, None, :]
        if len(pred_time.shape) == 2:
            pred_time = pred_time[:, None, :]
        if hist_time.shape[1] == 1:
            hist_time = hist_time.expand(-1, num_series, -1)
        if pred_time.shape[1] == 1:
            pred_time = pred_time.expand(-1, num_series, -1)

        if self.bagging_size:
            hist_time, hist_value, pred_time, pred_value, series_emb = self._apply_bagging(
                self.bagging_size, hist_time, hist_value, pred_time, pred_value, series_emb
            )

        hist_encoded = torch.cat(
            [
                hist_value[:, :, :, None],
                series_emb[:, :, None, :].expand(num_batches, -1, num_hist_timesteps, -1),
                torch.ones(num_batches, num_series, num_hist_timesteps, device=device),
            ],
            dim=3,
        )
        # For the prediction embedding, replace the values by zeros, since they won't be available during sampling
        pred_encoded = torch.cat(
            [
                torch.zeros(num_batches, num_series, num_pred_timesteps, device=device),
                series_emb[:, :, None, :].expand(num_batches, -1, num_pred_timesteps, -1),
                torch.zeros(num_batches, num_series, num_pred_timesteps, device=device),
            ],
            dim=3,
        )

        encoded = torch.cat([hist_encoded, pred_encoded], dim=2)
        encoded = self.input_encoder(encoded)
        if self.input_encoding_normalization:
            encoded *= self.encoder.embedding_dim**0.5

        # Add the time encoding here after the input encoding to be compatible with how positional encoding is used.
        # Adjustments may be required for other ways to encode time.
        if self.time_encoding:
            timesteps = torch.cat([hist_time, pred_time], dim=2)
            encoded = self.time_encoding(encoded, timesteps.to(int))

    def sample(
        self, num_samples: int, hist_time: torch.Tensor, hist_value: torch.Tensor, pred_time: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate the given number of samples from the forecasted distribution.

        Parameters:
        -----------
        num_samples: int
            How many samples to generate, must be >= 1.
        hist_time: Tensor [batch, series, time steps] or [batch, 1, time steps] or [batch, time steps]
            A tensor containing the times associated with the values of hist_value.
            If the series dimension is singleton or missing, then the time steps are taken as constant across all series.
        hist_value: Tensor [batch, series, time steps]
            A tensor containing the available values
        pred_time: Tensor [batch, series, time steps] or [batch, 1, time steps] or [batch, time steps]
            A tensor containing the times at which we want forecasts.
            If the series dimension is singleton or missing, then the time steps are taken as constant across all series.

        Returns:
        --------
        samples: torch.Tensor [batch, series, time steps, samples]
            Samples from the forecasted distribution.
        """
