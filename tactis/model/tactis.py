"""
Copyright 2023 ServiceNow
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


from typing import Any, Dict, Optional, Tuple

import numpy
import torch
from copy import deepcopy
from torch import nn

from .decoder import CopulaDecoder
from .encoder import Encoder, TemporalEncoder


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

    def get_pos_encoding(self, timesteps: torch.IntTensor) -> torch.Tensor:
        min_t = timesteps.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        delta_t = timesteps - min_t

        return self.pos_encoding[delta_t]


class NormalizationIdentity:
    """
    Trivial normalization helper. Do nothing to its data.
    """

    def __init__(self, hist_value: torch.Tensor):
        """
        Parameters:
        -----------
        hist_value: torch.Tensor [batch, series, time steps]
            Historical data which can be used in the normalization.
        """
        pass

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        """
        Normalize the given values according to the historical data sent in the constructor.

        Parameters:
        -----------
        value: Tensor [batch, series, time steps]
            A tensor containing the values to be normalized.

        Returns:
        --------
        norm_value: Tensor [batch, series, time steps]
            The normalized values.
        """
        return value

    def denormalize(self, norm_value: torch.Tensor) -> torch.Tensor:
        """
        Undo the normalization done in the normalize() function.

        Parameters:
        -----------
        norm_value: Tensor [batch, series, time steps, samples]
            A tensor containing the normalized values to be denormalized.

        Returns:
        --------
        value: Tensor [batch, series, time steps, samples]
            The denormalized values.
        """
        return norm_value


class NormalizationStandardization:
    """
    Normalization helper for the standardization.

    The data for each batch and each series will be normalized by:
    - substracting the historical data mean,
    - and dividing by the historical data standard deviation.

    Use a lower bound of 1e-8 for the standard deviation to avoid numerical problems.
    """

    def __init__(self, hist_value: torch.Tensor):
        """
        Parameters:
        -----------
        hist_value: torch.Tensor [batch, series, time steps]
            Historical data which can be used in the normalization.
        """
        std, mean = torch.std_mean(hist_value, dim=2, unbiased=True, keepdim=True)
        self.std = std.clamp(min=1e-8)
        self.mean = mean

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        """
        Normalize the given values according to the historical data sent in the constructor.

        Parameters:
        -----------
        value: Tensor [batch, series, time steps]
            A tensor containing the values to be normalized.

        Returns:
        --------
        norm_value: Tensor [batch, series, time steps]
            The normalized values.
        """
        value = (value - self.mean) / self.std
        return value

    def denormalize(self, norm_value: torch.Tensor) -> torch.Tensor:
        """
        Undo the normalization done in the normalize() function.

        Parameters:
        -----------
        norm_value: Tensor [batch, series, time steps, samples]
            A tensor containing the normalized values to be denormalized.

        Returns:
        --------
        value: Tensor [batch, series, time steps, samples]
            The denormalized values.
        """
        norm_value = (norm_value * self.std[:, :, :, None]) + self.mean[:, :, :, None]
        return norm_value


class TACTiS(nn.Module):
    """
    The top-level module for TACTiS.

    The role of this module is to handle everything outside of the encoder and decoder.
    This consists mainly the data manipulation ahead of the encoder and after the decoder.
    """

    def __init__(
        self,
        num_series: int,
        flow_series_embedding_dim: int,
        copula_series_embedding_dim: int,
        flow_input_encoder_layers: int,
        copula_input_encoder_layers: int,
        bagging_size: Optional[int] = None,
        input_encoding_normalization: bool = True,
        data_normalization: str = "none",
        loss_normalization: str = "series",
        positional_encoding: Optional[Dict[str, Any]] = None,
        flow_encoder: Optional[Dict[str, Any]] = None,
        copula_encoder: Optional[Dict[str, Any]] = None,
        flow_temporal_encoder: Optional[Dict[str, Any]] = None,
        copula_temporal_encoder: Optional[Dict[str, Any]] = None,
        copula_decoder: Optional[Dict[str, Any]] = None,
        skip_copula: bool = True,
        experiment_mode: str = "forecasting",
    ):
        """
        Parameters:
        -----------
        num_series: int
            Number of series of the data which will be sent to the model. Required for the series embedding.
        bagging_size: Optional[int], default to None
            If set, the loss() method will only consider a random subset of the series at each call.
            The number of series kept is the value of this parameter.
        subset_series: default to None
            If set, should be a list of series to consider during training/evaluation. Used for toy experiments but with large datasets.
            Most importantly, different from bagging, here, the series is also used at inference.
        input_encoding_normalization: bool, default to True
            If true, the encoded input values (prior to the positional encoding) are scaled
            by the square root of their dimensionality.
        data_normalization: str ["", "none", "standardization"], default to "series"
            How to normalize the input values before sending them to the model.
        loss_normalization: str ["", "none", "series", "timesteps", "both"], default to "series"
            Scale the loss function by the number of series, timesteps, or both.
        positional_encoding: Optional[Dict[str, Any]], default to None
            If set to a non-None value, uses a PositionalEncoding for the time encoding.
            The options sent to the PositionalEncoding is content of this dictionary.
        copula_decoder: Optional[Dict[str, Any]], default to None
            If set to a non-None value, uses a CopulaDecoder as the decoder.
            The options sent to the CopulaDecoder is content of this dictionary.
        """
        super().__init__()

        assert copula_decoder is not None, "Must select exactly one type of decoder"

        assert (not bagging_size) or bagging_size <= num_series, "Bagging size must not be above number of series"

        data_normalization = data_normalization.lower()
        assert data_normalization in {"", "none", "standardization"}
        loss_normalization = loss_normalization.lower()
        assert loss_normalization in {"", "none", "series", "timesteps", "both"}

        self.num_series = num_series
        self.flow_series_embedding_dim = flow_series_embedding_dim
        self.copula_series_embedding_dim = copula_series_embedding_dim
        self.flow_input_encoder_layers = flow_input_encoder_layers
        self.copula_input_encoder_layers = copula_input_encoder_layers
        self.bagging_size = bagging_size
        self.input_encoding_normalization = input_encoding_normalization
        self.loss_normalization = loss_normalization
        self.positional_encoding = positional_encoding
        self.flow_encoder_args = flow_encoder
        self.copula_encoder_args = copula_encoder
        self.flow_temporal_encoder_args = flow_temporal_encoder
        self.copula_temporal_encoder_args = copula_temporal_encoder
        self.copula_decoder_args = copula_decoder

        self.skip_copula = skip_copula

        assert experiment_mode in ["forecasting", "interpolation"]
        self.experiment_mode = experiment_mode

        # Data normalization
        self.data_normalization = {
            "": NormalizationIdentity,
            "none": NormalizationIdentity,
            "standardization": NormalizationStandardization,
        }[data_normalization]

        # Actual encoder
        # Split encoder
        if flow_encoder is not None and copula_encoder is not None:
            self.flow_encoder = Encoder(**flow_encoder)
            if not self.skip_copula:
                self.copula_encoder = Encoder(**copula_encoder)
        elif flow_temporal_encoder is not None and copula_temporal_encoder is not None:
            self.flow_encoder = TemporalEncoder(**flow_temporal_encoder)
            if not self.skip_copula:
                self.copula_encoder = TemporalEncoder(**copula_temporal_encoder)
        self.flow_encoder_embedding_dim = self.flow_encoder.embedding_dim
        if not self.skip_copula:
            self.copula_encoder_embedding_dim = self.copula_encoder.embedding_dim
            copula_decoder["attentional_copula"]["input_dim"] = self.copula_encoder_embedding_dim

        # Split input encoder
        # Series encoding, Positional encoding and input encoder (that transforms [x*m, c, m] into [z])
        self.flow_series_encoder = nn.Embedding(num_embeddings=num_series, embedding_dim=self.flow_series_embedding_dim)
        if not self.skip_copula:
            self.copula_series_encoder = nn.Embedding(
                num_embeddings=num_series,
                embedding_dim=self.copula_series_embedding_dim,
            )

        if positional_encoding is not None:
            self.flow_time_encoding = PositionalEncoding(self.flow_encoder_embedding_dim, **positional_encoding)
            if not self.skip_copula:
                self.copula_time_encoding = PositionalEncoding(self.copula_encoder_embedding_dim, **positional_encoding)
        else:
            self.flow_time_encoding = None
            if not self.skip_copula:
                self.copula_time_encoding = None

        flow_elayers = nn.ModuleList([])
        for i in range(self.flow_input_encoder_layers):
            if i == 0:
                flow_elayers.append(nn.Linear(self.flow_series_embedding_dim + 2, self.flow_encoder_embedding_dim))
            else:
                flow_elayers.append(nn.Linear(self.flow_encoder_embedding_dim, self.flow_encoder_embedding_dim))
            flow_elayers.append(nn.ReLU())
        self.flow_input_encoder = nn.Sequential(*flow_elayers)

        if not self.skip_copula:
            copula_elayers = nn.ModuleList([])
            for i in range(self.copula_input_encoder_layers):
                if i == 0:
                    copula_elayers.append(
                        nn.Linear(self.copula_series_embedding_dim + 2, self.copula_encoder_embedding_dim)
                    )  # +1 for the value, +1 for the mask, and the per series embedding
                else:
                    copula_elayers.append(
                        nn.Linear(self.copula_encoder_embedding_dim, self.copula_encoder_embedding_dim)
                    )
                copula_elayers.append(nn.ReLU())
            self.copula_input_encoder = deepcopy(nn.Sequential(*copula_elayers))

        if copula_decoder is not None:
            flow_input_dim = self.flow_encoder_embedding_dim
            copula_input_dim = (
                None if self.skip_copula else self.copula_encoder_embedding_dim
            )  # Since we do not have a copula encoder yet
            self.decoder = CopulaDecoder(
                flow_input_dim=flow_input_dim,
                copula_input_dim=copula_input_dim,
                skip_copula=self.skip_copula,
                **copula_decoder
            )

        self.stage = 1
        self.copula_loss = None
        self.marginal_logdet = None
        self.current_normalizer = None

    def set_stage(self, stage: int):
        self.stage = stage

    def initialize_stage2(self):
        self.set_stage(2)
        self.skip_copula = False
        if self.copula_encoder_args:
            self.copula_encoder = Encoder(**self.copula_encoder_args)
            self.decoder.attentional_copula_args["input_dim"] = self.copula_encoder.embedding_dim
        elif self.copula_temporal_encoder_args:
            self.copula_encoder = TemporalEncoder(**self.copula_temporal_encoder_args)
            self.decoder.attentional_copula_args["input_dim"] = self.copula_encoder.embedding_dim
        self.copula_encoder_embedding_dim = self.copula_encoder.embedding_dim
        copula_dim = self.copula_encoder_embedding_dim
        self.copula_series_encoder = nn.Embedding(
            num_embeddings=self.num_series,
            embedding_dim=self.copula_series_embedding_dim,
        )
        if self.positional_encoding:
            self.copula_time_encoding = PositionalEncoding(copula_dim, **self.positional_encoding)

        copula_elayers = nn.ModuleList([])
        for i in range(self.copula_input_encoder_layers):
            if i == 0:
                copula_elayers.append(
                    nn.Linear(self.copula_series_embedding_dim + 2, copula_dim)
                )  # +1 for the value, +1 for the mask, and the per series embedding
            else:
                copula_elayers.append(nn.Linear(copula_dim, copula_dim))
            copula_elayers.append(nn.ReLU())
        self.copula_input_encoder = deepcopy(nn.Sequential(*copula_elayers))

        if self.copula_decoder_args:
            self.decoder.create_attentional_copula()

    def set_experiment_mode(self, experiment_mode: str):
        assert experiment_mode in ["forecasting", "interpolation"]
        self.experiment_mode = experiment_mode

    @staticmethod
    def _apply_bagging(
        bagging_size,
        hist_time: torch.Tensor,
        hist_value: torch.Tensor,
        pred_time: torch.Tensor,
        pred_value: torch.Tensor,
        flow_series_emb=None,
        copula_series_emb=None,
    ):
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
        flow_series_emb: Tensor [batch, series, flow embedding size]
            An embedding for each series for the marginals, expanded over the batches.
        copula_series_emb: Tensor [batch, series, copula embedding size]
            An embedding for each series for the copula, expanded over the batches.

        Returns:
        --------
        hist_time: Tensor [batch, series, time steps]
        hist_value: Tensor [batch, series, time steps]
        pred_time: Tensor [batch, series, time steps]
        pred_value: Tensor [batch, series, time steps]
        flow_series_emb: Tensor [batch, series, flow embedding size]
        copula_series_emb: Tensor [batch, series, copula embedding size]
        """
        num_batches = hist_time.shape[0]
        num_series = hist_time.shape[1]

        # Make sure to have the exact same bag for all series
        bags = [torch.randperm(num_series)[0:bagging_size] for _ in range(num_batches)]

        hist_time = torch.stack([hist_time[i, bags[i], :] for i in range(num_batches)], dim=0)
        hist_value = torch.stack([hist_value[i, bags[i], :] for i in range(num_batches)], dim=0)
        pred_time = torch.stack([pred_time[i, bags[i], :] for i in range(num_batches)], dim=0)
        pred_value = torch.stack([pred_value[i, bags[i], :] for i in range(num_batches)], dim=0)

        flow_series_emb = torch.stack([flow_series_emb[i, bags[i], :] for i in range(num_batches)], dim=0)
        if type(copula_series_emb) != type(None):
            copula_series_emb = torch.stack([copula_series_emb[i, bags[i], :] for i in range(num_batches)], dim=0)

        return (
            hist_time,
            hist_value,
            pred_time,
            pred_value,
            flow_series_emb,
            copula_series_emb,
        )

    @staticmethod
    def _apply_subsetting(
        series_to_keep,
        hist_time: torch.Tensor,
        hist_value: torch.Tensor,
        pred_time: torch.Tensor,
        pred_value=None,
        permute_series=False,
        flow_series_emb: torch.Tensor = None,
        copula_series_emb: torch.Tensor = None,
    ):
        """
        Only keep a small number of series for each of the input tensors.
        Which series will be kept is randomly selected for each batch. The order is not preserved.

        Parameters:
        -----------
        series_to_keep: int
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

        if permute_series:
            bags = [
                torch.Tensor(series_to_keep)[torch.randperm(len(series_to_keep))].long() for _ in range(num_batches)
            ]
        else:
            bags = [torch.Tensor(series_to_keep).long() for _ in range(num_batches)]

        hist_time = torch.stack([hist_time[i, bags[i], :] for i in range(num_batches)], dim=0)
        hist_value = torch.stack([hist_value[i, bags[i], :] for i in range(num_batches)], dim=0)
        pred_time = torch.stack([pred_time[i, bags[i], :] for i in range(num_batches)], dim=0)

        flow_series_emb = torch.stack([flow_series_emb[i, bags[i], :] for i in range(num_batches)], dim=0)
        if type(copula_series_emb) != type(None):
            copula_series_emb = torch.stack([copula_series_emb[i, bags[i], :] for i in range(num_batches)], dim=0)

        if type(pred_value) != type(None):
            pred_value = torch.stack([pred_value[i, bags[i], :] for i in range(num_batches)], dim=0)
            return (
                hist_time,
                hist_value,
                pred_time,
                pred_value,
                flow_series_emb,
                copula_series_emb,
            )
        else:
            return hist_time, hist_value, pred_time, flow_series_emb, copula_series_emb

    def loss(
        self,
        hist_time: torch.Tensor,
        hist_value: torch.Tensor,
        pred_time: torch.Tensor,
        pred_value: torch.Tensor,
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
        loss: torch.Tensor []
            The loss function of TACTiS, with lower values being better. Averaged over batches.
        """
        num_batches = hist_value.shape[0]
        num_series = hist_value.shape[1]
        num_hist_timesteps = hist_value.shape[2]
        num_pred_timesteps = pred_value.shape[2]
        device = hist_value.device

        copula_series_emb = None

        # Gets the embedding for each series [batch, series, embedding size]
        # Expand over batches to be compatible with the bagging procedure, which select different series for each batch
        flow_series_emb = self.flow_series_encoder(torch.arange(num_series, device=device))
        if not self.skip_copula:
            copula_series_emb = self.copula_series_encoder(torch.arange(num_series, device=device))
        flow_series_emb = flow_series_emb[None, :, :].expand(num_batches, -1, -1)
        if not self.skip_copula:
            copula_series_emb = copula_series_emb[None, :, :].expand(num_batches, -1, -1)

        # Make sure that both time tensors are in the correct format
        if len(hist_time.shape) == 2:
            hist_time = hist_time[:, None, :]
        if len(pred_time.shape) == 2:
            pred_time = pred_time[:, None, :]
        if hist_time.shape[1] == 1:
            hist_time = hist_time.expand(-1, num_series, -1)  # the shape will be [batch, series, num_hist_timesteps]
        if pred_time.shape[1] == 1:
            pred_time = pred_time.expand(-1, num_series, -1)  # the shape will be [batch, series, num_pred_timesteps]

        # If the setup is interpolation, fix the length on each side of the interpolated window
        if self.experiment_mode == "interpolation":
            total_observed_timesteps_on_each_side = num_hist_timesteps / 2
            if int(total_observed_timesteps_on_each_side) != total_observed_timesteps_on_each_side:
                raise Exception("Odd history lengths haven't been handled for now")
            total_observed_timesteps_on_each_side = int(total_observed_timesteps_on_each_side)

        # Perform bagging
        if self.bagging_size:
            (
                hist_time,
                hist_value,
                pred_time,
                pred_value,
                flow_series_emb,
                copula_series_emb,
            ) = self._apply_bagging(
                self.bagging_size,
                hist_time,
                hist_value,
                pred_time,
                pred_value,
                flow_series_emb=flow_series_emb,
                copula_series_emb=copula_series_emb,
            )
            num_series = self.bagging_size

        # The normalizer uses the same parameters for both historical and prediction values
        if self.experiment_mode == "forecasting":
            # The normalizer uses the history's stats
            normalizer = self.data_normalization(hist_value)
            hist_value = normalizer.normalize(hist_value)
            pred_value = normalizer.normalize(pred_value)
            true_value = torch.cat(
                [
                    hist_value,
                    pred_value,
                ],
                dim=2,
            )
            # The mask is at the end
            mask = torch.cat(
                [
                    torch.ones(
                        num_batches,
                        num_series,
                        num_hist_timesteps,
                        dtype=bool,
                        device=device,
                    ),
                    torch.zeros(
                        num_batches,
                        num_series,
                        num_pred_timesteps,
                        dtype=bool,
                        device=device,
                    ),
                ],
                dim=2,
            )
        elif self.experiment_mode == "interpolation":
            # The normalizer uses the stats of the available windows
            all_values = torch.cat(
                [
                    hist_value,
                    pred_value,
                ],
                dim=2,
            )
            observed_values = torch.cat(
                [
                    all_values[:, :, :total_observed_timesteps_on_each_side],
                    all_values[:, :, -total_observed_timesteps_on_each_side:],
                ],
                dim=2,
            )
            missing_values = all_values[
                :,
                :,
                total_observed_timesteps_on_each_side:-total_observed_timesteps_on_each_side,
            ]
            normalizer = self.data_normalization(observed_values)
            observed_values = normalizer.normalize(observed_values)
            missing_values = normalizer.normalize(missing_values)
            # true_value is used only in the decoder along with the mask; so creating it for interpolation is useful
            true_value = torch.cat(
                [
                    observed_values[:, :, :total_observed_timesteps_on_each_side],
                    missing_values,
                    observed_values[:, :, -total_observed_timesteps_on_each_side:],
                ],
                dim=2,
            )
            # The mask has to be such that the window of interpolation comes in the middle of the entire window
            mask = torch.cat(
                [
                    torch.ones(
                        num_batches,
                        num_series,
                        total_observed_timesteps_on_each_side,
                        dtype=bool,
                        device=device,
                    ),
                    torch.zeros(
                        num_batches,
                        num_series,
                        num_pred_timesteps,
                        dtype=bool,
                        device=device,
                    ),
                    torch.ones(
                        num_batches,
                        num_series,
                        total_observed_timesteps_on_each_side,
                        dtype=bool,
                        device=device,
                    ),
                ],
                dim=2,
            )
        self.current_normalizer = normalizer

        if self.experiment_mode == "forecasting":
            # Forecasting
            hist_encoded_flow = torch.cat(
                [
                    hist_value[:, :, :, None],
                    flow_series_emb[:, :, None, :].expand(num_batches, -1, num_hist_timesteps, -1),
                    torch.ones(num_batches, num_series, num_hist_timesteps, 1, device=device),
                ],
                dim=3,
            )  # Shape: [batch, num_series, num_hist_timesteps, embedding_size+2]
            if not self.skip_copula:
                hist_encoded_copula = torch.cat(
                    [
                        hist_value[:, :, :, None],
                        copula_series_emb[:, :, None, :].expand(num_batches, -1, num_hist_timesteps, -1),
                        torch.ones(
                            num_batches,
                            num_series,
                            num_hist_timesteps,
                            1,
                            device=device,
                        ),
                    ],
                    dim=3,
                )  # Shape: [batch, num_series, num_hist_timesteps, embedding_size+2]
            # For the prediction embedding, replace the values by zeros, since they won't be available during sampling
            pred_encoded_flow = torch.cat(
                [
                    torch.zeros(num_batches, num_series, num_pred_timesteps, 1, device=device),
                    flow_series_emb[:, :, None, :].expand(num_batches, -1, num_pred_timesteps, -1),
                    torch.zeros(num_batches, num_series, num_pred_timesteps, 1, device=device),
                ],
                dim=3,
            )  # Shape: [batch, num_series, num_hist_timesteps, embedding_size+2]. Note that we are also leaving out the values as zeros
            if not self.skip_copula:
                pred_encoded_copula = torch.cat(
                    [
                        torch.zeros(
                            num_batches,
                            num_series,
                            num_pred_timesteps,
                            1,
                            device=device,
                        ),
                        copula_series_emb[:, :, None, :].expand(num_batches, -1, num_pred_timesteps, -1),
                        torch.zeros(
                            num_batches,
                            num_series,
                            num_pred_timesteps,
                            1,
                            device=device,
                        ),
                    ],
                    dim=3,
                )  # Shape: [batch, num_series, num_hist_timesteps, embedding_size+2]. Note that we are also leaving out the values as zeros

            flow_encoded = torch.cat(
                [hist_encoded_flow, pred_encoded_flow], dim=2
            )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, embedding_size+2]
            if not self.skip_copula:
                copula_encoded = torch.cat(
                    [hist_encoded_copula, pred_encoded_copula], dim=2
                )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, embedding_size+2]
            flow_encoded = self.flow_input_encoder(
                flow_encoded
            )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]
            if not self.skip_copula:
                copula_encoded = self.copula_input_encoder(
                    copula_encoded
                )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]
            if self.input_encoding_normalization:
                flow_encoded = flow_encoded * self.flow_encoder_embedding_dim**0.5
                if not self.skip_copula:
                    copula_encoded = copula_encoded * self.copula_encoder_embedding_dim**0.5

            # Add the time encoding here after the input encoding to be compatible with how positional encoding is used.
            # Adjustments may be required for other ways to encode time.
            timesteps = torch.cat([hist_time, pred_time], dim=2)
            flow_encoded = self.flow_time_encoding(flow_encoded, timesteps.to(int))
            if not self.skip_copula:
                copula_encoded = self.copula_time_encoding(copula_encoded, timesteps.to(int))

            flow_encoded = self.flow_encoder.forward(
                flow_encoded
            )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]
            if not self.skip_copula:
                copula_encoded = self.copula_encoder.forward(
                    copula_encoded
                )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]
        else:
            # Interpolation
            hist_encoded_flow = torch.cat(
                [
                    observed_values[:, :, :, None],
                    flow_series_emb[:, :, None, :].expand(num_batches, -1, num_hist_timesteps, -1),
                    torch.ones(num_batches, num_series, num_hist_timesteps, 1, device=device),
                ],
                dim=3,
            )  # Shape: [batch, num_series, num_hist_timesteps, embedding_size+2]
            if not self.skip_copula:
                hist_encoded_copula = torch.cat(
                    [
                        observed_values[:, :, :, None],
                        copula_series_emb[:, :, None, :].expand(num_batches, -1, num_hist_timesteps, -1),
                        torch.ones(
                            num_batches,
                            num_series,
                            num_hist_timesteps,
                            1,
                            device=device,
                        ),
                    ],
                    dim=3,
                )  # Shape: [batch, num_series, num_hist_timesteps, embedding_size+2]
            # For the prediction embedding, replace the values by zeros, since they won't be available during sampling
            pred_encoded_flow = torch.cat(
                [
                    torch.zeros(num_batches, num_series, num_pred_timesteps, 1, device=device),
                    flow_series_emb[:, :, None, :].expand(num_batches, -1, num_pred_timesteps, -1),
                    torch.zeros(num_batches, num_series, num_pred_timesteps, 1, device=device),
                ],
                dim=3,
            )  # Shape: [batch, num_series, num_hist_timesteps, embedding_size+2]. Note that we are also leaving out the values as zeros
            if not self.skip_copula:
                pred_encoded_copula = torch.cat(
                    [
                        torch.zeros(
                            num_batches,
                            num_series,
                            num_pred_timesteps,
                            1,
                            device=device,
                        ),
                        copula_series_emb[:, :, None, :].expand(num_batches, -1, num_pred_timesteps, -1),
                        torch.zeros(
                            num_batches,
                            num_series,
                            num_pred_timesteps,
                            1,
                            device=device,
                        ),
                    ],
                    dim=3,
                )  # Shape: [batch, num_series, num_hist_timesteps, embedding_size+2]. Note that we are also leaving out the values as zeros

            flow_encoded = torch.cat(
                [hist_encoded_flow, pred_encoded_flow], dim=2
            )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, embedding_size+2]
            if not self.skip_copula:
                copula_encoded = torch.cat(
                    [hist_encoded_copula, pred_encoded_copula], dim=2
                )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, embedding_size+2]

            # Important step for interpolation - reorder the encoded vectors to reflect the order of the mask
            # This is because everything else (mask, true_value) is in the correct timesteps order
            flow_encoded = torch.cat(
                [
                    flow_encoded[:, :, :total_observed_timesteps_on_each_side],
                    flow_encoded[:, :, -num_pred_timesteps:],
                    flow_encoded[
                        :,
                        :,
                        total_observed_timesteps_on_each_side : total_observed_timesteps_on_each_side
                        + total_observed_timesteps_on_each_side,
                    ],
                ],
                dim=2,
            )
            if not self.skip_copula:
                copula_encoded = torch.cat(
                    [
                        copula_encoded[:, :, :total_observed_timesteps_on_each_side],
                        copula_encoded[:, :, -num_pred_timesteps:],
                        copula_encoded[
                            :,
                            :,
                            total_observed_timesteps_on_each_side : total_observed_timesteps_on_each_side
                            + total_observed_timesteps_on_each_side,
                        ],
                    ],
                    dim=2,
                )

            flow_encoded = self.flow_input_encoder(
                flow_encoded
            )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]
            if not self.skip_copula:
                copula_encoded = self.copula_input_encoder(
                    copula_encoded
                )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]
            if self.input_encoding_normalization:
                flow_encoded = flow_encoded * self.flow_encoder_embedding_dim**0.5
                if not self.skip_copula:
                    copula_encoded = copula_encoded * self.copula_encoder_embedding_dim**0.5
            # Add the time encoding here after the input encoding to be compatible with how positional encoding is used.
            # Adjustments may be required for other ways to encode time.
            timesteps = torch.cat([hist_time, pred_time], dim=2)
            flow_encoded = self.flow_time_encoding(flow_encoded, timesteps.to(int))
            if not self.skip_copula:
                copula_encoded = self.copula_time_encoding(copula_encoded, timesteps.to(int))

            flow_encoded = self.flow_encoder.forward(
                flow_encoded
            )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]
            if not self.skip_copula:
                copula_encoded = self.copula_encoder.forward(
                    copula_encoded
                )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]

        if self.skip_copula:
            copula_encoded = None

        _ = self.decoder.loss(
            flow_encoded=flow_encoded,
            copula_encoded=copula_encoded,
            mask=mask,
            true_value=true_value,
        )  # previously returned loss here when loss coefficients weren't used

        self.copula_loss = self.decoder.copula_loss
        self.marginal_logdet = self.decoder.marginal_logdet
        self.unnormalized_copula_loss = torch.clone(self.decoder.copula_loss)
        self.unnormalized_marginal_logdet = torch.clone(self.decoder.marginal_logdet)

        if self.loss_normalization in {"series", "both"}:
            self.copula_loss = self.copula_loss / num_series
            self.marginal_logdet = self.marginal_logdet / num_series
        if self.loss_normalization in {"timesteps", "both"}:
            self.copula_loss = self.copula_loss / num_pred_timesteps
            self.marginal_logdet = self.marginal_logdet / num_pred_timesteps

        return self.marginal_logdet, self.copula_loss

    def sample(
        self,
        num_samples: int,
        hist_time: torch.Tensor,
        hist_value: torch.Tensor,
        pred_time: torch.Tensor,
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
        num_batches = hist_value.shape[0]
        num_series = hist_value.shape[1]
        num_hist_timesteps = hist_value.shape[2]
        num_pred_timesteps = pred_time.shape[-1]
        device = hist_value.device

        # Gets the embedding for each series [batch, series, embedding size]
        # Expand over batches to be compatible with the bagging procedure, which select different series for each batch
        copula_series_emb = None

        flow_series_emb = self.flow_series_encoder(torch.arange(num_series, device=device))
        if not self.skip_copula:
            copula_series_emb = self.copula_series_encoder(torch.arange(num_series, device=device))
        flow_series_emb = flow_series_emb[None, :, :].expand(num_batches, -1, -1)
        if not self.skip_copula:
            copula_series_emb = copula_series_emb[None, :, :].expand(num_batches, -1, -1)

        # Make sure that both time tensors are in the correct format
        if len(hist_time.shape) == 2:
            hist_time = hist_time[:, None, :]
        if len(pred_time.shape) == 2:
            pred_time = pred_time[:, None, :]
        if hist_time.shape[1] == 1:
            hist_time = hist_time.expand(-1, num_series, -1)
        if pred_time.shape[1] == 1:
            pred_time = pred_time.expand(-1, num_series, -1)

        # If the setup is interpolation, fix the length on each side of the interpolated window
        if self.experiment_mode == "interpolation":
            actual_history_length = num_hist_timesteps - num_pred_timesteps
            total_observed_timesteps_on_each_side = actual_history_length / 2
            if int(total_observed_timesteps_on_each_side) != total_observed_timesteps_on_each_side:
                raise Exception("Odd history lengths haven't been handled for now")
            total_observed_timesteps_on_each_side = int(total_observed_timesteps_on_each_side)

        # If we are performing interpolation, we need to modify hist_time and pred_time to reflect the mask
        # We assume pred_time is invalid here since hist_time contains the entire window
        # But hist_time and pred_time should not reflect the mask since by the time the positional encoding is added, the order itself is correct
        # So we only use the `prediction_length` or `num_pred_timesteps` from pred_time. We throw away pred_time itself
        # Also remove the prediction window from `hist_value`
        if self.experiment_mode == "interpolation":
            # actual_hist_time = torch.cat(
            #     [
            #         hist_time[:, :, :total_observed_timesteps_on_each_side],
            #         hist_time[:, :, -total_observed_timesteps_on_each_side:],
            #     ],
            #     dim=2,
            # )
            # actual_pred_time = hist_time[:, :, total_observed_timesteps_on_each_side:total_observed_timesteps_on_each_side+num_pred_timesteps]
            hist_time = hist_time[:, :, :-num_pred_timesteps]
            pred_time = hist_time[:, :, -num_pred_timesteps:]
            hist_value = torch.cat(
                [
                    hist_value[:, :, :total_observed_timesteps_on_each_side],
                    hist_value[:, :, -total_observed_timesteps_on_each_side:],
                ],
                dim=2,
            )
            observed_values = hist_value

        # The normalizer remembers its parameter to reverse it with the samples
        if self.experiment_mode == "forecasting":
            normalizer = self.data_normalization(hist_value)
            hist_value = normalizer.normalize(hist_value)
            # Make the mask and true_value
            mask = torch.cat(
                [
                    torch.ones(
                        num_batches,
                        num_series,
                        num_hist_timesteps,
                        dtype=bool,
                        device=device,
                    ),
                    torch.zeros(
                        num_batches,
                        num_series,
                        num_pred_timesteps,
                        dtype=bool,
                        device=device,
                    ),
                ],
                dim=2,
            )
            true_value = torch.cat(
                [
                    hist_value,
                    torch.zeros(num_batches, num_series, num_pred_timesteps, device=device),
                ],
                dim=2,
            )
        else:
            normalizer = self.data_normalization(observed_values)
            observed_values = normalizer.normalize(observed_values)
            # Make the mask and true_value
            mask = torch.cat(
                [
                    torch.ones(
                        num_batches,
                        num_series,
                        total_observed_timesteps_on_each_side,
                        dtype=bool,
                        device=device,
                    ),
                    torch.zeros(
                        num_batches,
                        num_series,
                        num_pred_timesteps,
                        dtype=bool,
                        device=device,
                    ),
                    torch.ones(
                        num_batches,
                        num_series,
                        total_observed_timesteps_on_each_side,
                        dtype=bool,
                        device=device,
                    ),
                ],
                dim=2,
            )
            # true_value is used only in the decoder along with the mask; so creating it for interpolation is useful
            true_value = torch.cat(
                [
                    observed_values[:, :, :total_observed_timesteps_on_each_side],
                    torch.zeros(num_batches, num_series, num_pred_timesteps, device=device),
                    observed_values[:, :, -total_observed_timesteps_on_each_side:],
                ],
                dim=2,
            )

        if self.experiment_mode == "forecasting":
            hist_encoded_flow = torch.cat(
                [
                    hist_value[:, :, :, None],
                    flow_series_emb[:, :, None, :].expand(num_batches, -1, num_hist_timesteps, -1),
                    torch.ones(num_batches, num_series, num_hist_timesteps, 1, device=device),
                ],
                dim=3,
            )  # Shape: [batch, num_series, num_hist_timesteps, embedding_size+2]
            if not self.skip_copula:
                hist_encoded_copula = torch.cat(
                    [
                        hist_value[:, :, :, None],
                        copula_series_emb[:, :, None, :].expand(num_batches, -1, num_hist_timesteps, -1),
                        torch.ones(
                            num_batches,
                            num_series,
                            num_hist_timesteps,
                            1,
                            device=device,
                        ),
                    ],
                    dim=3,
                )  # Shape: [batch, num_series, num_hist_timesteps, embedding_size+2]
            # For the prediction embedding, replace the values by zeros, since they won't be available during sampling
            pred_encoded_flow = torch.cat(
                [
                    torch.zeros(num_batches, num_series, num_pred_timesteps, 1, device=device),
                    flow_series_emb[:, :, None, :].expand(num_batches, -1, num_pred_timesteps, -1),
                    torch.zeros(num_batches, num_series, num_pred_timesteps, 1, device=device),
                ],
                dim=3,
            )  # Shape: [batch, num_series, num_hist_timesteps, embedding_size+2]. Note that we are also leaving out the values as zeros
            if not self.skip_copula:
                pred_encoded_copula = torch.cat(
                    [
                        torch.zeros(
                            num_batches,
                            num_series,
                            num_pred_timesteps,
                            1,
                            device=device,
                        ),
                        copula_series_emb[:, :, None, :].expand(num_batches, -1, num_pred_timesteps, -1),
                        torch.zeros(
                            num_batches,
                            num_series,
                            num_pred_timesteps,
                            1,
                            device=device,
                        ),
                    ],
                    dim=3,
                )  # Shape: [batch, num_series, num_hist_timesteps, embedding_size+2]. Note that we are also leaving out the values as zeros

            flow_encoded = torch.cat(
                [hist_encoded_flow, pred_encoded_flow], dim=2
            )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, embedding_size+2]
            if not self.skip_copula:
                copula_encoded = torch.cat(
                    [hist_encoded_copula, pred_encoded_copula], dim=2
                )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, embedding_size+2]
            flow_encoded = self.flow_input_encoder(
                flow_encoded
            )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]
            if not self.skip_copula:
                copula_encoded = self.copula_input_encoder(
                    copula_encoded
                )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]
            if self.input_encoding_normalization:
                flow_encoded = flow_encoded * self.flow_encoder_embedding_dim**0.5
                if not self.skip_copula:
                    copula_encoded = copula_encoded * self.copula_encoder_embedding_dim**0.5

            # Add the time encoding here after the input encoding to be compatible with how positional encoding is used.
            # Adjustments may be required for other ways to encode time.
            timesteps = torch.cat([hist_time, pred_time], dim=2)
            flow_encoded = self.flow_time_encoding(flow_encoded, timesteps.to(int))
            if not self.skip_copula:
                copula_encoded = self.copula_time_encoding(copula_encoded, timesteps.to(int))

            flow_encoded = self.flow_encoder.forward(
                flow_encoded
            )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]
            if not self.skip_copula:
                copula_encoded = self.copula_encoder.forward(
                    copula_encoded
                )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]
        else:
            # Interpolation
            hist_encoded_flow = torch.cat(
                [
                    observed_values[:, :, :, None],
                    flow_series_emb[:, :, None, :].expand(num_batches, -1, num_hist_timesteps - num_pred_timesteps, -1),
                    torch.ones(
                        num_batches,
                        num_series,
                        num_hist_timesteps - num_pred_timesteps,
                        1,
                        device=device,
                    ),
                ],
                dim=3,
            )  # Shape: [batch, num_series, num_hist_timesteps, embedding_size+2]
            if not self.skip_copula:
                hist_encoded_copula = torch.cat(
                    [
                        observed_values[:, :, :, None],
                        copula_series_emb[:, :, None, :].expand(
                            num_batches, -1, num_hist_timesteps - num_pred_timesteps, -1
                        ),
                        torch.ones(
                            num_batches,
                            num_series,
                            num_hist_timesteps - num_pred_timesteps,
                            1,
                            device=device,
                        ),
                    ],
                    dim=3,
                )  # Shape: [batch, num_series, num_hist_timesteps, embedding_size+2]
            # For the prediction embedding, replace the values by zeros, since they won't be available during sampling
            pred_encoded_flow = torch.cat(
                [
                    torch.zeros(num_batches, num_series, num_pred_timesteps, 1, device=device),
                    flow_series_emb[:, :, None, :].expand(num_batches, -1, num_pred_timesteps, -1),
                    torch.zeros(num_batches, num_series, num_pred_timesteps, 1, device=device),
                ],
                dim=3,
            )  # Shape: [batch, num_series, num_hist_timesteps, embedding_size+2]. Note that we are also leaving out the values as zeros
            if not self.skip_copula:
                pred_encoded_copula = torch.cat(
                    [
                        torch.zeros(
                            num_batches,
                            num_series,
                            num_pred_timesteps,
                            1,
                            device=device,
                        ),
                        copula_series_emb[:, :, None, :].expand(num_batches, -1, num_pred_timesteps, -1),
                        torch.zeros(
                            num_batches,
                            num_series,
                            num_pred_timesteps,
                            1,
                            device=device,
                        ),
                    ],
                    dim=3,
                )  # Shape: [batch, num_series, num_hist_timesteps, embedding_size+2]. Note that we are also leaving out the values as zeros

            flow_encoded = torch.cat(
                [hist_encoded_flow, pred_encoded_flow], dim=2
            )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, embedding_size+2]
            if not self.skip_copula:
                copula_encoded = torch.cat(
                    [hist_encoded_copula, pred_encoded_copula], dim=2
                )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, embedding_size+2]

            # Important step for interpolation - reorder the encoded vectors to reflect the order of the mask
            flow_encoded = torch.cat(
                [
                    flow_encoded[:, :, :total_observed_timesteps_on_each_side],
                    flow_encoded[:, :, -num_pred_timesteps:],
                    flow_encoded[
                        :,
                        :,
                        total_observed_timesteps_on_each_side : total_observed_timesteps_on_each_side
                        + total_observed_timesteps_on_each_side,
                    ],
                ],
                dim=2,
            )
            if not self.skip_copula:
                copula_encoded = torch.cat(
                    [
                        copula_encoded[:, :, :total_observed_timesteps_on_each_side],
                        copula_encoded[:, :, -num_pred_timesteps:],
                        copula_encoded[
                            :,
                            :,
                            total_observed_timesteps_on_each_side : total_observed_timesteps_on_each_side
                            + total_observed_timesteps_on_each_side,
                        ],
                    ],
                    dim=2,
                )

            flow_encoded = self.flow_input_encoder(
                flow_encoded
            )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]
            if not self.skip_copula:
                copula_encoded = self.copula_input_encoder(
                    copula_encoded
                )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]
            if self.input_encoding_normalization:
                flow_encoded = flow_encoded * self.flow_encoder_embedding_dim**0.5
                if not self.skip_copula:
                    copula_encoded = copula_encoded * self.copula_encoder_embedding_dim**0.5
            # Add the time encoding here after the input encoding to be compatible with how positional encoding is used.
            # Adjustments may be required for other ways to encode time.
            timesteps = torch.cat([hist_time, pred_time], dim=2)
            flow_encoded = self.flow_time_encoding(flow_encoded, timesteps.to(int))
            if not self.skip_copula:
                copula_encoded = self.copula_time_encoding(copula_encoded, timesteps.to(int))

            flow_encoded = self.flow_encoder.forward(
                flow_encoded
            )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]
            if not self.skip_copula:
                copula_encoded = self.copula_encoder.forward(
                    copula_encoded
                )  # Shape: [batch, num_series, num_hist_timesteps+num_pred_timesteps, encoder_size]

        if self.skip_copula:
            copula_encoded = None
        decoder_output = self.decoder.sample(
            num_samples,
            mask=mask,
            true_value=true_value,
            flow_encoded=flow_encoded,
            copula_encoded=copula_encoded,
        )

        samples = decoder_output
        samples = normalizer.denormalize(samples)
        return samples
