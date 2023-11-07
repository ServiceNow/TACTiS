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


import torch
import time
from torch import nn


class Encoder(nn.Module):
    """
    The traditional encoder for TACTiS, based on the Transformer architecture.

    The encoder receives an input which contains for each series and time step:
    * The series value at the time step, masked to zero if part of the values to be forecasted
    * The mask
    * The embedding for the series
    * The embedding for the time step
    And has already been through any input encoder.

    The decoder returns an output containing an embedding for each series and time step.
    """

    def __init__(
        self,
        attention_layers: int,
        attention_heads: int,
        attention_dim: int,
        attention_feedforward_dim: int,
        dropout: float = 0.1,
    ):
        """
        Parameters:
        -----------
        attention_layers: int
            How many successive attention layers this encoder will use.
        attention_heads: int
            How many independant heads the attention layer will have.
        attention_dim: int
            The size of the attention layer input and output, for each head.
        attention_feedforward_dim: int
            The dimension of the hidden layer in the feed forward step.
        dropout: float, default to 0.1
            Dropout parameter for the attention.
        """
        super().__init__()

        self.attention_layers = attention_layers
        self.attention_heads = attention_heads
        self.attention_dim = attention_dim
        self.attention_feedforward_dim = attention_feedforward_dim
        self.dropout = dropout

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.attention_dim * self.attention_heads,
                self.attention_heads,
                self.attention_feedforward_dim,
                self.dropout,
            ),
            self.attention_layers,
        )
        self.total_attention_time = 0.0

    @property
    def embedding_dim(self) -> int:
        """
        Returns:
        --------
        dim: int
            The expected dimensionality of the input embedding, and the dimensionality of the output embedding
        """
        return self.attention_dim * self.attention_heads

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Compute the embedding for each series and time step.

        Parameters:
        -----------
        encoded: Tensor [batch, series, time steps, input embedding dimension]
            A tensor containing an embedding for each series and time step.
            This embedding is expected to only contain local information, with no interaction between series or time steps.

        Returns:
        --------
        output: torch.Tensor [batch, series, time steps, output embedding dimension]
            The transformed embedding for each series and time step.
        """
        num_batches = encoded.shape[0]
        num_series = encoded.shape[1]
        num_timesteps = encoded.shape[2]

        # Merge the series and time steps, since the PyTorch attention implementation only accept three-dimensional input,
        # and the attention is applied between all tokens, no matter their series or time step.
        encoded = encoded.view(num_batches, num_series * num_timesteps, self.embedding_dim)

        attention_start_time = time.time()

        # The PyTorch implementation wants the following order: [tokens, batch, embedding]
        encoded = encoded.transpose(0, 1)
        # Arjun: I have changed this to the usual full attention
        output = self.transformer_encoder(encoded)
        # # This was the previous code
        # output = self.transformer_encoder(
        #     encoded, mask=torch.zeros(encoded.shape[0], encoded.shape[0], device=encoded.device)
        # )
        # Reset to the original shape
        output = output.transpose(0, 1)
        attention_end_time = time.time()
        self.total_attention_time = attention_end_time - attention_start_time

        # Resize back to original shape
        output = output.view(num_batches, num_series, num_timesteps, self.embedding_dim)

        return output


class TemporalEncoder(nn.Module):
    """
    The encoder for TACTiS, based on the Temporal Transformer architecture.
    This encoder alternate between doing self-attention between different series of the same time steps,
    and doing self-attention between different time steps of the same series.
    This greatly reduces the memory footprint compared to TACTiSEncoder.

    The encoder receives an input which contains for each variable and time step:
    * The series value at the time step, masked to zero if part of the values to be forecasted
    * The mask
    * The embedding for the series
    * The embedding for the time step
    And has already been through any input encoder.

    The decoder returns an output containing an embedding for each series and time step.
    """

    def __init__(
        self,
        attention_layers: int,
        attention_heads: int,
        attention_dim: int,
        attention_feedforward_dim: int,
        dropout: float = 0.1,
    ):
        """
        Parameters:
        -----------
        attention_layers: int
            How many successive attention pairs of layers this will use.
            Note that the total number of layers is going to be the double of this number.
            Each pair will consist of a layer with attention done over time steps,
            followed by a layer with attention done over series.
        attention_heads: int
            How many independant heads the attention layer will have.
        attention_dim: int
            The size of the attention layer input and output, for each head.
        attention_feedforward_dim: int
            The dimension of the hidden layer in the feed forward step.
        dropout: float, default to 0.1
            Dropout parameter for the attention.
        """
        super().__init__()

        self.attention_layers = attention_layers
        self.attention_heads = attention_heads
        self.attention_dim = attention_dim
        self.attention_feedforward_dim = attention_feedforward_dim
        self.dropout = dropout
        self.total_attention_time = 0.0

        self.layer_timesteps = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    self.attention_dim * self.attention_heads,
                    self.attention_heads,
                    self.attention_feedforward_dim,
                    self.dropout,
                )
                for _ in range(self.attention_layers)
            ]
        )

        self.layer_series = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    self.attention_dim * self.attention_heads,
                    self.attention_heads,
                    self.attention_feedforward_dim,
                    self.dropout,
                )
                for _ in range(self.attention_layers)
            ]
        )

    @property
    def embedding_dim(self) -> int:
        """
        Returns:
        --------
        dim: int
            The expected dimensionality of the input embedding, and the dimensionality of the output embedding
        """
        return self.attention_dim * self.attention_heads

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Compute the embedding for each series and time step.

        Parameters:
        -----------
        encoded: Tensor [batch, series, time steps, input embedding dimension]
            A tensor containing an embedding for each series and time step.
            This embedding is expected to only contain local information, with no interaction between series or time steps.

        Returns:
        --------
        output: torch.Tensor [batch, series, time steps, output embedding dimension]
            The transformed embedding for each series and time step.
        """
        num_batches = encoded.shape[0]
        num_series = encoded.shape[1]
        num_timesteps = encoded.shape[2]

        data = encoded

        attention_start_time = time.time()
        for i in range(self.attention_layers):
            # Treat the various series as a batch dimension
            mod_timesteps = self.layer_timesteps[i]
            # [batch * series, time steps, embedding]
            data = data.flatten(start_dim=0, end_dim=1)
            # [time steps, batch * series, embedding] Correct order for PyTorch module
            data = data.transpose(0, 1)
            # Perform attention
            data = mod_timesteps(data)
            # [batch * series, time steps, embedding]
            data = data.transpose(0, 1)
            # [batch, series, time steps, embedding]
            data = data.unflatten(dim=0, sizes=(num_batches, num_series))

            # Treat the various time steps as a batch dimension
            mod_series = self.layer_series[i]
            # Transpose to [batch, timesteps, series, embedding]
            data = data.transpose(1, 2)
            # [batch * time steps, series, embedding] Correct order for PyTorch module
            data = data.flatten(start_dim=0, end_dim=1)
            # [series, batch * time steps, embedding]
            data = data.transpose(0, 1)
            # Perform attention
            data = mod_series(data)
            # [batch * time steps, series, embedding]
            data = data.transpose(0, 1)
            # [batch, time steps, series, embedding]
            data = data.unflatten(dim=0, sizes=(num_batches, num_timesteps))
            # Transpose to [batch, series, time steps, embedding]
            data = data.transpose(1, 2)

        attention_end_time = time.time()
        self.total_attention_time = attention_end_time - attention_start_time
        # The resulting tensor may not be contiguous, which can cause problems further down the line.
        output = data.contiguous()

        return output
