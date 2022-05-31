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

>> The two possible encoders for TACTiS, based on the Transformer architecture.
"""


import torch
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

    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute the embedding for each series and time step.

        Parameters:
        -----------
        data: Tensor [batch, series, time steps, input embedding dimension]
            A tensor containing an embedding for each series and time step.
            This embedding is expected to only contain local information, with no interaction between series or time steps.

        Returns:
        --------
        encoded: torch.Tensor [batch, series, time steps, output embedding dimension]
            The encoded embedding for each series and time step.
        """
        pass


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

    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute the embedding for each series and time step.

        Parameters:
        -----------
        data: Tensor [batch, series, time steps, input embedding dimension]
            A tensor containing an embedding for each series and time step.
            This embedding is expected to only contain local information, with no interaction between series or time steps.

        Returns:
        --------
        encoded: torch.Tensor [batch, series, time steps, output embedding dimension]
            The encoded embedding for each series and time step.
        """
        pass
