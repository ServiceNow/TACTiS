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

>> The marginal distribution for the forecasts
"""


import torch
from torch import nn
from typing import Tuple


class DSFMarginal(nn.Module):
    """
    Compute the marginals using a Deep Sigmoid Flow conditioned using a MLP.
    The conditioning MLP uses the embedding from the encoder as its input.
    """
    def __init__(self):
        super().__init__()

    def forward_logdet(self, context: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the cumulative density function of a marginal conditioned using the given context, for the given value of x.
        Also returns the logarithm of the derivative of this transformation.

        Parameters:
        -----------
        context: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            The series and time steps dimensions are merged.
        x: Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            A tensor containing the value to be transformed using the CDF.
            The series and time steps dimensions are merged.
            If a third dimension is present, then the context is considered to be constant across this dimension.
        Returns:
        --------
        u: torch.Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            The CDF at the given point, a value between 0 and 1.
            The series and time steps dimensions are merged.
            The shape of the output is the same as the shape of x.
        logdet: torch.Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            The logarithm of the derivative of the transformation.
            The series and time steps dimensions are merged.
            The shape of the output is the same as the shape of x.
        """
        pass

    def forward_logdet(self, context: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the cumulative density function of a marginal conditioned using the given context, for the given value of x.

        Parameters:
        -----------
        context: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            The series and time steps dimensions are merged.
        x: Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            A tensor containing the value to be transformed using the CDF.
            The series and time steps dimensions are merged.
            If a third dimension is present, then the context is considered to be constant across this dimension.
        Returns:
        --------
        u: torch.Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            The CDF at the given point, a value between 0 and 1.
            The series and time steps dimensions are merged.
            The shape of the output is the same as the shape of x.
        """
        pass

    def inverse(self, context: torch.Tensor, u: torch.Tensor, max_iter: int = 100, precision: float = 1e-6, max_value: float = 1000.0) -> torch.Tensor:
        """
        Compute the inverse cumulative density function of a marginal conditioned using the given context, for the given value of u.
        This method uses a dichotomic search.
        The gradient of this method cannot be computed, so it should only be used for sampling.

        Parameters:
        -----------
        context: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            The series and time steps dimensions are merged.
        u: Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            A tensor containing the value to be transformed using the inverse CDF.
            The series and time steps dimensions are merged.
            If a third dimension is present, then the context is considered to be constant across this dimension.
        max_iter: int, default = 100
            The maximum number of iterations for the dichotomic search.
            The precision of the result should improve by a factor of 2 at each iteration.
        precision: float, default = 1e-6
            If the difference between CDF(x) and u is less than this value for all variables, stop the search.
        max_value: float, default = 1000.0
            The absolute upper bound on the possible output.
        Returns:
        --------
        x: torch.Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            The inverse CDF at the given value.
            The series and time steps dimensions are merged.
            The shape of the output is the same as the shape of u.
        """
        pass
