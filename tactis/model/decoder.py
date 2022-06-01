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

>> The various decoders for TACTiS, to output the forecasted distributions.
"""


import torch
from torch import nn


def _merge_series_time_dims(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a Tensor with dimensions [batch, series, time steps] to one with dimensions [batch, series * time steps]
    """
    assert x.dim() == 3
    return x.view(x.shape[0], x.shape[1] * x.shape[2])


def _split_series_time_dims(x: torch.Tensor, old_shape: torch.Size) -> torch.Tensor:
    """
    Convert a Tensor with dimensions [batch, series * time steps] to one with dimensions [batch, series, time steps]
    """
    assert x.dim() == 2
    assert len(old_shape) == 3
    return x.view(old_shape[0], old_shape[1], old_shape[2])


class CopulaDecoder(nn.Module):
    """
    A decoder which forecast using a distribution built from a copula and marginal distributions.
    """

    def __init__(self):
        super().__init__()

    def loss(self, encoded: torch.Tensor, mask: torch.BoolTensor, true_value: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss function of the decoder.

        Parameters:
        -----------
        encoded: Tensor [batch, series, time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            This embedding is coming from the encoder, so contains shared information across series and time steps.
        mask: BoolTensor [batch, series, time steps]
            A tensor containing a mask indicating whether a given value was available for the encoder.
            The decoder only forecasts values for which the mask is set to False.
        true_value: Tensor [batch, series, time steps]
            A tensor containing the true value for the values to be forecasted.
            Only the values where the mask is set to False will be considered in the loss function.

        Returns:
        --------
        loss: torch.Tensor [batch]
            The loss function, equal to the negative log likelihood of the distribution.
        """
        pass

    def sample(
        self, num_samples: int, encoded: torch.Tensor, mask: torch.BoolTensor, true_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate the given number of samples from the forecasted distribution.

        Parameters:
        -----------
        num_samples: int
            How many samples to generate, must be >= 1.
        encoded: Tensor [batch, series, time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            This embedding is coming from the encoder, so contains shared information across series and time steps.
        mask: BoolTensor [batch, series, time steps]
            A tensor containing a mask indicating whether a given value was available for the encoder.
            The decoder only forecasts values for which the mask is set to False.
        true_value: Tensor [batch, series, time steps]
            A tensor containing the true value for the values to be forecasted.
            The values where the mask is set to True will be copied as-is in the output.

        Returns:
        --------
        samples: torch.Tensor [batch, series, time steps, samples]
            Samples drawn from the forecasted distribution.
        """
        pass


class AttentionalCopula(nn.Module):
    """
    A non-parametric copula based on attention between the various variables.
    """

    def __init__(self):
        super().__init__()

    def loss(
        self,
        hist_encoded: torch.Tensor,
        hist_true_u: torch.Tensor,
        pred_encoded: torch.Tensor,
        pred_true_u: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss function of the copula portion of the decoder.

        Parameters:
        -----------
        hist_encoded: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each series and time step that does not have to be forecasted.
            The series and time steps dimensions are merged.
        hist_true_u: Tensor [batch, series * time steps]
            A tensor containing the true value for the values that do not have to be forecasted, transformed by the marginal distribution into U(0,1) values.
            The series and time steps dimensions are merged.
        pred_encoded: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step that does have to be forecasted.
            The series and time steps dimensions are merged.
        pred_true_u: Tensor [batch, series * time steps]
            A tensor containing the true value for the values to be forecasted, transformed by the marginal distribution into U(0,1) values.
            The series and time steps dimensions are merged.

        Returns:
        --------
        embedding: torch.Tensor [batch]
            The loss function, equal to the negative log likelihood of the copula.
        """
        pass

    def sample(
        self, num_samples: int, hist_encoded: torch.Tensor, hist_true_u: torch.Tensor, pred_encoded: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate the given number of samples from the forecasted copula.

        Parameters:
        -----------
        num_samples: int
            How many samples to generate, must be >= 1.
        hist_encoded: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step that does not have to be forecasted.
            The series and time steps dimensions are merged.
        hist_true_u: Tensor [batch, series * time steps]
            A tensor containing the true value for the values that do not have to be forecasted, transformed by the marginal distribution into U(0,1) values.
            The series and time steps dimensions are merged.
        pred_encoded: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step that does have to be forecasted.
            The series and time steps dimensions are merged.

        Returns:
        --------
        samples: torch.Tensor [batch, series * time steps, samples]
            Samples drawn from the forecasted copula, thus in the [0, 1] range.
            The series and time steps dimensions are merged.
        """
        pass


class TrivialCopula(nn.Module):
    """
    The trivial copula where all variables are independent.
    """

    def __init__(self):
        super().__init__()

    def loss(
        self,
        hist_encoded: torch.Tensor,
        hist_true_u: torch.Tensor,
        pred_encoded: torch.Tensor,
        pred_true_u: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss function of the copula portion of the decoder.

        Parameters:
        -----------
        hist_encoded: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step that does not have to be forecasted.
            The series and time steps dimensions are merged.
        hist_true_u: Tensor [batch, series * time steps]
            A tensor containing the true value for the values that do not have to be forecasted, transformed by the marginal distribution into U(0,1) values.
            The series and time steps dimensions are merged.
        pred_encoded: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step that does have to be forecasted.
            The series and time steps dimensions are merged.
        pred_true_u: Tensor [batch, series * time steps]
            A tensor containing the true value for the values to be forecasted, transformed by the marginal distribution into U(0,1) values.
            The series and time steps dimensions are merged.

        Returns:
        --------
        embedding: torch.Tensor [batch]
            The loss function, equal to the negative log likelihood of the copula.
            This is always equal to zero.
        """
        batch_size = hist_encoded.shape[0]
        device = hist_encoded.device
        # Trivially, the probability of all u is equal to 1 if in the unit cube (which it should always be by construction)
        return torch.zeros(batch_size, device=device)

    def sample(
        self, num_samples: int, hist_encoded: torch.Tensor, hist_true_u: torch.Tensor, pred_encoded: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate the given number of samples from the trivial copula.

        Parameters:
        -----------
        num_samples: int
            How many samples to generate, must be >= 1.
        hist_encoded: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step that does not have to be forecasted.
            The series and time steps dimensions are merged.
        hist_true_u: Tensor [batch, series * time steps]
            A tensor containing the true value for the values that do not have to be forecasted, transformed by the marginal distribution into U(0,1) values.
            The series and time steps dimensions are merged.
        pred_encoded: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step that does have to be forecasted.
            The series and time steps dimensions are merged.

        Returns:
        --------
        samples: torch.Tensor [batch, series * time steps, samples]
            Samples drawn from the trivial copula, which is equal to the multi-dimensional Uniform(0, 1) distribution.
            The series and time steps dimensions are merged.
        """
        num_batches, num_variables, _ = pred_encoded.shape
        device = pred_encoded.device
        return torch.rand(num_batches, num_variables, num_samples, device=device)


class GaussianDecoder(nn.Module):
    """
    A decoder which forecast using a low-rank multivariate Gaussian distribution.
    """

    def __init__(self):
        super().__init__()

    def loss(self, encoded: torch.Tensor, mask: torch.BoolTensor, true_value: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss function of the decoder.

        Parameters:
        -----------
        encoded: Tensor [batch, series, time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            This embedding is coming from the encoder, so contains shared information across series and time steps.
        mask: BoolTensor [batch, series, time steps]
            A tensor containing a mask indicating whether a given value was available for the encoder.
            The decoder only forecasts values for which the mask is set to False.
        true_value: Tensor [batch, series, time steps]
            A tensor containing the true value for the values to be forecasted.
            Only the values where the mask is set to False will be considered in the loss function.

        Returns:
        --------
        embedding: torch.Tensor [batch]
            The loss function, equal to the negative log likelihood of the distribution.
        """
        pass

    def sample(
        self, num_samples: int, encoded: torch.Tensor, mask: torch.BoolTensor, true_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate the given number of samples from the forecasted distribution.

        Parameters:
        -----------
        num_samples: int
            How many samples to generate, must be >= 1.
        encoded: Tensor [batch, series, time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            This embedding is coming from the encoder, so contains shared information across series and time steps.
        mask: BoolTensor [batch, series, time steps]
            A tensor containing a mask indicating whether a given value was available for the encoder.
            The decoder only forecasts values for which the mask is set to False.
        true_value: Tensor [batch, series, time steps]
            A tensor containing the true value for the values to be forecasted.
            The values where the mask is set to True will be copied as-is in the output.

        Returns:
        --------
        samples: torch.Tensor [batch, series, time steps, samples]
            Samples drawn from the forecasted distribution.
        """
        pass
