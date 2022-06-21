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


import math
from typing import Any, Dict, Optional, Tuple, Type

import torch
from torch import nn
from torch.distributions import LowRankMultivariateNormal
from torch.nn import functional

from .marginal import DSFMarginal


def _merge_series_time_dims(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a Tensor with dimensions [batch, series, time steps, ...] to one with dimensions [batch, series * time steps, ...]
    """
    assert x.dim() >= 3
    return x.view((x.shape[0], x.shape[1] * x.shape[2]) + x.shape[3:])


def _split_series_time_dims(x: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """
    Convert a Tensor with dimensions [batch, series * time steps, ...] to one with dimensions [batch, series, time steps, ...]
    """
    assert x.dim() + 1 == len(target_shape)
    return x.view(target_shape)


def _easy_mlp(
    input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, activation: Type[nn.Module]
) -> nn.Sequential:
    """
    Generate a MLP with the given parameters.
    """
    elayers = [nn.Linear(input_dim, hidden_dim), activation()]
    for _ in range(1, num_layers):
        elayers += [nn.Linear(hidden_dim, hidden_dim), activation()]
    elayers += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*elayers)


class CopulaDecoder(nn.Module):
    """
    A decoder which forecast using a distribution built from a copula and marginal distributions.
    """

    def __init__(
        self,
        input_dim: int,
        min_u: float = 0.0,
        max_u: float = 1.0,
        skip_sampling_marginal: bool = False,
        trivial_copula: Optional[Dict[str, Any]] = None,
        attentional_copula: Optional[Dict[str, Any]] = None,
        dsf_marginal: Optional[Dict[str, Any]] = None,
    ):
        """
        Parameters:
        -----------
        input_dim: int
            The dimension of the encoded representation (upstream data encoder).
        min_u: float, default to 0.0
        max_u: float, default to 1.0
            The values sampled from the copula will be scaled from [0, 1] to [min_u, max_u] before being sent to the marginal.
        skip_sampling_marginal: bool, default to False
            If set to True, then the output from the copula will not be transformed using the marginal during sampling.
            Does not impact the other transformations from observed values to the [0, 1] range.
        trivial_copula: Dict[str, Any], default to None
            If set to a non-None value, uses a TrivialCopula.
            The options sent to the TrivialCopula is content of this dictionary.
        attentional_copula: Dict[str, Any], default to None
            If set to a non-None value, uses a AttentionalCopula.
            The options sent to the AttentionalCopula is content of this dictionary.
        dsf_marginal: Dict[str, Any], default to None
            If set to a non-None value, uses a DSFMarginal.
            The options sent to the DSFMarginal is content of this dictionary.
        """
        super().__init__()

        assert (trivial_copula is not None) + (
            attentional_copula is not None
        ) == 1, "Must select exactly one type of copula"
        assert (dsf_marginal is not None) == 1, "Must select exactly one type of marginal"

        self.min_u = min_u
        self.max_u = max_u
        self.skip_sampling_marginal = skip_sampling_marginal

        if trivial_copula is not None:
            self.copula = TrivialCopula(**trivial_copula)
        if attentional_copula is not None:
            self.copula = AttentionalCopula(input_dim=input_dim, **attentional_copula)

        if dsf_marginal is not None:
            self.marginal = DSFMarginal(context_dim=input_dim, **dsf_marginal)

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
        encoded = _merge_series_time_dims(encoded)
        mask = _merge_series_time_dims(mask)
        true_value = _merge_series_time_dims(true_value)

        # Assume that the mask is constant inside the batch
        mask = mask[0, :]

        hist_encoded = encoded[:, mask, :]
        pred_encoded = encoded[:, ~mask, :]
        hist_true_x = true_value[:, mask]
        pred_true_x = true_value[:, ~mask]

        # Transform to [0,1] using the marginals
        hist_true_u = self.marginal.forward_no_logdet(hist_encoded, hist_true_x)
        pred_true_u, marginal_logdet = self.marginal.forward_logdet(pred_encoded, pred_true_x)

        copula_loss = self.copula.loss(
            hist_encoded=hist_encoded,
            hist_true_u=hist_true_u,
            pred_encoded=pred_encoded,
            pred_true_u=pred_true_u,
        )

        # Loss = negative log likelihood
        return copula_loss - marginal_logdet

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
            A tensor containing a mask indicating whether a given value is masked (available) for the encoder.
            The decoder only forecasts values for which the mask is set to False.
        true_value: Tensor [batch, series, time steps]
            A tensor containing the true value for the values to be forecasted.
            The values where the mask is set to True will be copied as-is in the output.

        Returns:
        --------
        samples: torch.Tensor [batch, series, time steps, samples]
            Samples drawn from the forecasted distribution.
        """
        target_shape = torch.Size((true_value.shape[0], true_value.shape[1], true_value.shape[2], num_samples))

        encoded = _merge_series_time_dims(encoded)
        mask = _merge_series_time_dims(mask)
        true_value = _merge_series_time_dims(true_value)

        # Assume that the mask is constant inside the batch
        mask = mask[0, :]

        hist_encoded = encoded[:, mask, :]
        pred_encoded = encoded[:, ~mask, :]
        hist_true_x = true_value[:, mask]

        # Transform to [0,1] using the marginals
        hist_true_u = self.marginal.forward_no_logdet(hist_encoded, hist_true_x)

        pred_samples = self.copula.sample(
            num_samples=num_samples,
            hist_encoded=hist_encoded,
            hist_true_u=hist_true_u,
            pred_encoded=pred_encoded,
        )
        if not self.skip_sampling_marginal:
            # Transform away from [0,1] using the marginals
            pred_samples = self.min_u + (self.max_u - self.min_u) * pred_samples
            pred_samples = self.marginal.inverse(
                pred_encoded,
                pred_samples,
            )

        samples = torch.zeros(
            target_shape[0], target_shape[1] * target_shape[2], target_shape[3], device=encoded.device
        )
        samples[:, mask, :] = hist_true_x[:, :, None]
        samples[:, ~mask, :] = pred_samples

        return _split_series_time_dims(samples, target_shape)


class AttentionalCopula(nn.Module):
    """
    A non-parametric copula based on attention between the various variables.
    """

    def __init__(
        self,
        input_dim: int,
        attention_heads: int,
        attention_layers: int,
        attention_dim: int,
        mlp_layers: int,
        mlp_dim: int,
        resolution: int = 10,
        dropout: float = 0.1,
        fixed_permutation: bool = False,
    ):
        """
        Parameters:
        -----------
        input_dim: int
            Dimension of the encoded representation.
        attention_heads: int
            How many independant heads the attention layer will have. Each head will have its own independant MLP
            to generate the keys and values.
        attention_layers: int
            How many successive attention layers copula will use. Each layer will have its own independant MLPs
            to generate the keys and values.
        attention_dim: int
            The size of the attention layer output.
        mlp_layers: int
            The number of hidden layers in the MLP that produces the keys and values for the attention layer,
            and in the MLP that takes the attention output to generate the distribution parameter.
        mlp_dim: int
            The size of the hidden layers in the MLP that produces the keys and values for the attention layer,
            and in the MLP that takes the attention output to generate the distribution parameter.
        resolution: int, default to 10
            How many bins to pick from when sampling variables.
            Higher values are more precise, but slower to train.
        dropout: float, default to 0.1
            Dropout parameter for the attention.
        fixed_permutation: bool, default False
            If set to true, then the copula always use the same permutation, instead of using random ones.
        """
        super().__init__()

        self.input_dim = input_dim
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.attention_dim = attention_dim
        self.mlp_layers = mlp_layers
        self.mlp_dim = mlp_dim
        self.resolution = resolution
        self.dropout = dropout
        self.fixed_permutation = fixed_permutation

        # Parameters for the attention layers in the copula
        # For each layer and each head, we have two MLP to create the keys and values
        # After each layer, we transform the embedding using a feed-forward network, consisting of
        # two linear layer with a ReLu in-between both
        # At the very beginning, we have a linear layer to change the embedding to the proper dimensionality
        self.dimension_shifting_layer = nn.Linear(self.input_dim, self.attention_heads * self.attention_dim)

        # one per layer and per head
        # The key and value creators take the input embedding together with the sampled [0,1] value as an input
        self.key_creators = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _easy_mlp(
                            input_dim=self.input_dim + 1,
                            hidden_dim=self.mlp_dim,
                            output_dim=self.attention_dim,
                            num_layers=self.mlp_layers,
                            activation=nn.ReLU,
                        )
                        for _ in range(self.attention_heads)
                    ]
                )
                for _ in range(self.attention_layers)
            ]
        )
        self.value_creators = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _easy_mlp(
                            input_dim=self.input_dim + 1,
                            hidden_dim=self.mlp_dim,
                            output_dim=self.attention_dim,
                            num_layers=self.mlp_layers,
                            activation=nn.ReLU,
                        )
                        for _ in range(self.attention_heads)
                    ]
                )
                for _ in range(self.attention_layers)
            ]
        )

        # one per layer
        self.attention_dropouts = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.attention_layers)])
        self.attention_layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.attention_heads * self.attention_dim) for _ in range(self.attention_layers)]
        )
        self.feed_forwards = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.attention_heads * self.attention_dim, self.attention_heads * self.attention_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.attention_heads * self.attention_dim, self.attention_heads * self.attention_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(self.attention_layers)
            ]
        )
        self.feed_forward_layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.attention_heads * self.attention_dim) for _ in range(self.attention_layers)]
        )

        # Parameter extractor for the categorical distribution
        self.dist_extractors = _easy_mlp(
            input_dim=self.attention_heads * self.attention_dim,
            hidden_dim=self.mlp_dim,
            output_dim=self.resolution,
            num_layers=self.mlp_layers,
            activation=nn.ReLU,
        )

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
        loss: torch.Tensor [batch]
            The loss function, equal to the negative log likelihood of the copula.
        """
        num_batches = pred_encoded.shape[0]
        num_variables = pred_encoded.shape[1]
        num_history = hist_encoded.shape[1]
        device = pred_encoded.device

        if self.fixed_permutation:
            # This fixed permutation would better be done series by series, instead of time step by time step.
            # However, we cannot have the other behaviour without explicitly sending the number of series (or time steps).
            permutation = torch.range(0, num_variables)
        else:
            permutation = torch.randperm(num_variables)

        # Permute the variables according the random permutation
        pred_encoded = pred_encoded[:, permutation, :]
        pred_true_u = pred_true_u[:, permutation]

        # The MLP which generates the keys and values used the encoded embedding + transformed true values.
        key_value_input_hist = torch.cat([hist_encoded, hist_true_u[:, :, None]], axis=2)
        key_value_input_pred = torch.cat([pred_encoded, pred_true_u[:, :, None]], axis=2)
        key_value_input = torch.cat([key_value_input_hist, key_value_input_pred], axis=1)

        keys = [
            torch.cat([mlp(key_value_input)[:, None, :, :] for mlp in self.key_creators[layer]], axis=1)
            for layer in range(self.attention_layers)
        ]
        values = [
            torch.cat([mlp(key_value_input)[:, None, :, :] for mlp in self.value_creators[layer]], axis=1)
            for layer in range(self.attention_layers)
        ]

        # During attention, we will add -float("inf") to pairs of indices where the variable to be forecasted (query)
        # is after the variable that gives information (key), after the random permutation.
        # Doing this prevent information from flowing from later in the permutation to before in the permutation,
        # which cannot happen during inference.
        # tril fill the diagonal and values that are below it, flip rotates it by 180 degrees,
        # leaving only the pairs of indices which represent not yet sampled values.
        # Note float("inf") * 0 is unsafe, so do the multiplication inside the torch.tril()
        # pred/hist_encoded dimensions: number of batches, number of variables, size of embedding per variable
        product_mask = torch.ones(
            num_batches,
            self.attention_heads,
            num_variables,
            num_variables + num_history,
            device=device,
        )
        product_mask = torch.tril(float("inf") * product_mask).flip((2, 3))

        # At the beginning of the attention, we start with the input embedding.
        # Since it does not necessarily have the same dimensions as the hidden layers, apply a linear layer to scale it up.
        att_value = self.dimension_shifting_layer(pred_encoded)

        for layer in range(self.attention_layers):
            # Split the hidden layer into its various heads
            att_value_heads = att_value.reshape(
                att_value.shape[0], att_value.shape[1], self.attention_heads, self.attention_dim
            )

            # Attention layer, for each batch and head:
            # A_vi' = sum_w(softmax_w(sum_i(Q_vi * K_wi) / sqrt(d)) * V_wi')

            # Einstein sum indices:
            # b: batch number
            # h: attention head number (Note the change in order for att_value_heads)
            # v: variable we want to predict
            # w: variable we want to get information from (history or prediction)
            # i: embedding dimension of the keys and queries (self.attention_dim)
            product_base = torch.einsum("bvhi,bhwi->bhvw", att_value_heads, keys[layer])

            # Adding -inf shunts the attention to zero, for any variable that has not "yet" been predicted,
            # aka: are in the future according to the permutation.
            product = product_base - product_mask
            product = self.attention_dim ** (-0.5) * product
            weights = nn.functional.softmax(product, dim=-1)

            # Einstein sum indices:
            # b: batch number
            # h: attention head number (Note the change in order for the result)
            # v: variable we want to predict
            # w: variable we want to get information from (history or prediction)
            # j: embedding dimension of the values (self.attention_dim)
            att = torch.einsum("bhvw,bhwj->bvhj", weights, values[layer])

            # Merge back the various heads to allow the feed forwards module to share information between heads
            att_merged_heads = att.reshape(att.shape[0], att.shape[1], att.shape[2] * att.shape[3])
            att_merged_heads = self.attention_dropouts[layer](att_merged_heads)
            att_value = att_value + att_merged_heads
            att_value = self.attention_layer_norms[layer](att_value)
            att_feed_forward = self.feed_forwards[layer](att_value)
            att_value = att_value + att_feed_forward
            att_value = self.feed_forward_layer_norms[layer](att_value)

        # Compute the logarithm likelihood of the conditional distribution.
        # Note: This section could instead call a specialized module to allow for easier customization.
        # Get conditional distributions over bins for all variables but the first one.
        # The first one is considered to always be U(0,1), which has a constant logarithm likelihood of 0.
        logits = self.dist_extractors(att_value)[:, 1:, :]

        # Assign each observed U(0,1) value to a bin. The clip is to avoid issues with numerical inaccuracies.
        target = torch.clip(torch.floor(pred_true_u[:, 1:] * self.resolution).long(), min=0, max=self.resolution - 1)

        # We multiply the probability by self.resolution to get the PDF of the continuous-by-part distribution.
        logprob = math.log(self.resolution) + nn.functional.log_softmax(logits, dim=2)
        # For each batch + variable pair, we want the value of the logits associated with its true value (target):
        # logprob[batch,variable] = logits[batch,variable,target[batch,variable]]
        # Since gather wants the same number of dimensions for both tensors, add and remove a dummy third dimension.
        logprob = torch.gather(logprob, dim=2, index=target[:, :, None])[:, :, 0]

        return -logprob.sum(axis=1)  # Only keep the batch dimension

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
        num_batches = pred_encoded.shape[0]
        num_variables = pred_encoded.shape[1]
        num_history = hist_encoded.shape[1]
        device = pred_encoded.device

        if self.fixed_permutation:
            # This fixed permutation would better be done series by series, instead of time step by time step.
            # However, we cannot have the other behaviour without explicitly sending the number of series (or time steps).
            permutations = torch.stack([torch.range(0, num_variables) for _ in range(num_samples)])
        else:
            # Have an independant permutation for each sample.
            # Note that different elements of a single batch will share the same permutations.
            # This was done due to avoid an overly complex implementation,
            # but it does has an impact on the sampling accuracy if num_samples is small and num_batches is large
            # (aka: when the sampling of a given forecast is spread over multiple entries of a single batch).
            permutations = torch.stack([torch.randperm(num_variables) for _ in range(num_samples)])

        # The MLP which generates the keys and values used the encoded embedding + transformed true values.
        key_value_input_hist = torch.cat([hist_encoded, hist_true_u[:, :, None]], axis=2)
        keys_hist = [
            torch.cat([mlp(key_value_input_hist)[:, None, :, :] for mlp in self.key_creators[layer]], axis=1)
            for layer in range(self.attention_layers)
        ]
        values_hist = [
            torch.cat([mlp(key_value_input_hist)[:, None, :, :] for mlp in self.value_creators[layer]], axis=1)
            for layer in range(self.attention_layers)
        ]

        # We will store the keys and values from the sampled variables as we do the sampling
        samples = torch.zeros(num_batches, num_variables, num_samples).to(device)
        keys_samples = [
            torch.zeros(
                num_batches, num_samples, self.attention_heads, num_variables, self.attention_dim, device=device
            )
            for _ in range(self.attention_layers)
        ]
        values_samples = [
            torch.zeros(
                num_batches, num_samples, self.attention_heads, num_variables, self.attention_dim, device=device
            )
            for _ in range(self.attention_layers)
        ]

        # We sample the copula one variable at a time, following the order from the drawn permutation.
        for i in range(num_variables):
            # Vector containing which variable we sample at this step of the copula.
            p = permutations[:, i]
            # Note that second dimension here no longer represent the variables (as in the loss method), but the samples.
            current_pred_encoded = pred_encoded[:, p, :]

            if i == 0:
                # By construction, the first variable to be sampled is always sampled according to a Uniform(0,1).
                current_samples = torch.rand(num_batches, num_samples, device=device)
            else:
                att_value = self.dimension_shifting_layer(current_pred_encoded)

                for layer in range(self.attention_layers):
                    # Split the hidden layer into its various heads
                    att_value_heads = att_value.reshape(
                        att_value.shape[0], att_value.shape[1], self.attention_heads, self.attention_dim
                    )

                    # Calculate attention weights
                    # Einstein sum indices:
                    # b: batch number
                    # n: sample number
                    # h: attention head number
                    # w: variable we want to get information from (history or prediction)
                    # i: embedding dimension of the keys and queries (self.input_dim)
                    product_hist = torch.einsum("bnhi,bhwi->bnhw", att_value_heads, keys_hist[layer])
                    # keys_samples is full of zero starting at i of the 4th dimension (w)
                    product_samples = torch.einsum(
                        "bnhi,bnhwi->bnhw", att_value_heads, keys_samples[layer][:, :, :, 0:i, :]
                    )
                    # Combine the attention from the history and from the previous samples.
                    product = torch.cat([product_hist, product_samples], axis=3)
                    product = self.attention_dim ** (-0.5) * product

                    weights = nn.functional.softmax(product, dim=3)
                    weights_hist = weights[:, :, :, :num_history]
                    weights_samples = weights[:, :, :, num_history:]

                    # Get attention representation using weights (for conditional distribution)
                    # Einstein sum indices:
                    # b: batch number
                    # n: sample number
                    # h: attention head number
                    # w: variable we want to get information from (history or prediction)
                    # j: embedding dimension of the values (self.hid_dim)
                    att_hist = torch.einsum("bnhw,bhwj->bnhj", weights_hist, values_hist[layer])
                    att_samples = torch.einsum(
                        "bnhw,bnhwj->bnhj", weights_samples, values_samples[layer][:, :, :, 0:i, :]
                    )  # i >= 1
                    att = att_hist + att_samples

                    # Merge back the various heads to allow the feed forwards module to share information between heads
                    att_merged_heads = att.reshape(att.shape[0], att.shape[1], att.shape[2] * att.shape[3])
                    att_merged_heads = self.attention_dropouts[layer](att_merged_heads)
                    att_value = att_value + att_merged_heads
                    att_value = self.attention_layer_norms[layer](att_value)
                    att_feed_forward = self.feed_forwards[layer](att_value)
                    att_value = att_value + att_feed_forward
                    att_value = self.feed_forward_layer_norms[layer](att_value)

                # Get the output distribution parameters
                logits = self.dist_extractors(att_value).reshape(num_batches * num_samples, self.resolution)
                # Select a single variable in {0, 1, 2, ..., self.resolution-1} according to the probabilities from the softmax
                current_samples = torch.multinomial(input=torch.softmax(logits, dim=1), num_samples=1)
                # Each point in the same bucket is equiprobable, and we used a floor function in the training
                current_samples = current_samples + torch.rand(*current_samples.shape).to(device)
                # Normalize to a variable in the [0, 1) range
                current_samples /= self.resolution
                current_samples = current_samples.reshape(num_batches, num_samples)

            # Compute the key and value associated with the newly sampled variable, for the attention of the next ones.
            key_value_input = torch.cat([current_pred_encoded, current_samples[:, :, None]], axis=-1)
            for layer in range(self.attention_layers):
                new_keys = torch.cat([k(key_value_input)[:, :, None, :] for k in self.key_creators[layer]], axis=2)
                new_values = torch.cat([v(key_value_input)[:, :, None, :] for v in self.value_creators[layer]], axis=2)
                keys_samples[layer][:, :, :, i, :] = new_keys
                values_samples[layer][:, :, :, i, :] = new_values

            # Collate the results, reversing the effect of the permutation
            # By using two lists of equal lengths, the resulting slice will be 2d, not 3d.
            samples[:, p, range(num_samples)] = current_samples

        return samples


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

    def __init__(
        self,
        input_dim: int,
        matrix_rank: int,
        mlp_layers: int,
        mlp_dim: int,
        min_d: float = 0.01,
        max_v: float = 50.0,
    ):
        """
        Parameters:
        -----------
        input_dim: int
            Dimension of the encoded representation.
        matrix_rank: int
            Rank of the covariance matrix, prior to adding its diagonal component.
        mlp_layers: int
            The number of hidden layers in the MLP that produces the components of the covariance matrix.
        mlp_dim: int
            The size of the hidden layers in the MLP that produces the components of the covariance matrix.
        min_d: float, default to 0.01
            Minimum value of the diagonal component of the covariance matrix.
            Too low values can lead to exceptions due to numerical errors.
        max_v: float, default to 50.0
            Maximum weight of the contribution from the latent variables to the observed variables.
            Too high values can lead to exceptions due to numerical errors.
        """
        super().__init__()

        self.input_dim = input_dim
        self.matrix_rank = matrix_rank
        self.mlp_layers = mlp_layers
        self.mlp_dim = mlp_dim
        self.min_d = min_d
        self.max_v = max_v

        # The covariance matrix low-rank approximation is as such:
        # Cov = V * V^t + d
        # Where V is a number of variables * matrix rank rectangular matrix, and d is diagonal.
        # This gives the same covariance as having matrix rank latent Normal(0,1) variables, and generating the output as:
        # output_i = sum_j V_ij latent_j + N(0, d_i)
        self.param_V_extractor = _easy_mlp(
            input_dim=self.input_dim,
            hidden_dim=self.mlp_dim,
            output_dim=self.matrix_rank,
            num_layers=self.mlp_layers,
            activation=nn.ReLU,
        )
        self.param_d_extractor = _easy_mlp(
            input_dim=self.input_dim,
            hidden_dim=self.mlp_dim,
            output_dim=1,
            num_layers=self.mlp_layers,
            activation=nn.ReLU,
        )
        self.param_mean_extractor = _easy_mlp(
            input_dim=self.input_dim,
            hidden_dim=self.mlp_dim,
            output_dim=1,
            num_layers=self.mlp_layers,
            activation=nn.ReLU,
        )

    def extract_params(self, pred_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the parameters of the low-rank Gaussian distribution.

        Parameters:
        -----------
        pred_encoded: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.

        Returns:
        --------
        param_mean: torch.Tensor [batch, series * time steps]
            The mean of the Gaussian distribution
        param_d: torch.Tensor [batch, series * time steps]
            The diagonal of the Gaussian distribution covariance matrix
        param_V: torch.Tensor [batch, series * time steps, matrix rank]
            The contribution to each variable from each latent variable
        """
        # The last dimension of the mean and d parameters is a dummy one
        param_mean = self.param_mean_extractor(pred_encoded)[:, :, 0]
        # Parametrized covariance matrix d + V*V^t
        # This is the same parametrization as what Salinas et al. (2019) used for the covariance matrix.
        # An upper bound of the condition number of the matrix that will be used in the logdet or Cholesky is:
        # 1 + hid_dim * max_v^2 / min_d
        # We add these limits since a condition number near or above 2^23 will lead to grave numerical instability in the Cholesky decomposition.
        param_d = functional.softplus(self.param_d_extractor(pred_encoded))[:, :, 0]
        param_d = param_d + self.min_d
        param_V = self.param_V_extractor(pred_encoded)
        param_V = torch.tanh(param_V / self.max_v) * self.max_v

        return param_mean, param_d, param_V

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
        encoded = _merge_series_time_dims(encoded)
        mask = _merge_series_time_dims(mask)
        true_value = _merge_series_time_dims(true_value)

        # Assume that the mask is constant inside the batch
        mask = mask[0, :]

        # Ignore the encoding from the historical variables, since there are no interaction between the variables in this decoder.
        pred_encoded = encoded[:, ~mask, :]
        pred_true_x = true_value[:, ~mask]

        param_mean, param_d, param_V = self.extract_params(pred_encoded)
        log_prob = LowRankMultivariateNormal(loc=param_mean, cov_factor=param_V, cov_diag=param_d).log_prob(pred_true_x)

        return -log_prob

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
        num_batches = encoded.shape[0]
        num_series = encoded.shape[1]
        num_timesteps = encoded.shape[2]
        device = encoded.device

        encoded = _merge_series_time_dims(encoded)
        mask = _merge_series_time_dims(mask)
        true_value = _merge_series_time_dims(true_value)

        # Assume that the mask is constant inside the batch
        mask = mask[0, :]

        # Ignore the encoding from the historical variables, since there are no interaction between the variables in this decoder.
        pred_encoded = encoded[:, ~mask, :]
        # Except what is needed to copy to the output
        hist_true_x = true_value[:, mask]

        param_mean, param_d, param_V = self.extract_params(pred_encoded)

        dist = LowRankMultivariateNormal(loc=param_mean, cov_factor=param_V, cov_diag=param_d)
        # rsamples have the samples as the first dimension, so send it to the last dimension
        pred_samples = dist.rsample((num_samples,)).permute((1, 2, 0))

        samples = torch.zeros(num_batches, num_series * num_timesteps, num_samples, device=device)
        samples[:, mask, :] = hist_true_x[:, :, None]
        samples[:, ~mask, :] = pred_samples

        return _split_series_time_dims(samples, torch.Size((num_batches, num_series, num_timesteps, num_samples)))
