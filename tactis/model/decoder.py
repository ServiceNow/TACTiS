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

import math
from typing import Any, Dict, Optional, Type
import torch
import copy
from torch import nn
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


def _simple_linear_projection(input_dim: int, output_dim: int) -> nn.Sequential:
    layers = [nn.Linear(input_dim, output_dim)]
    return nn.Sequential(*layers)


def _easy_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    activation: Type[nn.Module],
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
        flow_input_dim: int,
        copula_input_dim: int,
        min_u: float = 0.0,
        max_u: float = 1.0,
        skip_sampling_marginal: bool = False,
        attentional_copula: Optional[Dict[str, Any]] = None,
        dsf_marginal: Optional[Dict[str, Any]] = None,
        skip_copula=False,
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

        self.flow_input_dim = flow_input_dim
        self.copula_input_dim = copula_input_dim
        self.min_u = min_u
        self.max_u = max_u
        self.skip_sampling_marginal = skip_sampling_marginal
        self.attentional_copula_args = attentional_copula
        self.dsf_marginal_args = dsf_marginal
        self.skip_copula = skip_copula

        if not self.skip_copula:
            if attentional_copula is not None:
                self.copula = AttentionalCopula(**attentional_copula)

        if dsf_marginal is not None:
            self.marginal = DSFMarginal(context_dim=flow_input_dim, **dsf_marginal)

        self.copula_loss = None
        self.marginal_logdet = None

    def create_attentional_copula(self):
        self.skip_copula = False
        if self.attentional_copula_args is not None:
            self.copula = AttentionalCopula(**self.attentional_copula_args)

    def loss(
        self,
        flow_encoded: torch.Tensor,
        copula_encoded: torch.Tensor,
        mask: torch.BoolTensor,
        true_value: torch.Tensor,
    ) -> torch.Tensor:
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
        B, S, T, E = flow_encoded.shape
        flow_encoded = _merge_series_time_dims(flow_encoded)
        if not self.skip_copula:
            copula_encoded = _merge_series_time_dims(copula_encoded)

        mask = _merge_series_time_dims(mask)
        true_value = _merge_series_time_dims(true_value)

        # Assume that the mask is constant inside the batch (every sample has the same hist-pred split)
        mask = mask[0, :]  # (series * time steps)

        hist_encoded_flow = flow_encoded[:, mask, :]  # [batch, series*(num_hist_timesteps), embedding]
        if not self.skip_copula:
            hist_encoded_copula = copula_encoded[:, mask, :]  # [batch, series*(num_hist_timesteps), embedding]
        pred_encoded_flow = flow_encoded[:, ~mask, :]
        if not self.skip_copula:
            pred_encoded_copula = copula_encoded[:, ~mask, :]
        history_factor = hist_encoded_flow.shape[1] / pred_encoded_flow.shape[1]

        hist_true_x = true_value[:, mask]  # [batch, series*(num_hist_timesteps)]
        pred_true_x = true_value[:, ~mask]

        num_pred_variables = round(T / (history_factor + 1))

        # Transform to [0,1] using the marginals
        hist_true_u = self.marginal.forward_no_logdet(hist_encoded_flow, hist_true_x)
        pred_true_u, marginal_logdet = self.marginal.forward_logdet(pred_encoded_flow, pred_true_x)

        if not self.skip_copula:
            copula_loss = self.copula.loss(
                hist_encoded=hist_encoded_copula,
                hist_true_u=hist_true_u,
                pred_encoded=pred_encoded_copula,
                pred_true_u=pred_true_u,
                num_series=S,
                num_timesteps=num_pred_variables,
            )
        else:
            copula_loss = torch.tensor(0.0).to(mask.device)

        self.copula_loss = copula_loss
        self.marginal_logdet = marginal_logdet

        # Loss = negative log likelihood
        return copula_loss - marginal_logdet

    def sample(
        self,
        num_samples: int,
        flow_encoded: torch.Tensor,
        copula_encoded: torch.Tensor,
        mask: torch.Tensor,
        true_value: torch.Tensor,
    ):
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

        flow_encoded = _merge_series_time_dims(flow_encoded)
        if not self.skip_copula:
            copula_encoded = _merge_series_time_dims(copula_encoded)

        mask = _merge_series_time_dims(mask)
        true_value = _merge_series_time_dims(true_value)

        # Assume that the mask is constant inside the batch
        mask = mask[0, :]
        hist_true_x = true_value[:, mask]

        hist_encoded_flow = flow_encoded[:, mask, :]
        if not self.skip_copula:
            hist_encoded_copula = copula_encoded[:, mask, :]
        pred_encoded_flow = flow_encoded[:, ~mask, :]
        if not self.skip_copula:
            pred_encoded_copula = copula_encoded[:, ~mask, :]

        # Transform to [0,1] using the marginals
        hist_true_u = self.marginal.forward_no_logdet(hist_encoded_flow, hist_true_x)

        if not self.skip_copula:
            pred_samples = self.copula.sample(
                num_samples=num_samples,
                hist_encoded=hist_encoded_copula,
                hist_true_u=hist_true_u,
                pred_encoded=pred_encoded_copula,
            )
        else:
            num_batches, num_variables, _ = pred_encoded_flow.shape
            pred_samples = torch.rand(num_batches, num_variables, num_samples, device=pred_encoded_flow.device)

        if not self.skip_sampling_marginal:
            # Transform away from [0,1] using the marginals
            # Transform to [min_u, max_u]
            pred_samples = self.min_u + (self.max_u - self.min_u) * pred_samples
            # Transform to the distribution of each token
            # Note that pred_encoded is passed here since it has token, variable information
            # In addition, pred_encoded also has context of the window from attention with other tokens in the window
            pred_samples = self.marginal.inverse(
                pred_encoded_flow,
                pred_samples,
            )

        samples = torch.zeros(
            target_shape[0],
            target_shape[1] * target_shape[2],
            target_shape[3],
            device=flow_encoded.device,
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
        attention_mlp_class: str = "_easy_mlp",
        activation_function: str = "relu",
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

        output_dims = self.attention_heads * self.attention_dim
        # Parameters for the attention layers in the copula
        # For each layer and each head, we have two MLP to create the keys and values
        # After each layer, we transform the embedding using a feed-forward network, consisting of
        # two linear layer with a ReLu in-between both
        # At the very beginning, we have a linear layer to change the embedding to the proper dimensionality
        self.dimension_shifting_layer = nn.Linear(self.input_dim, self.attention_heads * self.attention_dim)

        if activation_function == "relu":
            activation = nn.ReLU
        else:
            raise NotImplementedError("Activation functions other than ReLU are not implemented")

        if attention_mlp_class == "_easy_mlp":
            mlp_args_key = {
                "input_dim": self.input_dim + 1,  # + 1 since the marginals will be concatenated
                "hidden_dim": self.mlp_dim,
                "output_dim": self.attention_dim,
                "num_layers": self.mlp_layers,
                "activation": activation,
            }
            mlp_args_value = {
                "input_dim": self.input_dim + 1,  # + 1 since the marginals will be concatenated
                "hidden_dim": self.mlp_dim,
                "output_dim": self.attention_dim,
                "num_layers": self.mlp_layers,
                "activation": activation,
            }
            mlp_class = _easy_mlp
        elif attention_mlp_class == "_simple_linear_projection":
            mlp_args_key = {
                "input_dim": self.input_dim + 1,  # + 1 since the marginals will be concatenated
                "output_dim": self.attention_dim,
            }
            mlp_args_value = {
                "input_dim": self.input_dim + 1,  # + 1 since the marginals will be concatenated
                "output_dim": self.attention_dim,
            }
            mlp_class = _simple_linear_projection
        else:
            raise NotImplementedError()

        key_attention_heads = self.attention_heads
        value_attention_heads = self.attention_heads

        # one per layer and per head
        # The key and value creators take the input embedding together with the sampled [0,1] value as an input
        self.key_creators = nn.ModuleList(
            [
                nn.ModuleList([mlp_class(**mlp_args_key) for _ in range(key_attention_heads)])
                for _ in range(self.attention_layers)
            ]
        )
        self.value_creators = nn.ModuleList(
            [
                nn.ModuleList([mlp_class(**mlp_args_value) for _ in range(value_attention_heads)])
                for _ in range(self.attention_layers)
            ]
        )

        # one per layer
        self.attention_dropouts = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.attention_layers)])
        self.attention_layer_norms = nn.ModuleList(
            [nn.LayerNorm(output_dims, elementwise_affine=False) for _ in range(self.attention_layers)]
        )
        self.feed_forwards = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(output_dims, output_dims),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(output_dims, output_dims),
                    nn.Dropout(dropout),
                )
                for _ in range(self.attention_layers)
            ]
        )
        self.feed_forward_layer_norms = nn.ModuleList(
            [nn.LayerNorm(output_dims, elementwise_affine=False) for _ in range(self.attention_layers)]
        )

        # Parameter extractor for the categorical distribution
        self.dist_extractors = _easy_mlp(
            input_dim=output_dims,
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
        num_series: int,
        num_timesteps: int,
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

        assert num_variables == num_series * num_timesteps, (
            "num_variables:"
            + str(num_variables)
            + " but num_series:"
            + str(num_series)
            + " and num_timesteps:"
            + str(num_timesteps)
        )

        permutation = torch.arange(0, num_variables).long()

        # Permute the variables according the random permutation
        pred_encoded = pred_encoded[:, permutation, :]
        pred_true_u = pred_true_u[:, permutation]

        # At the beginning of the attention, we start with the input embedding.
        # Since it does not necessarily have the same dimensions as the hidden layers, apply a linear layer to scale it up.
        att_value = self.dimension_shifting_layer(pred_encoded)

        # # The MLP which generates the keys and values used the encoded embedding + transformed true values.
        # key_value_input_hist = torch.cat([hist_encoded, hist_true_u[:, :, None]], axis=2)
        # key_value_input_pred = torch.cat([pred_encoded, pred_true_u[:, :, None]], axis=2)
        # # key_value_input shape: [bsz, num_history+num_variables, embedding dimension+1]
        # key_value_input = torch.cat([key_value_input_hist, key_value_input_pred], axis=1)

        keys = []
        values = []
        for layer in range(self.attention_layers):
            key_input_hist = torch.cat([hist_encoded, hist_true_u[:, :, None]], axis=2)
            key_input_pred = torch.cat([pred_encoded, pred_true_u[:, :, None]], axis=2)
            key_input = torch.cat([key_input_hist, key_input_pred], axis=1)

            value_input_hist = torch.cat([hist_encoded, hist_true_u[:, :, None]], axis=2)
            value_input_pred = torch.cat([pred_encoded, pred_true_u[:, :, None]], axis=2)
            value_input = torch.cat([value_input_hist, value_input_pred], axis=1)

            # Keys shape in every layer: [bsz, num_attention_heads, num_history+num_variables, attention_dim]
            keys.append(
                torch.cat(
                    [mlp(key_input)[:, None, :, :] for mlp in self.key_creators[layer]],
                    axis=1,
                )
            )
            # Values shape in every layer: [bsz, num_attention_heads, num_history+num_variables, attention_dim]
            values.append(
                torch.cat(
                    [mlp(value_input)[:, None, :, :] for mlp in self.value_creators[layer]],
                    axis=1,
                )
            )

        for layer in range(self.attention_layers):
            # Split the hidden layer into its various heads
            # Basically the Query in the attention
            # Shape: [bsz, num_variables, num_attention_heads, attention_dim]
            att_value_heads = att_value.reshape(
                att_value.shape[0],
                att_value.shape[1],
                self.attention_heads,
                self.attention_dim,
            )

            # Attention layer, for each batch and head:
            # A_vi' = sum_w(softmax_w(sum_i(Q_vi * K_wi) / sqrt(d)) * V_wi')

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

            # Perform Attention
            # Einstein sum indices:
            # b: batch number
            # h: attention head number (Note the change in order for att_value_heads)
            # v: variable we want to predict
            # w: variable we want to get information from (history or prediction)
            # i: embedding dimension of the keys and queries (self.attention_dim)
            # Output shape: [bsz, num_attention_heads, num_variables, num_history+num_variables]
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
            # Output shape: [bsz, num_variables, num_attention_heads, attention_dim]
            att = torch.einsum("bhvw,bhwj->bvhj", weights, values[layer])

            # Merge back the various heads to allow the feed forwards module to share information between heads
            # Shape: [b, v, h*j]
            att_merged_heads = att.reshape(att.shape[0], att.shape[1], att.shape[2] * att.shape[3])
            # print("Q:", att_value_heads.shape, "K:", keys[layer].shape, "V:", values[layer].shape, "Mask:", product_mask.shape)
            # print("Attn:", att_merged_heads.shape, "Attn Scores", weights.shape)

            # Compute and add dropout
            att_merged_heads = self.attention_dropouts[layer](att_merged_heads)

            # att_value = att_value + att_merged_heads
            # Layernorm
            att_value = att_value + att_merged_heads
            att_value = self.attention_layer_norms[layer](att_value)

            # Add the contribution of the feed-forward layer, mixing up the heads
            att_feed_forward = self.feed_forwards[layer](att_value)
            att_value = att_value + att_feed_forward
            att_value = self.feed_forward_layer_norms[layer](att_value)

        # Assign each observed U(0,1) value to a bin. The clip is to avoid issues with numerical inaccuracies.
        # shape: [b, variables*timesteps - 1]
        target = torch.clip(
            torch.floor(pred_true_u[:, 1:] * self.resolution).long(),
            min=0,
            max=self.resolution - 1,
        )

        # Final shape of att_value would be: [bsz, num_variables, num_attention_heads*attention_dim]
        # Compute the (un-normalized) logarithm likelihood of the conditional distribution.
        # Note: This section could instead call a specialized module to allow for easier customization.
        # Get conditional distributions over bins for all variables but the first one.
        # The first one is considered to always be U(0,1), which has a constant logarithm likelihood of 0.
        logits = self.dist_extractors(att_value)[:, 1:, :]  # shape: [b, variables*timesteps - 1, self.resolution]

        # We multiply the probability by self.resolution to get the PDF of the continuous-by-part distribution.
        # prob = self.resolution * softmax(logits, dim=2)
        # logprob = log(self.resolution) + log_softmax(logits, dim=2)
        # Basically softmax of the logits, but the softmax is scaled according to the resolution instead of being in [0,1]
        logprob = math.log(self.resolution) + nn.functional.log_softmax(logits, dim=2)
        # For each batch + variable pair, we want the value of the logits associated with its true value (target):
        # logprob[batch,variable] = logits[batch,variable,target[batch,variable]]
        # Since gather wants the same number of dimensions for both tensors, add and remove a dummy third dimension.
        logprob = torch.gather(logprob, dim=2, index=target[:, :, None])[:, :, 0]

        return -logprob.sum(axis=1)  # Only keep the batch dimension

    def sample(
        self,
        num_samples: int,
        hist_encoded: torch.Tensor,
        hist_true_u: torch.Tensor,
        pred_encoded: torch.Tensor,
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

        permutation = torch.arange(0, num_variables).long()
        permutations = torch.stack([permutation for _ in range(num_samples)])

        key_value_input_hist = torch.cat([hist_encoded, hist_true_u[:, :, None]], axis=2)

        # The MLP which generates the keys and values used the encoded embedding + transformed true values.
        # Keys and values shape in every layer FOR HISTORY: [bsz, num_attention_heads, num_history, attention_dim]
        keys_hist = [
            torch.cat(
                [mlp(key_value_input_hist)[:, None, :, :] for mlp in self.key_creators[layer]],
                axis=1,
            )
            for layer in range(self.attention_layers)
        ]
        values_hist = [
            torch.cat(
                [mlp(key_value_input_hist)[:, None, :, :] for mlp in self.value_creators[layer]],
                axis=1,
            )
            for layer in range(self.attention_layers)
        ]

        # We will store the keys and values from the sampled variables as we do the sampling
        # We have an additional dimension num_samples here
        samples = torch.zeros(num_batches, num_variables, num_samples).to(device)
        keys_samples = [
            torch.zeros(
                num_batches,
                num_samples,
                self.attention_heads,
                num_variables,
                self.attention_dim,
                device=device,
            )
            for _ in range(self.attention_layers)
        ]
        values_samples = [
            torch.zeros(
                num_batches,
                num_samples,
                self.attention_heads,
                num_variables,
                self.attention_dim,
                device=device,
            )
            for _ in range(self.attention_layers)
        ]

        # Importantly, at every step (for every variable), the attention outputs a vector of size [bsz, num_samples]
        # We sample the copula one variable at a time, following the order from the drawn permutation (for each sample)
        # Sampling of the ith variable is done for all samples
        for i in range(num_variables):
            # Vector containing which variable we sample at this step of the copula. It is different for each sample.
            # Shape: [num_samples]
            p = permutations[:, i]
            # Note that second dimension here no longer represent the variables (as in the loss method), but the samples.
            # Shape: [num_batches, num_samples, embedding dimension]
            current_pred_encoded = pred_encoded[:, p, :]

            if i == 0:
                # By construction, the first variable to be sampled is always sampled according to a Uniform(0,1).
                current_samples = torch.rand(num_batches, num_samples, device=device)
            else:
                att_value = self.dimension_shifting_layer(current_pred_encoded)

                for layer in range(self.attention_layers):
                    # Split the hidden layer into its various heads
                    # Basically the Query in the attention
                    # Shape: [bsz, num_samples, num_attention_heads, attention_dim]
                    att_value_heads = att_value.reshape(
                        att_value.shape[0],
                        att_value.shape[1],
                        self.attention_heads,
                        self.attention_dim,
                    )

                    keys_hist_current_layer = keys_hist[layer]
                    keys_samples_current_layer = keys_samples[layer][:, :, :, 0:i, :]
                    values_hist_current_layer = values_hist[layer]
                    values_samples_current_layer = values_samples[layer][:, :, :, 0:i, :]

                    # Calculate attention weights
                    # Einstein sum indices:
                    # b: batch number
                    # n: sample number of the current variable (treated as a separate variable)
                    # h: attention head number
                    # w: variable we want to get information from (history or prediction)
                    # i: embedding dimension of the keys and queries (self.input_dim)
                    product_hist = torch.einsum("bnhi,bhwi->bnhw", att_value_heads, keys_hist_current_layer)
                    # keys_samples is full of zero starting at i of the 4th dimension (w)
                    # Shape of product_samples: [bsz, num_samples, num_attention_heads, i]
                    # i since only i variables can be seen
                    product_samples = torch.einsum("bnhi,bnhwi->bnhw", att_value_heads, keys_samples_current_layer)
                    # # NOTE: product_samples could also be computed as follows
                    # b, n, h, d = att_value_heads.shape
                    # b, n, h, w, d = keys_samples_current_layer.shape
                    # product_samples = torch.einsum(
                    #     "bhi,bhwi->bhw", att_value_heads.reshape(b*n, h, d), keys_samples_current_layer.reshape(b*n, h, w, d)
                    # ).reshape(b, n, h, w)

                    # Combine the attention from the history and from the previous samples.
                    # Product is of shape [bsz, num_samples, num_attention_heads, num_history+i]
                    product = torch.cat([product_hist, product_samples], axis=3)
                    product = self.attention_dim ** (-0.5) * product

                    weights = nn.functional.softmax(product, dim=3)
                    weights_hist = weights[:, :, :, :num_history]
                    weights_samples = weights[:, :, :, num_history:]

                    # Get attention representation using weights (for conditional distribution)
                    # Einstein sum indices:
                    # b: batch number
                    # n: sample number of the current variable (treated as a separate variable)
                    # h: attention head number
                    # w: variable we want to get information from (history or prediction)
                    # j: embedding dimension of the values (self.hid_dim)
                    # att_hist is of shape [bsz, num_samples, num_attention_heads, hid_dim]
                    att_hist = torch.einsum("bnhw,bhwj->bnhj", weights_hist, values_hist_current_layer)
                    # att_hist is of shape [bsz, num_samples, num_attention_heads, hid_dim]
                    att_samples = torch.einsum(
                        "bnhw,bnhwj->bnhj",
                        weights_samples,
                        values_samples_current_layer,
                    )  # i >= 1
                    # # NOTE: att_samples could also be computed as follows
                    # b, n, h, w = weights_samples.shape
                    # b, n, h, w, j = values_samples_current_layer.shape
                    # att_samples = torch.einsum(
                    #     "bhw,bhwj->bhj", weights_samples.reshape(b*n, h, w), values_samples_current_layer.reshape(b*n, h, w, j)
                    # ).reshape(b, n, h, j)

                    att = att_hist + att_samples

                    # Merge back the various heads to allow the feed forwards module to share information between heads
                    att_merged_heads = att.reshape(att.shape[0], att.shape[1], att.shape[2] * att.shape[3])

                    att_merged_heads = self.attention_dropouts[layer](att_merged_heads)
                    att_value = att_value + att_merged_heads
                    att_value = self.attention_layer_norms[layer](att_value)
                    att_feed_forward = self.feed_forwards[layer](att_value)
                    att_value = att_value + att_feed_forward
                    att_value = self.feed_forward_layer_norms[layer](att_value)

                # Below code is executed once all the layers of attention are complete
                # att_value is of shape [bsz, num_samples, attn_heads*attn_dim]
                # Get the output distribution parameters
                logits = self.dist_extractors(att_value).reshape(num_batches * num_samples, self.resolution)

                # Select a single variable in {0, 1, 2, ..., self.resolution-1} according to the probabilities from the softmax
                # Why not the variable corresponding to the maximum value?
                current_samples = torch.multinomial(input=torch.softmax(logits, dim=1), num_samples=1)
                # Each point in the same bucket is equiprobable, and we used a floor function in the training
                current_samples = current_samples + torch.rand(*current_samples.shape).to(device)
                # Normalize to a variable in the [0, 1) range
                current_samples /= self.resolution
                current_samples = current_samples.reshape(num_batches, num_samples)

            # Compute the key and value associated with the newly sampled variable, for the attention of the next ones.
            # Shape: [num_batches, num_samples, embedding dimension+1]
            key_value_input = torch.cat([current_pred_encoded, current_samples[:, :, None]], axis=-1)
            for layer in range(self.attention_layers):
                # Shape: [num_batches, num_samples, num_attention_heads, attn_dim]
                new_keys = torch.cat(
                    [k(key_value_input)[:, :, None, :] for k in self.key_creators[layer]],
                    axis=2,
                )
                # Shape: [num_batches, num_samples, num_attention_heads, attn_dim]
                new_values = torch.cat(
                    [v(key_value_input)[:, :, None, :] for v in self.value_creators[layer]],
                    axis=2,
                )
                # Store the computed keys and values for the ith variable in the ith slot
                keys_samples[layer][:, :, :, i, :] = new_keys
                values_samples[layer][:, :, :, i, :] = new_values

            # p is of length num_samples, and each value indicates the variable sampled at this i'th timestep
            # Store the samples at the respective positions indicated by `p` for each sample
            # Collate the results, reversing the effect of the permutation
            # By using two lists of equal lengths, the resulting slice will be 2d, not 3d.
            samples[:, p, range(num_samples)] = current_samples

        return samples
