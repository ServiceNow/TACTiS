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
"""

import torch
from tactis.model.decoder import (
    AttentionalCopula,
    CopulaDecoder,
    GaussianDecoder,
    TrivialCopula,
    _merge_series_time_dims,
    _split_series_time_dims,
)
from torch import nn


def test_merge_series():
    """
    Test that the series merging method gets the right dimensions
    """
    assert _merge_series_time_dims(torch.ones(2, 3, 4)).shape == (2, 12)
    assert _merge_series_time_dims(torch.ones(2, 3, 4, 5)).shape == (2, 12, 5)
    assert _merge_series_time_dims(torch.ones(2, 3, 4, 5, 6)).shape == (2, 12, 5, 6)


def test_split_series():
    """
    Test that the series splitting method gets the right dimensions
    """
    assert _split_series_time_dims(torch.ones(2, 12), (2, 3, 4)).shape == (2, 3, 4)
    assert _split_series_time_dims(torch.ones(2, 12, 5), (2, 3, 4, 5)).shape == (2, 3, 4, 5)
    assert _split_series_time_dims(torch.ones(2, 12, 5, 6), (2, 3, 4, 5, 6)).shape == (2, 3, 4, 5, 6)


class __ParameterSavingShell(nn.Module):
    """
    A module that goes around a copula or a marginal.
    It saves all of the parameters sent to the various methods for testing purposes.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_params = {}
        self.sample_params = {}
        self.forward_logdet_params = {}
        self.forward_no_logdet_params = {}
        self.inverse_params = {}

    def loss(self, hist_encoded, hist_true_u, pred_encoded, pred_true_u):
        self.loss_params = {
            "hist_encoded": hist_encoded,
            "hist_true_u": hist_true_u,
            "pred_encoded": pred_encoded,
            "pred_true_u": pred_true_u,
        }
        return self.net.loss(hist_encoded, hist_true_u, pred_encoded, pred_true_u)

    def sample(self, num_samples, hist_encoded, hist_true_u, pred_encoded):
        self.sample_params = {
            "num_samples": num_samples,
            "hist_encoded": hist_encoded,
            "hist_true_u": hist_true_u,
            "pred_encoded": pred_encoded,
        }
        return self.net.sample(num_samples, hist_encoded, hist_true_u, pred_encoded)

    def forward_logdet(self, context, x):
        self.forward_logdet_params = {
            "context": context,
            "x": x,
        }
        return self.net.forward_logdet(context, x)

    def forward_no_logdet(self, context, x):
        self.forward_no_logdet_params = {
            "context": context,
            "x": x,
        }
        return self.net.forward_no_logdet(context, x)

    def inverse(self, context, u):
        self.inverse_params = {
            "context": context,
            "u": u,
        }
        return self.net.inverse(context, u)


def test_masking_loss():
    """
    Test that the correct portion of the embedding is tagged as being part of the history or as being part of the prediction,
    when computing the loss in the decoder.
    """
    num_batches = 3
    num_series = 4
    num_hist_time = 2
    num_pred_time = 3
    embed_dim = 6

    net = CopulaDecoder(
        input_dim=embed_dim,
        trivial_copula={},
        dsf_marginal={
            "mlp_layers": 1,
            "mlp_dim": 1,
            "flow_layers": 1,
            "flow_hid_dim": 1,
        },
    )

    true_value = torch.cat(
        [
            2 * torch.ones(num_batches, num_series, num_hist_time),
            3 * torch.ones(num_batches, num_series, num_pred_time),
        ],
        dim=2,
    )
    mask = true_value == 2
    encoded = torch.cat(
        [
            4 * torch.ones(num_batches, num_series, num_hist_time, embed_dim),
            5 * torch.ones(num_batches, num_series, num_pred_time, embed_dim),
        ],
        dim=2,
    )

    net.copula = __ParameterSavingShell(net.copula)

    loss = net.loss(
        encoded=encoded,
        mask=mask,
        true_value=true_value,
    )

    hist_encoded = net.copula.loss_params["hist_encoded"]
    hist_true_u = net.copula.loss_params["hist_true_u"]
    pred_encoded = net.copula.loss_params["pred_encoded"]
    pred_true_u = net.copula.loss_params["pred_true_u"]

    # Cannot test the true_u values due to being transformed by the randomized marginal module
    assert (hist_encoded == 4).all()
    assert hist_encoded.shape == (num_batches, num_series * num_hist_time, embed_dim)
    assert hist_true_u.shape == (num_batches, num_series * num_hist_time)
    assert (pred_encoded == 5).all()
    assert pred_encoded.shape == (num_batches, num_series * num_pred_time, embed_dim)
    assert pred_true_u.shape == (num_batches, num_series * num_pred_time)

    assert loss.shape == (num_batches,)


def test_decoder_sample():
    """
    Test that the output of the sampling has the correct dimensions and historical values.
    """
    num_batches = 3
    num_series = 4
    num_hist_time = 2
    num_pred_time = 3
    embed_dim = 6
    num_samples = 7

    net = CopulaDecoder(
        input_dim=embed_dim,
        trivial_copula={},
        dsf_marginal={
            "mlp_layers": 1,
            "mlp_dim": 1,
            "flow_layers": 1,
            "flow_hid_dim": 1,
        },
    )

    true_value = torch.cat(
        [
            2 * torch.ones(num_batches, num_series, num_hist_time),
            3 * torch.ones(num_batches, num_series, num_pred_time),
        ],
        dim=2,
    )
    mask = true_value == 2
    encoded = torch.cat(
        [
            4 * torch.ones(num_batches, num_series, num_hist_time, embed_dim),
            5 * torch.ones(num_batches, num_series, num_pred_time, embed_dim),
        ],
        dim=2,
    )

    samples = net.sample(
        num_samples=num_samples,
        encoded=encoded,
        mask=mask,
        true_value=true_value,
    )

    assert samples.shape == (num_batches, num_series, num_hist_time + num_pred_time, num_samples)
    assert (samples[:, :, :num_hist_time, :] == 2).all()


def test_decoder_sample_scaling():
    """
    Test that the input of the marginal is properly scaled when the min_u and max_u parameters are set.
    """
    num_batches = 3
    num_series = 4
    num_hist_time = 2
    num_pred_time = 3
    embed_dim = 6
    num_samples = 7
    min_u = 0.2
    max_u = 0.6

    net = CopulaDecoder(
        input_dim=embed_dim,
        min_u=min_u,
        max_u=max_u,
        trivial_copula={},
        dsf_marginal={
            "mlp_layers": 1,
            "mlp_dim": 1,
            "flow_layers": 1,
            "flow_hid_dim": 1,
        },
    )

    true_value = torch.cat(
        [
            2 * torch.ones(num_batches, num_series, num_hist_time),
            3 * torch.ones(num_batches, num_series, num_pred_time),
        ],
        dim=2,
    )
    mask = true_value == 2
    encoded = torch.cat(
        [
            4 * torch.ones(num_batches, num_series, num_hist_time, embed_dim),
            5 * torch.ones(num_batches, num_series, num_pred_time, embed_dim),
        ],
        dim=2,
    )

    net.marginal = __ParameterSavingShell(net.marginal)

    _ = net.sample(
        num_samples=num_samples,
        encoded=encoded,
        mask=mask,
        true_value=true_value,
    )

    u = net.marginal.inverse_params["u"]
    assert u.min() >= min_u
    assert u.max() <= max_u


def test_attentional_copula_loss():
    """
    Test that the attentional copula loss function runs without error.
    We cannot direcly test its accuracy due to the model complexity.
    """
    num_batches = 4
    num_var_hist = 7
    num_var_pred = 8
    embed_dim = 5

    net = AttentionalCopula(
        input_dim=embed_dim,
        attention_heads=3,
        attention_layers=2,
        attention_dim=13,
        mlp_layers=4,
        mlp_dim=9,
        resolution=16,
        dropout=0.2,
        fixed_permutation=False,
    )

    hist_encoded = torch.randn(num_batches, num_var_hist, embed_dim)
    hist_true_u = torch.randn(num_batches, num_var_hist)
    pred_encoded = torch.randn(num_batches, num_var_pred, embed_dim)
    pred_true_u = torch.randn(num_batches, num_var_pred)

    loss = net.loss(
        hist_encoded=hist_encoded, hist_true_u=hist_true_u, pred_encoded=pred_encoded, pred_true_u=pred_true_u
    )

    assert loss.shape == (num_batches,)


def test_attentional_copula_sample():
    """
    Test that the attentional copula sampling method runs without error.
    We cannot direcly test its accuracy due to the model complexity.
    """
    num_batches = 4
    num_var_hist = 7
    num_var_pred = 8
    embed_dim = 5
    num_samples = 9

    net = AttentionalCopula(
        input_dim=embed_dim,
        attention_heads=3,
        attention_layers=2,
        attention_dim=13,
        mlp_layers=4,
        mlp_dim=9,
        resolution=16,
        dropout=0.2,
        fixed_permutation=False,
    )

    hist_encoded = torch.randn(num_batches, num_var_hist, embed_dim)
    hist_true_u = torch.randn(num_batches, num_var_hist)
    pred_encoded = torch.randn(num_batches, num_var_pred, embed_dim)

    samples = net.sample(
        num_samples=num_samples, hist_encoded=hist_encoded, hist_true_u=hist_true_u, pred_encoded=pred_encoded
    )

    assert samples.shape == (num_batches, num_var_pred, num_samples)
    assert (samples >= 0).all()
    assert (samples <= 1).all()


def test_trivial_copula_loss():
    """
    Test that the trivial copula loss is equal to zero, and has the correct output dimensions.
    """
    net = TrivialCopula()

    num_batches = 4
    num_var_hist = 7
    num_var_pred = 8
    embed_dim = 5

    hist_encoded = torch.randn(num_batches, num_var_hist, embed_dim)
    hist_true_u = torch.randn(num_batches, num_var_hist)
    pred_encoded = torch.randn(num_batches, num_var_pred, embed_dim)
    pred_true_u = torch.randn(num_batches, num_var_pred)

    loss = net.loss(
        hist_encoded=hist_encoded, hist_true_u=hist_true_u, pred_encoded=pred_encoded, pred_true_u=pred_true_u
    )

    assert loss.shape == (num_batches,)
    assert (loss == 0).all()


def test_trivial_copula_sample():
    """
    Test that the trivial copula sampling is bound by 0 and 1, and has the correct output dimensions.
    """
    net = TrivialCopula()

    num_batches = 4
    num_var_hist = 7
    num_var_pred = 8
    embed_dim = 5
    num_samples = 9

    hist_encoded = torch.randn(num_batches, num_var_hist, embed_dim)
    hist_true_u = torch.randn(num_batches, num_var_hist)
    pred_encoded = torch.randn(num_batches, num_var_pred, embed_dim)

    samples = net.sample(
        num_samples=num_samples, hist_encoded=hist_encoded, hist_true_u=hist_true_u, pred_encoded=pred_encoded
    )

    assert samples.shape == (num_batches, num_var_pred, num_samples)
    assert (samples >= 0).all()
    assert (samples <= 1).all()


def test_gaussian_loss():
    """
    Test that the gaussian decoder loss function method runs without error.
    We cannot direcly test its accuracy due to the model complexity.
    """
    num_batches = 3
    num_series = 4
    num_hist_time = 2
    num_pred_time = 3
    embed_dim = 6

    net = GaussianDecoder(
        input_dim=embed_dim,
        matrix_rank=5,
        mlp_layers=3,
        mlp_dim=11,
    )

    true_value = torch.cat(
        [
            2 * torch.ones(num_batches, num_series, num_hist_time),
            3 * torch.ones(num_batches, num_series, num_pred_time),
        ],
        dim=2,
    )
    mask = true_value == 2
    encoded = torch.cat(
        [
            4 * torch.ones(num_batches, num_series, num_hist_time, embed_dim),
            5 * torch.ones(num_batches, num_series, num_pred_time, embed_dim),
        ],
        dim=2,
    )

    loss = net.loss(
        encoded=encoded,
        mask=mask,
        true_value=true_value,
    )

    assert loss.shape == (num_batches,)


def test_gaussian_sample():
    """
    Test that the gaussian decoder sampling method runs without error.
    We cannot direcly test its accuracy due to the model complexity.
    """
    num_batches = 3
    num_series = 4
    num_hist_time = 2
    num_pred_time = 3
    embed_dim = 6
    num_samples = 7

    net = GaussianDecoder(
        input_dim=embed_dim,
        matrix_rank=5,
        mlp_layers=3,
        mlp_dim=11,
    )

    true_value = torch.cat(
        [
            2 * torch.ones(num_batches, num_series, num_hist_time),
            3 * torch.ones(num_batches, num_series, num_pred_time),
        ],
        dim=2,
    )
    mask = true_value == 2
    encoded = torch.cat(
        [
            4 * torch.ones(num_batches, num_series, num_hist_time, embed_dim),
            5 * torch.ones(num_batches, num_series, num_pred_time, embed_dim),
        ],
        dim=2,
    )

    samples = net.sample(
        num_samples=num_samples,
        encoded=encoded,
        mask=mask,
        true_value=true_value,
    )

    assert samples.shape == (num_batches, num_series, num_hist_time + num_pred_time, num_samples)
    assert (samples[:, :, :num_hist_time, :] == 2).all()
