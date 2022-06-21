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
from tactis.model.tactis import PositionalEncoding, TACTiS


def test_positional_encoding_time_broadcast():
    """
    Verify that the broadcasting lead to the same results with singleton series dimension.
    """
    num_batches = 2
    num_series = 3
    num_timesteps = 4
    emb_dim = 6  # Must be even

    net = PositionalEncoding(embedding_dim=emb_dim, dropout=0)

    encoded = torch.zeros(num_batches, num_series, num_timesteps, emb_dim)

    timesteps_singleton = torch.arange(0, num_timesteps, dtype=int)[None, None, :].expand(num_batches, 1, -1)
    output_singleton = net.forward(encoded, timesteps_singleton)

    timesteps_expanded = torch.arange(0, num_timesteps, dtype=int)[None, None, :].expand(num_batches, num_series, -1)
    output_expanded = net.forward(encoded, timesteps_expanded)

    assert (output_singleton == output_expanded).all()


def test_positional_encoding_values():
    """
    Check some values for the positional encoding.
    """
    num_batches = 2
    num_series = 3
    num_timesteps = 4
    emb_dim = 6  # Must be even

    net = PositionalEncoding(embedding_dim=emb_dim, dropout=0)

    encoded = torch.zeros(num_batches, num_series, num_timesteps, emb_dim)

    timesteps = torch.arange(0, num_timesteps, dtype=int)[None, None, :].expand(num_batches, num_series, -1)
    output = net.forward(encoded, timesteps)

    expected_output = torch.Tensor(
        [
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [
                0.8414709848078965,
                0.5403023058681398,
                0.04639922346473126,
                0.9989229760406304,
                0.0021544330233656027,
                0.9999976792064809,
            ],
            [
                0.9092974268256817,
                -0.4161468365471424,
                0.0926985007787272,
                0.9956942241237399,
                0.004308856046742809,
                0.9999907168366957,
            ],
            [
                0.1411200080598672,
                -0.9899924966004454,
                0.13879810108005047,
                0.990320699135675,
                0.0064632590701896395,
                0.9999791129229608,
            ],
        ]
    )

    assert torch.isclose(output[0, 0, :, :], expected_output).all()


def test_positional_encoding_shift():
    """
    Check that adding per-batch constants to the time steps does not change the encoding.
    """
    num_batches = 2
    num_series = 3
    num_timesteps = 4
    emb_dim = 6  # Must be even

    net = PositionalEncoding(embedding_dim=emb_dim, dropout=0)

    encoded = torch.zeros(num_batches, num_series, num_timesteps, emb_dim)

    timesteps_base = torch.arange(0, num_timesteps, dtype=int)[None, None, :].expand(num_batches, num_series, -1)
    output_base = net.forward(encoded, timesteps_base)

    timesteps_shift = timesteps_base + torch.Tensor([17, 2345]).to(int)[:, None, None]
    output_shift = net.forward(encoded, timesteps_shift)

    assert torch.isclose(output_base, output_shift).all()


def test_bagging_shapes():
    """
    Test that the bagging procedure output tensors of the correct shapes.
    """
    num_batches = 4
    num_series = 5
    num_timesteps_hist = 2
    num_timesteps_pred = 3
    emb_dim = 6
    bagging_size = 3

    hist_time = torch.zeros(num_batches, num_series, num_timesteps_hist)
    hist_value = torch.zeros(num_batches, num_series, num_timesteps_hist)
    pred_time = torch.zeros(num_batches, num_series, num_timesteps_pred)
    pred_value = torch.zeros(num_batches, num_series, num_timesteps_pred)
    series_emb = torch.zeros(num_batches, num_series, emb_dim)

    out_hist_time, out_hist_value, out_pred_time, out_pred_value, out_series_emb = TACTiS._apply_bagging(
        bagging_size, hist_time, hist_value, pred_time, pred_value, series_emb
    )

    assert out_hist_time.shape == torch.Size([num_batches, bagging_size, num_timesteps_hist])
    assert out_hist_value.shape == torch.Size([num_batches, bagging_size, num_timesteps_hist])
    assert out_pred_time.shape == torch.Size([num_batches, bagging_size, num_timesteps_pred])
    assert out_pred_value.shape == torch.Size([num_batches, bagging_size, num_timesteps_pred])
    assert out_series_emb.shape == torch.Size([num_batches, bagging_size, emb_dim])


def test_bagging_values():
    """
    Test that the bagging procedure output tensors with the same data as the input.
    """
    num_batches = 4
    num_series = 5
    num_timesteps_hist = 2
    num_timesteps_pred = 3
    emb_dim = 6
    bagging_size = 3

    # Data which only depends on which tensor it is
    hist_time = 11 * torch.ones(num_batches, num_series, num_timesteps_hist)
    hist_value = 12 * torch.ones(num_batches, num_series, num_timesteps_hist)
    pred_time = 13 * torch.ones(num_batches, num_series, num_timesteps_pred)
    pred_value = 14 * torch.ones(num_batches, num_series, num_timesteps_pred)
    series_emb = 15 * torch.ones(num_batches, num_series, emb_dim)

    out_hist_time, out_hist_value, out_pred_time, out_pred_value, out_series_emb = TACTiS._apply_bagging(
        bagging_size, hist_time, hist_value, pred_time, pred_value, series_emb
    )

    assert (out_hist_time == 11).all()
    assert (out_hist_value == 12).all()
    assert (out_pred_time == 13).all()
    assert (out_pred_value == 14).all()
    assert (out_series_emb == 15).all()


def test_bagging_common_series():
    """
    Test that the bagging procedure keep the same series for all input.
    """
    num_batches = 4
    num_series = 5
    num_timesteps_hist = 2
    num_timesteps_pred = 3
    emb_dim = 6
    bagging_size = 3

    # Data which only depends on the series
    hist_time = torch.arange(0, num_series)[None, :, None].expand(num_batches, -1, num_timesteps_hist)
    hist_value = torch.arange(0, num_series)[None, :, None].expand(num_batches, -1, num_timesteps_hist)
    pred_time = torch.arange(0, num_series)[None, :, None].expand(num_batches, -1, num_timesteps_pred)
    pred_value = torch.arange(0, num_series)[None, :, None].expand(num_batches, -1, num_timesteps_pred)
    series_emb = torch.arange(0, num_series)[None, :, None].expand(num_batches, -1, emb_dim)

    out_hist_time, out_hist_value, out_pred_time, out_pred_value, out_series_emb = TACTiS._apply_bagging(
        bagging_size, hist_time, hist_value, pred_time, pred_value, series_emb
    )

    for batch in range(num_batches):
        for bag in range(bagging_size):
            value = out_hist_time[batch, bag, 0]
            assert (out_hist_time[batch, bag, :] == value).all()
            assert (out_hist_value[batch, bag, :] == value).all()
            assert (out_pred_time[batch, bag, :] == value).all()
            assert (out_pred_value[batch, bag, :] == value).all()
            assert (out_series_emb[batch, bag, :] == value).all()


def test_loss_function():
    """
    Test that the loss function runs without error.
    Cannot do a more precise test due to the complexity of the model.
    """
    num_batches = 4
    num_series = 5
    num_timesteps_hist = 2
    num_timesteps_pred = 3

    net = TACTiS(
        num_series=num_series,
        series_embedding_dim=7,
        input_encoder_layers=3,
        bagging_size=4,
        input_encoding_normalization=True,
        positional_encoding={
            "dropout": 0.2,
        },
        temporal_encoder={
            "attention_layers": 3,
            "attention_heads": 2,
            "attention_dim": 8,
            "attention_feedforward_dim": 10,
            "dropout": 0.2,
        },
        copula_decoder={
            "attentional_copula": {
                "attention_heads": 2,
                "attention_layers": 2,
                "attention_dim": 12,
                "mlp_layers": 3,
                "mlp_dim": 8,
                "resolution": 20,
            },
            "dsf_marginal": {
                "mlp_layers": 3,
                "mlp_dim": 6,
                "flow_layers": 2,
                "flow_hid_dim": 10,
            },
        },
    )

    hist_time = torch.rand(num_batches, num_series, num_timesteps_hist)
    hist_value = torch.rand(num_batches, num_series, num_timesteps_hist)
    pred_time = torch.rand(num_batches, num_series, num_timesteps_pred)
    pred_value = torch.rand(num_batches, num_series, num_timesteps_pred)

    loss = net.loss(hist_time, hist_value, pred_time, pred_value)

    assert loss.shape == torch.Size([num_batches])


def test_sample_function():
    """
    Test that the sample function runs without error.
    Cannot do a more precise test due to the complexity of the model.
    """
    num_batches = 4
    num_series = 5
    num_timesteps_hist = 2
    num_timesteps_pred = 3
    num_samples = 17

    net = TACTiS(
        num_series=num_series,
        series_embedding_dim=7,
        input_encoder_layers=3,
        bagging_size=4,
        input_encoding_normalization=True,
        positional_encoding={
            "dropout": 0.2,
        },
        temporal_encoder={
            "attention_layers": 3,
            "attention_heads": 2,
            "attention_dim": 8,
            "attention_feedforward_dim": 10,
            "dropout": 0.2,
        },
        copula_decoder={
            "attentional_copula": {
                "attention_heads": 2,
                "attention_layers": 2,
                "attention_dim": 12,
                "mlp_layers": 3,
                "mlp_dim": 8,
                "resolution": 20,
            },
            "dsf_marginal": {
                "mlp_layers": 3,
                "mlp_dim": 6,
                "flow_layers": 2,
                "flow_hid_dim": 10,
            },
        },
    )

    hist_time = torch.rand(num_batches, num_series, num_timesteps_hist)
    hist_value = torch.rand(num_batches, num_series, num_timesteps_hist)
    pred_time = torch.rand(num_batches, num_series, num_timesteps_pred)

    samples = net.sample(num_samples, hist_time, hist_value, pred_time)

    assert samples.shape == torch.Size([num_batches, num_series, num_timesteps_hist + num_timesteps_pred, num_samples])
