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
from torch import nn

from tactis.model.encoder import Encoder, TemporalEncoder


def test_encoder():
    """
    Test that the Encoder class runs without error, with the correct dimensions.
    """
    num_batches = 4
    num_series = 5
    num_timesteps = 6

    attention_heads = 2
    attention_dim = 7

    net = Encoder(
        attention_layers=3,
        attention_heads=attention_heads,
        attention_dim=attention_dim,
        attention_feedforward_dim=11,
    )

    assert net.embedding_dim == attention_heads * attention_dim

    input = torch.rand(num_batches, num_series, num_timesteps, net.embedding_dim)
    output = net.forward(input)

    assert output.shape == (num_batches, num_series, num_timesteps, net.embedding_dim)


def test_temporal_encoder():
    """
    Test that the TemporalEncoder class runs without error, with the correct dimensions.
    """
    num_batches = 4
    num_series = 5
    num_timesteps = 6

    attention_heads = 2
    attention_dim = 7

    net = TemporalEncoder(
        attention_layers=3,
        attention_heads=attention_heads,
        attention_dim=attention_dim,
        attention_feedforward_dim=11,
    )

    assert net.embedding_dim == attention_heads * attention_dim

    input = torch.rand(num_batches, num_series, num_timesteps, net.embedding_dim)
    output = net.forward(input)

    assert output.shape == (num_batches, num_series, num_timesteps, net.embedding_dim)
