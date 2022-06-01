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

from tactis.model.marginal import DSFMarginal

import torch


def test_monotonicity():
    """
    Verify that the DSF is monotonic.
    We don't test that it is strictly monotonic in case of numerical precision issues.
    """
    net = DSFMarginal(
        context_dim=3,
        mlp_layers=2,
        mlp_dim=4,
        flow_layers=2,
        flow_hid_dim=5,
    )

    num_batches = 2
    num_variables = 3
    context = torch.randn(num_batches, num_variables, net.context_dim)
    x = torch.arange(-10, 10, 0.1).repeat(num_batches, num_variables, 1)
    
    u = net.forward_no_logdet(context=context, x=x)

    diff = u[:, :, 1:] - u[:, :, :-1]
    assert (diff >= 0).all()


def test_same_forward():
    """
    Verify that both DSF forward methods give the same results.
    """
    net = DSFMarginal(
        context_dim=3,
        mlp_layers=2,
        mlp_dim=4,
        flow_layers=2,
        flow_hid_dim=5,
    )

    num_batches = 2
    num_variables = 3
    context = torch.randn(num_batches, num_variables, net.context_dim)
    x = torch.arange(-10, 10, 0.1).repeat(num_batches, num_variables, 1)
    
    u_no_logdet = net.forward_no_logdet(context=context, x=x)
    u_logdet, _ = net.forward_logdet(context=context, x=x)

    assert torch.isclose(u_no_logdet, u_logdet).all()


def test_bounds_at_inf():
    """
    Verify that at +/- infinity, the DSF goes to 1 or 0.
    """
    net = DSFMarginal(
        context_dim=3,
        mlp_layers=2,
        mlp_dim=4,
        flow_layers=2,
        flow_hid_dim=5,
    )

    num_batches = 2
    num_variables = 3
    context = torch.randn(num_batches, num_variables, net.context_dim)

    x_pos = float("inf") * torch.ones(num_batches, num_variables)
    x_neg = -float("inf") * torch.ones(num_batches, num_variables)

    u_pos = net.forward_no_logdet(context=context, x=x_pos)
    u_neg = net.forward_no_logdet(context=context, x=x_neg)

    u_pos_min = u_pos.min()
    u_pos_max = u_pos.max()
    u_neg_min = u_neg.min()
    u_neg_max = u_neg.max()

    assert u_pos_min >= 1 - 1e-2
    assert u_pos_max <= 1
    assert u_neg_min >= 0
    assert u_neg_max <= 1e-2

def test_inverse_monotonicity():
    """
    Test that the DSF inverse function is monotonic.
    """

    net = DSFMarginal(
        context_dim=3,
        mlp_layers=2,
        mlp_dim=4,
        flow_layers=2,
        flow_hid_dim=5,
    )

    num_batches = 2
    num_variables = 3
    context = torch.randn(num_batches, num_variables, net.context_dim)
    u = torch.arange(1e-3, 1-1e-3, 1e-3).repeat(num_batches, num_variables, 1)
    
    x = net.inverse(context=context, u=u)

    diff = x[:, :, 1:] - x[:, :, :-1]
    assert (diff >= 0).all()


def test_inverse_forward():
    """
    Test that the DSF inverse function is indeed the inverse of the forward function.
    """

    net = DSFMarginal(
        context_dim=3,
        mlp_layers=2,
        mlp_dim=4,
        flow_layers=2,
        flow_hid_dim=5,
    )

    num_batches = 2
    num_variables = 3
    context = torch.randn(num_batches, num_variables, net.context_dim)
    # Avoid the extremes due to the maximum x values in the inverse function
    u = torch.arange(1e-1, 1-1e-1, 1e-3).repeat(num_batches, num_variables, 1)
    
    x = net.inverse(context=context, u=u, precision=1e-6)
    u_after = net.forward_no_logdet(context=context, x=x)

    assert torch.isclose(u, u_after, rtol=0, atol=1e-4).all()
