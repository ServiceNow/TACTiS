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

from tactis.model.decoder import TrivialCopula

import torch

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

    loss = net.loss(hist_encoded=hist_encoded, hist_true_u=hist_true_u, pred_encoded=pred_encoded, pred_true_u=pred_true_u)

    assert loss.shape == (num_batches, )
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

    samples = net.sample(num_samples=num_samples, hist_encoded=hist_encoded, hist_true_u=hist_true_u, pred_encoded=pred_encoded)

    assert samples.shape == (num_batches, num_var_pred, num_samples)
    assert (samples >= 0).all()
    assert (samples <= 1).all()
