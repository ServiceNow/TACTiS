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

>> The interface between the TACTiS model and PyTorch Lightning.
"""

import pytorch_lightning as pl
from typing import Dict, Any
import torch

from ..model.tactis import TACTiS


class TACTiSLightning(pl.LightningModule):
    """
    Encapsulate the TACTiS model inside a Lightning Module shell.
    """

    def __init__(self, num_series: int, num_samples: int, model_parameters: Dict[str, Any], learning_rate: float):
        """
        Parameters:
        -----------
        num_series: int
            The number of independent series in the dataset the model will learn from.
        num_samples: int
            When forecasting, how many independent samples to generate for each time point.
        model_parameters: Dict[str, Any]
            The parameters to be sent to the TACTiS model.
            These will be logged as being the hyperparameters in the log.
        learning_rate: float
            The learning rate for the Adam optimizer.
        """
        super().__init__()

        self.num_samples = num_samples
        self.net = TACTiS(num_series=num_series, **model_parameters)
        self.learning_rate = learning_rate
        self.save_hyperparameters("model_parameters")
        self.save_hyperparameters("learning_rate")

    def forward(self, hist_time: torch.Tensor, hist_value: torch.Tensor, pred_time: torch.Tensor) -> torch.Tensor:
        """
        Forecast the possible values for the series, at the given prediction time points.

        Parameters:
        -----------
        hist_time: Tensor [batch, time steps]
            A tensor containing the time steps associated with the values of hist_value.
        hist_value: Tensor [batch, series, time steps]
            A tensor containing the values that will be available at inference time.
        pred_time: Tensor [batch, time steps]
            A tensor containing the time steps associated with the values of pred_value.

        Returns:
        --------
        samples: torch.Tensor [batch, series, time steps, samples]
            Samples from the forecasted distribution.
        """
        return self.net.sample(self.num_samples, hist_time, hist_value, pred_time)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Compute the loss function on a training batch.
        """
        hist_time, hist_value, pred_time, pred_value = batch
        loss = self.net.loss(hist_time, hist_value, pred_time, pred_value)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Compute the loss function on a validation batch.
        """
        hist_time, hist_value, pred_time, pred_value = batch
        loss = self.net.loss(hist_time, hist_value, pred_time, pred_value)
        self.log("valid_loss", loss)
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Compute metrics on a test batch.
        Currently the only metric is the loss function.
        """
        hist_time, hist_value, pred_time, pred_value = batch
        loss = self.net.loss(hist_time, hist_value, pred_time, pred_value)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Forecast possible values for the series, over a test batch.
        """
        hist_time, hist_value, pred_time, _ = batch
        return self.forward(hist_time, hist_value, pred_time)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Create the optimizer for the model.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
