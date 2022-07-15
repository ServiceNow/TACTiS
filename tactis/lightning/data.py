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

>> Contains the DataModule allowing Lightning to handle multivariate time-series data.
"""

from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import random


class TimeseriesDataset(Dataset):
    """
    A Dataset for a multivariate time series.
    It will split the data into historical and prediction values,
    and assign time values to each time step.
    """

    def __init__(self, data: List[np.array], hist_length: int, pred_length: int):
        """
        Parameters:
        -----------
        data: List[np.array]
            The data for the dataset, as a list of aligned 1d series.
        hist_length: int
            When doing the forecast, how many time steps will be available.
        pred_length: int
            When doing the forecast, the length of said forecast.
        """
        self.data = torch.Tensor(data)
        self.hist_length = hist_length
        self.pred_length = pred_length

    def __len__(self):
        return self.data.shape[1] - (self.hist_length + self.pred_length - 1)

    def __getitem__(self, index):
        hist_time = torch.arange(0, self.hist_length)
        hist_value = self.data[:, index : index + self.hist_length]
        pred_time = torch.arange(self.hist_length, self.hist_length + self.pred_length)
        pred_value = self.data[:, index + self.hist_length : index + self.hist_length + self.pred_length]
        return hist_time, hist_value, pred_time, pred_value


class RandomTimeseriesSampler(Sampler):
    """
    A sampler which randomly pick the given number of samples at each epoch.
    The picking is done without replacement, so a given sample cannot be picked twice in a single epoch.
    """

    def __init__(self, start, end, num_samples):
        super().__init__(data_source=None)

        self.start = start
        self.end = end
        self.num_samples = num_samples

    def __iter__(self):
        yield from random.sample(range(self.start, self.end), k=self.num_samples)


class TimeseriesDataModule(pl.LightningDataModule):
    """
    The Lightning data interface for a multivariate time series.
    """

    def __init__(
        self,
        data: List[np.array],
        train_last_timestep: int,
        val_last_timestep: int,
        test_delta_timestep: int,
        batch_size: int,
        pred_length: int,
        hist_length_ratio: float,
        train_epoch_length: int,
        val_epoch_length: int,
    ):
        """
        Parameters:
        -----------
        data: List[np.array]
            The raw time series data.
        train_last_timestep: int
            Index of the time step at which the data goes from being training data to being validation data.
        val_last_timestep: int
            Index of the time step at which the data goes from being validation data to being testing data.
        test_delta_timestep: int
            In the testing data, each testing sample will be separated by that many time steps.
        batch_size: int
            The batch size when batching samples together.
        pred_length: int
            When doing the forecast, the length of said forecast.
        hist_length_ratio: float
            When doing the forecast, how many time steps will be available, as a multiple of pred_length.
        train_epoch_length: int
            How many samples per epoch when training.
        val_epoch_length: int
            How many samples per epoch when validating.
        """
        super().__init__()

        self.data = data

        self.train_last_timestep = train_last_timestep
        self.val_last_timestep = val_last_timestep
        self.test_delta_timestep = test_delta_timestep
        self.train_epoch_length = train_epoch_length
        self.val_epoch_length = val_epoch_length

        self.batch_size = batch_size
        self.pred_length = pred_length
        self.hist_length = int(pred_length * hist_length_ratio)

        self.save_hyperparameters("hist_length_ratio", "pred_length")

    def setup(self, stage=None):
        self.full_dataset = TimeseriesDataset(self.data, hist_length=self.hist_length, pred_length=self.pred_length)

    def train_dataloader(self):
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            sampler=RandomTimeseriesSampler(
                0, self.train_last_timestep - self.hist_length + 1, num_samples=self.train_epoch_length
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            sampler=RandomTimeseriesSampler(
                self.train_last_timestep - self.hist_length + 1,
                self.val_last_timestep - self.hist_length + 1,
                num_samples=self.val_epoch_length,
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            sampler=range(
                self.val_last_timestep - self.hist_length + 1, len(self.full_dataset), self.test_delta_timestep
            ),
        )

    def predict_groundtruth(self, include_hist=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the target ground truth associated with the forecasts which will be generated when doing the predict step.

        Parameters:
        -----------
        include_hist: bool, default to False
            If set to True, concatenate the historical data to the ground truth.

        Returns:
        --------
            values: torch.Tensor [samples, series, time steps]
                The ground truth values for the various prediction samples.
            times: torch.Tensor [samples, time steps]
                The time stamps associated with the ground truth values.
        """
        times = []
        values = []
        for target in range(
            self.val_last_timestep - self.hist_length + 1, len(self.full_dataset), self.test_delta_timestep
        ):
            hist_time, hist_value, pred_time, pred_value = self.full_dataset[target]
            if include_hist:
                times.append(torch.cat([hist_time, pred_time], dim=0))
                values.append(torch.cat([hist_value, pred_value], dim=1))
            else:
                times.append(pred_time)
                values.append(pred_value)
        return torch.stack(times, dim=0), torch.stack(values, dim=0)

    def predict_dataloader(self):
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            sampler=range(
                self.val_last_timestep - self.hist_length + 1, len(self.full_dataset), self.test_delta_timestep
            ),
        )
