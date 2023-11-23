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

from typing import Any, Dict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gluonts.dataset.field_names import FieldName
from gluonts.torch.util import copy_parameters
from gluonts.env import env
from gluonts.dataset.common import Dataset
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import SelectFields, Transformation
from gluonts.itertools import maybe_len
from gluonts.transform import (
    AddObservedValuesIndicator,
    CDFtoGaussianTransform,
    Chain,
    InstanceSampler,
    InstanceSplitter,
    RenameFields,
    ValidationSplitSampler,
    TestSplitSampler,
    cdf_to_gaussian_forward_transform,
)
from pts import Trainer
from pts.model import PyTorchEstimator
from pts.model.estimator import TrainOutput
from pts.model.utils import get_module_forward_input_names
from pts.dataset.loader import TransformedDataset, TransformedIterableDataset

from .network import (
    TACTiSPredictionNetwork,
    TACTiSTrainingNetwork,
    TACTiSPredictionNetworkInterpolation,
)

from tactis.gluon.metrics import compute_validation_metrics, SplitValidationTransform
from gluonts.torch.batchify import batchify


class SingleInstanceSampler(InstanceSampler):
    """
    Randomly pick a single valid window in the given time series.
    This fix the bias in ExpectedNumInstanceSampler which leads to varying sampling frequency
    of time series of unequal length, not only based on their length, but when they were sampled.
    """

    """End index of the history"""

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        window_size = b - a + 1

        if window_size <= 0:
            return np.array([], dtype=int)

        indices = np.random.randint(window_size, size=1)

        return indices + a


class TACTiSEstimator(PyTorchEstimator):
    """
    The compatibility layer between TACTiS and GluonTS / PyTorchTS.
    """

    def __init__(
        self,
        model_parameters: Dict[str, Any],
        num_series: int,
        history_length: int,
        prediction_length: int,
        freq: str,
        trainer: Trainer,
        cdf_normalization: bool = False,
        num_parallel_samples: int = 1,
    ):
        """
        A PytorchTS wrapper for TACTiS

        Parameters:
        -----------
        model_parameters: Dict[str, Any]
            The parameters that will be sent to the TACTiS model.
        num_series: int
            The number of series in the multivariate data.
        history_length: int
            How many time steps will be sent to the model as observed.
        prediction_length: int
            How many time steps will be sent to the model as unobserved, to be predicted.
        freq: str
            The frequency of the series to be forecasted.
        trainer: Trainer
            A Pytorch-TS trainer object
        cdf_normalization: bool, default to False
            If set to True, then the data will be transformed using an estimated CDF from the
            historical data points, followed by the inverse CDF of a Normal(0, 1) distribution.
            Should not be used concurrently with the standardization normalization option in TACTiS.
        num_parallel_samples: int, default to 1
            How many samples to draw at the same time during forecast.
        """
        super().__init__(trainer=trainer)

        self.model_parameters = model_parameters

        self.num_series = num_series
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.freq = freq

        self.cdf_normalization = cdf_normalization
        self.num_parallel_samples = num_parallel_samples

    def create_training_network(self, device: torch.device) -> nn.Module:
        """
        Create the encapsulated TACTiS model which can be used for training.

        Parameters:
        -----------
        device: torch.device
            The device where the model parameters should be placed.

        Returns:
        --------
        model: nn.Module
            An instance of TACTiSTrainingNetwork.
        """
        return TACTiSTrainingNetwork(
            num_series=self.num_series,
            model_parameters=self.model_parameters,
        ).to(device=device)

    def create_instance_splitter(self, mode: str) -> Transformation:
        """
        Create and return the instance splitter needed for training, validation or testing.

        Parameters:
        -----------
        mode: str, "training", "validation", or "test"
            Whether to split the data for training, validation, or test (forecast)

        Returns
        -------
        Transformation
            The InstanceSplitter that will be applied entry-wise to datasets,
            at training, validation and inference time based on mode.
        """
        assert mode in ["training", "validation", "test"]

        if mode == "training":
            instance_sampler = SingleInstanceSampler(
                min_past=self.history_length,  # Will not pick incomplete sequences
                min_future=self.prediction_length,
            )
        elif mode == "validation":
            instance_sampler = ValidationSplitSampler(
                min_past=self.history_length,  # Will not pick incomplete sequences
                min_future=self.prediction_length,
            )
        elif mode == "test":
            # This splitter takes the last valid window from each multivariate series,
            # so any multi-window split must be done in the data definition.
            instance_sampler = TestSplitSampler()

        if self.cdf_normalization:
            normalize_transform = CDFtoGaussianTransform(
                cdf_suffix="_norm",
                target_field=FieldName.TARGET,
                target_dim=self.num_series,
                max_context_length=self.history_length,
                observed_values_field=FieldName.OBSERVED_VALUES,
            )
        else:
            normalize_transform = RenameFields(
                {
                    f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_norm",
                    f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_norm",
                }
            )

        instance_sampler = (
            InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=instance_sampler,
                past_length=self.history_length,
                future_length=self.prediction_length,
                time_series_fields=[FieldName.OBSERVED_VALUES],
            )
            + normalize_transform
        )

        return instance_sampler

    def create_transformation(self) -> Transformation:
        """
        Add a transformation that replaces NaN in the input data with zeros,
        and mention whether the data was a NaN or not in another field.

        Returns:
        --------
        transformation: Transformation
            The chain of transformations defined for TACTiS.
        """
        return Chain(
            [
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
            ]
        )

    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: nn.Module,
        device: torch.device,
        experiment_mode: str = "forecasting",
        history_length: int = -1,
    ) -> PyTorchPredictor:
        """
        Create the predictor which can be used by GluonTS to do inference.

        Parameters:
        -----------
        transformation: Transformation
            The transformation to apply to the data prior to being sent to the model.
        trained_network: nn.Module
            An instance of TACTiSTrainingNetwork with trained parameters.
        device: torch.device
            The device where the model parameters should be placed.

        Returns:
        --------
        predictor: PyTorchPredictor
            The PyTorchTS predictor object.
        """
        if experiment_mode == "forecasting":
            prediction_network = TACTiSPredictionNetwork(
                num_series=self.num_series,
                model_parameters=self.model_parameters,
                prediction_length=self.prediction_length,
                num_parallel_samples=self.num_parallel_samples,
            ).to(device=device)
        else:
            prediction_network = TACTiSPredictionNetworkInterpolation(
                num_series=self.num_series,
                model_parameters=self.model_parameters,
                prediction_length=self.prediction_length,
                history_length=history_length,
                num_parallel_samples=self.num_parallel_samples,
            ).to(device=device)
        copy_parameters(trained_network, prediction_network)

        output_transform = cdf_to_gaussian_forward_transform if self.cdf_normalization else None
        input_names = get_module_forward_input_names(prediction_network)
        prediction_splitter = self.create_instance_splitter("test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            output_transform=output_transform,
            input_names=input_names,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            device=device,
        )

    def train_model(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        optimizer: str = "adam",
        backtesting=False,
        **kwargs,
    ) -> TrainOutput:
        transformation = self.create_transformation()

        trained_net = self.create_training_network(self.trainer.device)

        input_names = get_module_forward_input_names(trained_net)

        with env._let(max_idle_transforms=maybe_len(training_data) or 0):
            training_instance_splitter = self.create_instance_splitter("training")

        training_iter_dataset = TransformedIterableDataset(
            dataset=training_data,
            transform=transformation + training_instance_splitter + SelectFields(input_names),
            is_train=True,
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
        )

        training_data_loader = DataLoader(
            training_iter_dataset,
            batch_size=self.trainer.batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn,
            **kwargs,
        )

        validation_instance_splitter = self.create_instance_splitter("validation")

        input_transform = transformation + validation_instance_splitter + SelectFields(input_names)
        if not backtesting:
            validation_iter_dataset = TransformedDataset(
                validation_data,
                transformation=SplitValidationTransform(self.history_length + self.prediction_length),
            )
        else:
            validation_iter_dataset = validation_data
        validation_iter_args = {
            "dataset": validation_iter_dataset,
            "transform": input_transform,
            "stack_fn": lambda data: batchify(data, self.trainer.device),
        }

        self.trainer(
            net=trained_net,
            train_iter=training_data_loader,
            validation_iter_args=validation_iter_args,
            optimizer=optimizer,
        )

        return trained_net

    def train(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        backtesting: bool = False,
        **kwargs,
    ) -> PyTorchPredictor:
        train_model_output = self.train_model(
            training_data,
            validation_data,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
            backtesting=backtesting,
            **kwargs,
        )
        return train_model_output

    def validate_model(
        self,
        validation_data: Optional[Dataset] = None,
        backtesting=False,
    ):
        transformation = self.create_transformation()

        trained_net = self.create_training_network(self.trainer.device)

        input_names = get_module_forward_input_names(trained_net)

        validation_instance_splitter = self.create_instance_splitter("validation")

        input_transform = transformation + validation_instance_splitter + SelectFields(input_names)
        if not backtesting:
            validation_iter_dataset = TransformedDataset(
                validation_data,
                transformation=SplitValidationTransform(self.history_length + self.prediction_length),
            )
        else:
            validation_iter_dataset = validation_data
        validation_iter_args = {
            "dataset": validation_iter_dataset,
            "transform": input_transform,
            "stack_fn": lambda data: batchify(data, self.trainer.device),
        }

        nll = self.trainer.validate(
            net=trained_net,
            validation_iter_args=validation_iter_args,
        )

        return nll

    def validate(
        self,
        validation_data: Optional[Dataset] = None,
        backtesting: bool = False,
    ):
        return self.validate_model(
            validation_data,
            backtesting=backtesting,
        )
