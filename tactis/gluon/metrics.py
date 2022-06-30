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

>> The methods to compute the metrics on the GluonTS forecast objects.
"""


import os
import pickle
import sys
from typing import Dict, Iterable, Iterator, Optional

import numpy as np
import pandas as pd
from gluonts import transform
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.evaluation import MultivariateEvaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.forecast import Forecast
from gluonts.model.predictor import Predictor


class SplitValidationTransform(transform.FlatMapTransformation):
    """
    Split a dataset to do validation tests ending at each possible time step.
    A time step is possible if the resulting series is at least as long as the window_length parameter.
    """

    def __init__(self, window_length: int):
        super().__init__()
        self.window_length = window_length

    def flatmap_transform(self, data: DataEntry, is_train: bool) -> Iterator[DataEntry]:
        full_length = data["target"].shape[-1]
        for end_point in range(self.window_length, full_length):
            data_copy = data.copy()
            data_copy["target"] = data["target"][..., :end_point]
            yield data_copy


class SuppressOutput:
    """
    Context controller to remove any printing to standard output and standard error.
    Inspired from:
    https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    """

    def __enter__(self):
        self._stdout_bkp = sys.stdout
        self._stderr_bkp = sys.stderr
        sys.stdout = sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._stdout_bkp
        sys.stderr = self._stderr_bkp


def _compute_energy_score(target_data: np.array, samples: np.array, num_samples: int, beta: float) -> np.float32:
    """
    Compute the unnormalized energy score for a single forecast.

    Parameters:
    -----------
    target_data: np.array [two dimensions]
        The ground truth.
    samples: np.array [number of samples, two additional dimensions]
        The samples from the forecasting method to be assessed.
    num_samples: int
        The number of samples from the forecast.
    beta: float
        The beta parameter for the energy score.

    Returns
    -------
    score: np.float32
        The energy score.
    """
    # The Frobenius norm of a matrix is equal to the Euclidean norm of its element:
    # the square root of the sum of the square of its elements
    norm = np.linalg.norm(samples - target_data[None, :, :], ord="fro", axis=(1, 2))
    first_term = (norm**beta).mean()

    # For the second term of the energy score, we need two independant realizations of the distributions.
    # So we do a sum ignoring the i == j terms (which would be zero anyway), and we normalize by the
    # number of pairs of samples we have summed over.
    s = np.float32(0)
    for i in range(num_samples - 1):
        for j in range(i + 1, num_samples):
            norm = np.linalg.norm(samples[i] - samples[j], ord="fro")
            s += norm**beta
    second_term = s / (num_samples * (num_samples - 1) / 2)

    return first_term - 0.5 * second_term


def compute_energy_score(
    targets: Iterable[pd.DataFrame], forecasts: Iterable[Forecast], beta: float = 1.0
) -> np.float32:
    """
    Compute the non-normalized energy score for a multivariate stochastic prediction from samples.

    Parameters:
    -----------
    targets: Iterable[pd.DataFrame]
        The observed values, containing both the history and the prediction windows.
        Each element is taken independantly, and the result averaged over them.
    dataset: Iterable[Forecast]
        An object containing multiple samples of the probabilistic forecasts.
        This iterable should have the exact same length as targets.
    beta: float, default to 1.
        The energy score parameter, must be between 0 and 2, exclusive.

    Returns:
    --------
    result: np.float32
        A dictionary containing the various metrics
    """
    assert 0 < beta < 2

    cumulative_score = np.float32(0)
    num_forecasts = 0
    for target, forecast in zip(targets, forecasts):
        # The targets should always end with the prediction window
        assert target.index[-forecast.prediction_length] == forecast.start_date
        target_data = target.iloc[-forecast.prediction_length :].to_numpy()
        samples = forecast.samples

        cumulative_score += _compute_energy_score(target_data, samples, forecast.num_samples, beta)
        num_forecasts += 1
    return cumulative_score / num_forecasts


def compute_validation_metrics(
    predictor: Predictor,
    dataset: Dataset,
    window_length: int,
    num_samples: int,
    split: bool = True,
    savedir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute GluonTS metrics for the given predictor and dataset.

    Parameters:
    -----------
    predictor: Predictor
        The trained model to predict with.
    dataset: Dataset
        The dataset on which the model will be tested.
    window_length: int
        The prediction length + history length of the model.
    num_samples: int
        How many samples will be generated from the stochastic predictions.
    split: bool, default to True
        If set to False, the dataset is used as is, normally with one prediction per entry in the dataset.
        If set to True, the dataset is split into all possible subset, thus with one prediction per timestep in the dataset.
        Normally should be set to True during HP search, since the HP search validation dataset has only one entry;
        and set to False during backtesting, since the testing dataset has multiple entries.
    savedir: None or str, default to None
        If set, save the forecasts and the targets in a pickle file named forecasts_targets.pkl located in said location.

    Returns:
    --------
    result: Dict[str, float]
        A dictionary containing the various metrics.
    """
    if split:
        split_dataset = transform.TransformedDataset(dataset, transformation=SplitValidationTransform(window_length))
    else:
        split_dataset = dataset
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=split_dataset, predictor=predictor, num_samples=num_samples
    )
    forecasts = list(forecast_it)
    targets = list(ts_it)

    # A raw dump of the forecasts and targets for post-hoc analysis if needed, in the experiment folder
    # Can be loaded with the simple script:
    if savedir:
        savefile = os.path.join(savedir, "forecasts_targets.pkl")
        with open(savefile, "wb") as f:
            pickle.dump((forecasts, targets), f)

    # The results are going to be meaningless if any NaN shows up in the results,
    # so catch them here
    num_nan = 0
    num_inf = 0
    for forecast in forecasts:
        num_nan += np.isnan(forecast.samples).sum()
        num_inf += np.isinf(forecast.samples).sum()
    if num_nan > 0 or num_inf > 0:
        return {
            "CRPS": float("nan"),
            "ND": float("nan"),
            "NRMSE": float("nan"),
            "MSE": float("nan"),
            "CRPS-Sum": float("nan"),
            "ND-Sum": float("nan"),
            "NRMSE-Sum": float("nan"),
            "MSE-Sum": float("nan"),
            "Energy": float("nan"),
            "num_nan": num_nan,
            "num_inf": num_inf,
        }

    # Evaluate the quality of the model
    evaluator = MultivariateEvaluator(quantiles=(np.arange(20) / 20.0)[1:], target_agg_funcs={"sum": np.sum})

    # The GluonTS evaluator is very noisy on the standard error, so suppress it.
    with SuppressOutput():
        agg_metric, _ = evaluator(targets, forecasts)

    return {
        "CRPS": agg_metric.get("mean_wQuantileLoss", float("nan")),
        "ND": agg_metric.get("ND", float("nan")),
        "NRMSE": agg_metric.get("NRMSE", float("nan")),
        "MSE": agg_metric.get("MSE", float("nan")),
        "CRPS-Sum": agg_metric.get("m_sum_mean_wQuantileLoss", float("nan")),
        "ND-Sum": agg_metric.get("m_sum_ND", float("nan")),
        "NRMSE-Sum": agg_metric.get("m_sum_NRMSE", float("nan")),
        "MSE-Sum": agg_metric.get("m_sum_MSE", float("nan")),
        "Energy": compute_energy_score(targets, forecasts),
        "num_nan": num_nan,
        "num_inf": num_inf,
    }
