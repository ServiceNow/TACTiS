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


import os
import gc
import pickle
import sys
import time
import torch
from tqdm import tqdm
from typing import Dict, Iterable, Iterator, Optional

import numpy as np
import pandas as pd
from gluonts import transform
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.evaluation import MultivariateEvaluator
from gluonts.model.forecast import Forecast
from gluonts.model.predictor import Predictor

from tactis.gluon.backtest import make_evaluation_predictions


class SplitValidationTransform(transform.FlatMapTransformation):
    """
    Split a dataset to do validation tests ending at each possible time step.
    A time step is possible if the resulting series is at least as long as the window_length parameter.
    """

    def __init__(self, window_length: int, max_windows=None):
        super().__init__()
        self.window_length = window_length
        self.max_windows = max_windows
        self.num_windows_seen = 0

    def flatmap_transform(self, data: DataEntry, is_train: bool) -> Iterator[DataEntry]:
        full_length = data["target"].shape[-1]
        for end_point in tqdm(range(self.window_length, full_length + 1)):
            if self.max_windows and self.num_windows_seen == self.max_windows:
                break
            data_copy = data.copy()
            data_copy["target"] = data["target"][..., :end_point]
            self.num_windows_seen += 1
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


def _compute_energy_score(
    target_data: np.array, samples: np.array, num_samples: int, beta: float
) -> np.float32:
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

        cumulative_score += _compute_energy_score(
            target_data, samples, forecast.num_samples, beta
        )
        num_forecasts += 1
    return cumulative_score / num_forecasts


def compute_validation_metrics(
    predictor: Predictor,
    dataset: Dataset,
    window_length: int,
    prediction_length: int,
    num_samples: int,
    split: bool = True,
    savedir: Optional[str] = None,
    return_forecasts_and_targets: bool = False,
    subset_series=None,
    max_windows=None,
    skip_energy=True,
    n_quantiles=20,
):
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
    data_splitting_start_time = time.time()
    if split:
        split_dataset = transform.TransformedDataset(
            dataset, transformation=SplitValidationTransform(window_length, max_windows)
        )
    else:
        split_dataset = dataset
    data_splitting_end_time = time.time()
    print(
        "Metrics function: Data splitting time:",
        data_splitting_end_time - data_splitting_start_time,
    )

    while True:
        print("Batch size:", predictor.batch_size)
        try:
            predicting_start_time = time.time()
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=split_dataset, predictor=predictor, num_samples=num_samples
            )
            forecasts = list(forecast_it)
            targets = list(ts_it)
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            print(error)
            if predictor.batch_size == 1:
                print(
                    "Batch is already at the minimum. Cannot reduce further. Exiting..."
                )
                return None
            else:
                print("Caught OutOfMemoryError. Reducing batch size...")
                predictor.batch_size //= 2
                gc.collect()
                torch.cuda.empty_cache()

    if max_windows:
        targets = targets[:max_windows]

    if subset_series:
        targets = [target.iloc[:, subset_series] for target in targets]
        for target in targets:
            target.columns = list(range(len(subset_series)))

    print("#Forecasts:", len(forecasts), "#Targets:", len(targets))
    print("Shape of forecasts[0].samples", forecasts[0].samples.shape)
    predicting_end_time = time.time()
    print(
        "Metrics function: Predicting time:",
        predicting_end_time - predicting_start_time,
    )

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
        if skip_energy:
            return {
                "CRPS": float("nan"),
                "ND": float("nan"),
                "NRMSE": float("nan"),
                "MSE": float("nan"),
                "CRPS-Sum": float("nan"),
                "ND-Sum": float("nan"),
                "NRMSE-Sum": float("nan"),
                "MSE-Sum": float("nan"),
                "num_nan": num_nan,
                "num_inf": num_inf,
            }
        else:
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
    evaluator = MultivariateEvaluator(
        quantiles=(np.arange(n_quantiles) / float(n_quantiles))[1:],
        target_agg_funcs={"sum": np.sum},
    )

    # The GluonTS evaluator is very noisy on the standard error, so suppress it.
    metrics_calc_start_time = time.time()
    with SuppressOutput():
        agg_metric, ts_wise_metrics = evaluator(targets, forecasts)
    metrics_calc_end_time = time.time()
    print(
        "Metrics function: Metrics calc time:",
        metrics_calc_end_time - metrics_calc_start_time,
    )

    if skip_energy:
        metrics = {
            "CRPS": agg_metric.get("mean_wQuantileLoss", float("nan")),
            "ND": agg_metric.get("ND", float("nan")),
            "NRMSE": agg_metric.get("NRMSE", float("nan")),
            "MSE": agg_metric.get("MSE", float("nan")),
            "CRPS-Sum": agg_metric.get("m_sum_mean_wQuantileLoss", float("nan")),
            "ND-Sum": agg_metric.get("m_sum_ND", float("nan")),
            "NRMSE-Sum": agg_metric.get("m_sum_NRMSE", float("nan")),
            "MSE-Sum": agg_metric.get("m_sum_MSE", float("nan")),
            "num_nan": num_nan,
            "num_inf": num_inf,
        }
    else:
        metrics = {
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

    if return_forecasts_and_targets:
        return metrics, ts_wise_metrics, forecasts, targets
    else:
        return metrics, ts_wise_metrics


def compute_validation_metrics_interpolation(
    predictor: Predictor,
    dataset: Dataset,
    window_length: int,
    prediction_length: int,
    num_samples: int,
    split: bool = True,
    savedir: Optional[str] = None,
    return_forecasts_and_targets: bool = False,
    subset_series=None,
    max_windows=None,
    skip_energy=True,
    n_quantiles=20,
):
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
    history_length = window_length - prediction_length
    data_splitting_start_time = time.time()
    if not split:
        raise NotImplementedError(
            "Evaluating only last window is not supported for interpolation. Use --compute_validation_metrics_split."
        )
    if split:
        split_dataset = transform.TransformedDataset(
            dataset, transformation=SplitValidationTransform(window_length, max_windows)
        )
    else:
        split_dataset = dataset
    data_splitting_end_time = time.time()
    print(
        "Metrics function: Data splitting time:",
        data_splitting_end_time - data_splitting_start_time,
    )

    batch_size = predictor.batch_size
    while True:
        print("Initial batch size:", batch_size)
        predictor.batch_size = batch_size
        try:
            predicting_start_time = time.time()
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=split_dataset, predictor=predictor, num_samples=num_samples
            )
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            print(error)
            if batch_size == 1:
                print(
                    "Batch is already at the minimum. Cannot reduce further. Exiting..."
                )
                return None
            else:
                print("Caught OutOfMemoryError. Reducing batch size...")
                batch_size //= 2
                print("New batch size:", batch_size)

    forecasts = list(forecast_it)
    targets = list(ts_it)

    if max_windows:
        targets = targets[:max_windows]

    if subset_series:
        targets = [target.iloc[:, subset_series] for target in targets]
        for target in targets:
            target.columns = list(range(len(subset_series)))

    # Restrict targets array to until the interpolated segment to make it look like a forecasting task to MultivariateEvaluator
    # Also collect the start_date of the forecasts; and replace the forecast start_dates simultaneously
    # TODO: Verify this manually
    interpolation_segment_targets = []
    interpolation_start_dates = []
    interpolation_window_start = (
        window_length - prediction_length - (history_length // 2)
    )
    interpolation_window_end = window_length - (history_length // 2)
    end_ts = interpolation_window_end
    for k, target in enumerate(targets):
        modified_target = target.iloc[: end_ts + k]
        interpolation_segment_targets.append(modified_target)
        interpolation_start_dates.append(target.index[interpolation_window_start + k])
        forecasts[k].start_date = target.index[interpolation_window_start + k]

    print(
        "#Forecasts:", len(forecasts), "#Targets:", len(interpolation_segment_targets)
    )
    print("Shape of forecasts[0].samples", forecasts[0].samples.shape)
    predicting_end_time = time.time()
    print(
        "Metrics function: Predicting time:",
        predicting_end_time - predicting_start_time,
    )

    # A raw dump of the forecasts and targets for post-hoc analysis if needed, in the experiment folder
    # Can be loaded with the simple script:
    if savedir:
        savefile = os.path.join(savedir, "forecasts_targets.pkl")
        with open(savefile, "wb") as f:
            pickle.dump((forecasts, interpolation_segment_targets), f)

    # The results are going to be meaningless if any NaN shows up in the results,
    # so catch them here
    num_nan = 0
    num_inf = 0
    for forecast in forecasts:
        num_nan += np.isnan(forecast.samples).sum()
        num_inf += np.isinf(forecast.samples).sum()
    if num_nan > 0 or num_inf > 0:
        if skip_energy:
            return {
                "CRPS": float("nan"),
                "ND": float("nan"),
                "NRMSE": float("nan"),
                "MSE": float("nan"),
                "CRPS-Sum": float("nan"),
                "ND-Sum": float("nan"),
                "NRMSE-Sum": float("nan"),
                "MSE-Sum": float("nan"),
                "num_nan": num_nan,
                "num_inf": num_inf,
            }
        else:
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
    evaluator = MultivariateEvaluator(
        quantiles=(np.arange(n_quantiles) / float(n_quantiles))[1:],
        target_agg_funcs={"sum": np.sum},
    )

    # The GluonTS evaluator is very noisy on the standard error, so suppress it.
    metrics_calc_start_time = time.time()
    with SuppressOutput():
        agg_metric, _ = evaluator(interpolation_segment_targets, forecasts)
    metrics_calc_end_time = time.time()
    print(
        "Metrics function: Metrics calc time:",
        metrics_calc_end_time - metrics_calc_start_time,
    )

    if skip_energy:
        metrics = {
            "CRPS": agg_metric.get("mean_wQuantileLoss", float("nan")),
            "ND": agg_metric.get("ND", float("nan")),
            "NRMSE": agg_metric.get("NRMSE", float("nan")),
            "MSE": agg_metric.get("MSE", float("nan")),
            "CRPS-Sum": agg_metric.get("m_sum_mean_wQuantileLoss", float("nan")),
            "ND-Sum": agg_metric.get("m_sum_ND", float("nan")),
            "NRMSE-Sum": agg_metric.get("m_sum_NRMSE", float("nan")),
            "MSE-Sum": agg_metric.get("m_sum_MSE", float("nan")),
            "num_nan": num_nan,
            "num_inf": num_inf,
        }
    else:
        metrics = {
            "CRPS": agg_metric.get("mean_wQuantileLoss", float("nan")),
            "ND": agg_metric.get("ND", float("nan")),
            "NRMSE": agg_metric.get("NRMSE", float("nan")),
            "MSE": agg_metric.get("MSE", float("nan")),
            "CRPS-Sum": agg_metric.get("m_sum_mean_wQuantileLoss", float("nan")),
            "ND-Sum": agg_metric.get("m_sum_ND", float("nan")),
            "NRMSE-Sum": agg_metric.get("m_sum_NRMSE", float("nan")),
            "MSE-Sum": agg_metric.get("m_sum_MSE", float("nan")),
            "Energy": compute_energy_score(interpolation_segment_targets, forecasts),
            "num_nan": num_nan,
            "num_inf": num_inf,
        }

    if return_forecasts_and_targets:
        return metrics, forecasts, targets
    else:
        return metrics
