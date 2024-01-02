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
import torch
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

    def __init__(self, window_length: int):
        super().__init__()
        self.window_length = window_length
        self.num_windows_seen = 0

    def flatmap_transform(self, data: DataEntry, is_train: bool) -> Iterator[DataEntry]:
        full_length = data["target"].shape[-1]
        for end_point in range(self.window_length, full_length + 1):
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
    prediction_length: int,
    num_samples: int,
    split: bool = True,
    savedir: Optional[str] = None,
    return_forecasts_and_targets: bool = False,
    subset_series=None,
    skip_energy=True,
    n_quantiles=20,
):
    if split:
        split_dataset = transform.TransformedDataset(dataset, transformation=SplitValidationTransform(window_length))
    else:
        split_dataset = dataset

    while True:
        print("Using batch size:", predictor.batch_size)
        try:
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=split_dataset, predictor=predictor, num_samples=num_samples
            )
            forecasts = list(forecast_it)
            targets = list(ts_it)
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            print(error)
            if predictor.batch_size == 1:
                print("Batch is already at the minimum. Cannot reduce further. Exiting...")
                return None
            else:
                print("Caught OutOfMemoryError. Reducing batch size...")
                predictor.batch_size //= 2
                gc.collect()
                torch.cuda.empty_cache()

    if subset_series:
        targets = [target.iloc[:, subset_series] for target in targets]
        for target in targets:
            target.columns = list(range(len(subset_series)))

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
    with SuppressOutput():
        agg_metric, ts_wise_metrics = evaluator(targets, forecasts)

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
    skip_energy=True,
    n_quantiles=20,
):
    if split:
        split_dataset = transform.TransformedDataset(dataset, transformation=SplitValidationTransform(window_length))
    else:
        raise Exception("split=False is not support in compute_validation_metrics_interpolation")

    while True:
        print("Using batch size:", predictor.batch_size)
        try:
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=split_dataset, predictor=predictor, num_samples=num_samples
            )
            forecasts = list(forecast_it)
            targets = list(ts_it)
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            print(error)
            if predictor.batch_size == 1:
                print("Batch is already at the minimum. Cannot reduce further. Exiting...")
                return None
            else:
                print("Caught OutOfMemoryError. Reducing batch size...")
                predictor.batch_size //= 2
                gc.collect()
                torch.cuda.empty_cache()

    if subset_series:
        targets = [target.iloc[:, subset_series] for target in targets]
        for target in targets:
            target.columns = list(range(len(subset_series)))

    # Store the original targets (interpolation "before" + prediction window + interpolation "after") for plotting/visualization purposes
    full_targets = []
    # Restrict targets array to until the interpolated segment to make it look like a forecasting task to MultivariateEvaluator
    # Also collect the start_date of the forecasts; and replace the forecast start_dates simultaneously
    # interpolation_segment_targets is the targets
    interpolation_segment_targets = []
    num_timesteps_observed_on_each_side = (window_length - prediction_length) // 2
    interpolation_window_start = num_timesteps_observed_on_each_side
    interpolation_window_end = num_timesteps_observed_on_each_side + prediction_length
    end_ts = interpolation_window_end

    for k, target in enumerate(targets):
        # Calculate offset
        offset = len(target) - window_length
        # Obtain and store the actual target window for interpolation
        modified_target = target.iloc[interpolation_window_start + offset : end_ts + offset]
        interpolation_segment_targets.append(modified_target)
        # Modify the start date of the window to the right one, overriding the one set by GluonTS based on the forecasting window
        forecasts[k].start_date = target.index[interpolation_window_start + offset]

        ## Store the original targets
        full_targets.append(target.iloc[offset : end_ts + num_timesteps_observed_on_each_side + offset])

    # Set targets to the new targets array
    targets = interpolation_segment_targets

    # A raw dump of the forecasts and targets for post-hoc analysis if needed, in the experiment folder
    # Can be loaded with the simple script:
    if savedir:
        savefile = os.path.join(savedir, "interpolation_targets.pkl")
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
    with SuppressOutput():
        agg_metric, ts_wise_metrics = evaluator(targets, forecasts)

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
        # Note: we return the full targets since we need that for plotting
        return metrics, ts_wise_metrics, forecasts, full_targets
    else:
        return metrics, ts_wise_metrics
