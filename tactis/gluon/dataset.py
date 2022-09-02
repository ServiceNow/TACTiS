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

>> The methods to load the various datasets used in the TACTiS ICML paper.
"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from gluonts.dataset.common import DataEntry, MetaData, Dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository._tsf_datasets import Dataset as MonashDataset
from gluonts.dataset.repository._tsf_datasets import datasets as monash_datasets
from gluonts.dataset.repository.datasets import (
    dataset_recipes,
    default_dataset_path,
    generate_forecasting_dataset,
    get_dataset,
    partial,
)


_DATA_BACKTEST_DEF = {
    "solar_10min": {  # 137 series, 72 prediction length
        "train_dates": [
            pd.Timestamp(year=2006, month=11, day=20),  # Monday
            pd.Timestamp(year=2006, month=11, day=27),
            pd.Timestamp(year=2006, month=12, day=4),
            pd.Timestamp(year=2006, month=12, day=11),
            pd.Timestamp(year=2006, month=12, day=18),
            pd.Timestamp(year=2006, month=12, day=25),
        ],
        "end_date": pd.Timestamp(year=2007, month=1, day=1),  # A Monday
    },
    "electricity_hourly": {  # 321 series, 24 prediction length
        "train_dates": [
            pd.Timestamp(year=2014, month=11, day=17),  # A Monday
            pd.Timestamp(year=2014, month=11, day=24),
            pd.Timestamp(year=2014, month=12, day=1),
            pd.Timestamp(year=2014, month=12, day=8),
            pd.Timestamp(year=2014, month=12, day=15),
            pd.Timestamp(year=2014, month=12, day=22),
        ],
        "end_date": pd.Timestamp(year=2014, month=12, day=29),  # Last Monday before end of data
    },
    "kdd_cup_2018_without_missing": {  # 270 series, 48 prediction length
        "train_dates": [
            pd.Timestamp(year=2018, month=1, day=1),  # A Monday
            pd.Timestamp(year=2018, month=1, day=15),
            pd.Timestamp(year=2018, month=1, day=29),
            pd.Timestamp(year=2018, month=2, day=12),
            pd.Timestamp(year=2018, month=2, day=26),
            pd.Timestamp(year=2018, month=3, day=12),
        ],
        "end_date": pd.Timestamp(year=2018, month=3, day=26),  # Last Monday before end of data
    },
    "traffic": {  # 862 series, 24 prediction length
        "train_dates": [
            pd.Timestamp(year=2016, month=11, day=14),  # A Monday
            pd.Timestamp(year=2016, month=11, day=21),
            pd.Timestamp(year=2016, month=11, day=28),
            pd.Timestamp(year=2016, month=12, day=5),
            pd.Timestamp(year=2016, month=12, day=12),
            pd.Timestamp(year=2016, month=12, day=19),
        ],
        "end_date": pd.Timestamp(year=2016, month=12, day=26),  # Last Monday before end of data
    },
    "fred_md": {  # 107 series, 12 prediction length
        "train_dates": [
            pd.Timestamp(year=2013, month=1, day=30),
            pd.Timestamp(year=2014, month=1, day=30),
            pd.Timestamp(year=2015, month=1, day=30),
            pd.Timestamp(year=2016, month=1, day=30),
            pd.Timestamp(year=2017, month=1, day=30),
            pd.Timestamp(year=2018, month=1, day=30),
        ],
        "end_date": pd.Timestamp(year=2019, month=1, day=30),  # Last January before end of data
    },
}


def _monash_inject_datasets(name, filename, record, prediction_length=None):
    """
    Injects datasets from the Monash Time Series Repository that were not included in GluonTS.
    """
    dataset_recipes.update(
        {name: partial(generate_forecasting_dataset, dataset_name=name, prediction_length=prediction_length)}
    )
    monash_datasets.update({name: MonashDataset(file_name=filename, record=record)})


# Modifications to the GluonTS dataset repository
# * We add missing datasets from the Monash Time Series Repository
# * We rename datasets to have a prefix that makes their source explicit
_monash_inject_datasets("electricity_hourly", "electricity_hourly_dataset.zip", "4656140", prediction_length=24)
_monash_inject_datasets("solar_10min", "solar_10_minutes_dataset.zip", "4656144", prediction_length=72)
_monash_inject_datasets("traffic", "traffic_hourly_dataset.zip", "4656132", prediction_length=24)


def _count_timesteps(left: pd.Timestamp, right: pd.Timestamp, delta: pd.DateOffset) -> int:
    """
    Count how many timesteps there are between left and right, according to the given timesteps delta.
    If the number if not integer, round down.
    """

    # This is due to GluonTS replacing Timestamp by Period for version 0.10.0. 
    # Original code was tested on version 0.9.4
    if  type(left) == pd.Period: left = left.to_timestamp()
    if  type(right) == pd.Period: right = right.to_timestamp()
    
    assert right >= left, f"Case where left ({left}) is after right ({right}) is not implemented in _count_timesteps()."
    try:
        return (right - left) // delta
    except TypeError:
        # For MonthEnd offsets, the division does not work, so we count months one by one.
        for i in range(10000):
            if left + (i + 1) * delta > right:
                return i
        else:
            raise RuntimeError(
                f"Too large difference between both timestamps ({left} and {right}) for _count_timesteps()."
            )


def _load_raw_dataset(name: str, use_cached: bool) -> Tuple[MetaData, List[DataEntry]]:
    """
    Load the dataset using GluonTS method, and combining both the train and test data.

    The combination is needed due to going through GluonTS, and could be avoided by loading the data directly from
    the Monash repository.
    """
    # Where to save the downloaded input files
    cache_path = Path(os.environ.get("TACTIS_DATA_STORE", default_dataset_path))
    # This gives the univariate version of the dataset, with pre-made train/test split
    uv_dataset = get_dataset(name, regenerate=not use_cached, path=cache_path)

    # Combine the training and testing portion of the dataset, giving a single dataset containing all the data
    timestep_delta = pd.tseries.frequencies.to_offset(uv_dataset.metadata.freq)

    data = [series.copy() for series in uv_dataset.train]

    # This would be more efficient if done in reverse order, but we only have an iterator, not a list
    for i, new_series in enumerate(uv_dataset.test):
        # The test datasets are in the same order as the train datasets, but repeated for each test timestamp
        old_series = data[i % len(data)]

        # Make sure we don't combine series that we know are distinct
        if "feat_static_cat" in new_series:
            assert old_series["feat_static_cat"] == new_series["feat_static_cat"]

        if old_series["start"] > new_series["start"]:
            extra_timesteps = _count_timesteps(new_series["start"], old_series["start"], timestep_delta)
            # Not robust, but at the very least check that first common entry is the same before combining the data
            assert old_series["target"][0] == new_series["target"][extra_timesteps]
            old_series["start"] = new_series["start"]
            old_series["target"] = np.concatenate([new_series["target"][0:extra_timesteps], old_series["target"]])

        old_end = old_series["start"] + len(old_series["target"]) * timestep_delta
        new_end = new_series["start"] + len(new_series["target"]) * timestep_delta
        if new_end > old_end:
            extra_timesteps = _count_timesteps(old_end, new_end, timestep_delta)
            assert old_series["target"][-1] == new_series["target"][-extra_timesteps - 1]
            old_series["target"] = np.concatenate([old_series["target"], new_series["target"][-extra_timesteps:]])

    return uv_dataset.metadata, data


def generate_hp_search_datasets(
    name: str, history_length_multiple: float, use_cached: bool = True
) -> Tuple[MetaData, Dataset, Dataset]:
    """
    Generate the training and validation datasets to be used during the hyperparameter search.

    The validation dataset ends at the timestep of the first backtesting period.
    The length of the validation period is equal to 7 times the prediction length, plus the needed history length.
    The training dataset ends at the beginning of the validation dataset (ignoring the needed history length),
    and starts at the beginning of the full dataset.

    Parameters:
    -----------
    name: str
        The name of the dataset.
    history_length_multiple: float
        The length of the history that will be sent to the model, as a multiple of the dataset prediction length.
        The result is rounded down to the nearest integer.
    use_cached: bool, default to True
        If set to True, use the cached version of the data if available.

    Returns
    -------
    metadata: MetaData
        The MetaData of the dataset.
    train_data: Dataset
        The training dataset.
    valid_data: Dataset
        The validation dataset.
    """
    metadata, raw_dataset = _load_raw_dataset(name, use_cached=use_cached)

    first_backtest_timestamp = _DATA_BACKTEST_DEF[name]["train_dates"][0]
    history_length = int(history_length_multiple * metadata.prediction_length)
    validation_length = 7 * metadata.prediction_length

    timestep_delta = pd.tseries.frequencies.to_offset(metadata.freq)

    train_data = []
    valid_data = []
    for i, series in enumerate(raw_dataset):
        first_backtest_index = _count_timesteps(series["start"], first_backtest_timestamp, timestep_delta)
        train_end_index = first_backtest_index - validation_length
        validation_start_index = first_backtest_index - validation_length - history_length

        s_train = series.copy()
        s_train["target"] = series["target"][:train_end_index]
        s_train["item_id"] = i
        train_data.append(s_train)

        s_valid = series.copy()
        s_valid["start"] = s_valid["start"] + validation_length * timestep_delta
        s_valid["target"] = series["target"][validation_start_index:first_backtest_index]
        s_valid["item_id"] = i
        valid_data.append(s_valid)

    # MultivariateGrouper call operation is not without side-effect, so we need two independant ones.
    train_grouper = MultivariateGrouper()
    valid_grouper = MultivariateGrouper()

    return metadata, train_grouper(train_data), valid_grouper(valid_data)


def maximum_backtest_id(name: str) -> int:
    """
    Return the largest possible backtesting id for the given dataset.

    Parameters:
    -----------
    name: str
        The name of the dataset.

    Returns
    -------
    maximum_id
        The largest value for the backtest_id parameter in generate_backtesting_datasets().
    """
    return len(_DATA_BACKTEST_DEF[name]["train_dates"])


def generate_backtesting_datasets(
    name: str, backtest_id: int, history_length_multiple: float, use_cached: bool = True
) -> Tuple[MetaData, Dataset, Dataset]:
    """
    Generate the training and testing datasets to be used during the backtesting.

    The training dataset ends at the timestamp associated with the given backtesting id.
    The testing dataset contains multiple testing instances, each separated by the prediction length,
    starting from its backtesting timestamp to the next backtesting period timestamp (or the ending timestamp).

    Parameters:
    -----------
    name: str
        The name of the dataset.
    backtest_id: int
        The identifier for the backtesting period. Its maximum value can be told by maximum_backtest_id().
    history_length_multiple: float
        The length of the history that will be sent to the model, as a multiple of the dataset prediction length.
        The result is rounded down to the nearest integer.
    use_cached: bool, default to True
        If set to True, use the cached version of the data if available.

    Returns
    -------
    metadata: MetaData
        The MetaData of the dataset.
    train_data: Dataset
        The training dataset.
    test_data: Dataset
        The testing dataset.
    """
    metadata, raw_dataset = _load_raw_dataset(name, use_cached=use_cached)

    backtest_timestamp = _DATA_BACKTEST_DEF[name]["train_dates"][backtest_id]
    history_length = int(history_length_multiple * metadata.prediction_length)

    timestep_delta = pd.tseries.frequencies.to_offset(metadata.freq)
    test_offset = timestep_delta * metadata.prediction_length
    if backtest_id + 1 < maximum_backtest_id(name):
        num_test_dates = _count_timesteps(
            _DATA_BACKTEST_DEF[name]["train_dates"][backtest_id],
            _DATA_BACKTEST_DEF[name]["train_dates"][backtest_id + 1],
            test_offset,
        )
    else:
        num_test_dates = _count_timesteps(
            _DATA_BACKTEST_DEF[name]["train_dates"][backtest_id], _DATA_BACKTEST_DEF[name]["end_date"], test_offset
        )

    train_data = []
    for i, series in enumerate(raw_dataset):
        train_end_index = _count_timesteps(series["start"], backtest_timestamp, timestep_delta)

        s_train = series.copy()
        s_train["target"] = series["target"][:train_end_index]
        s_train["item_id"] = i
        train_data.append(s_train)

    # GluonTS multivariate format for multiple tests is ordered first by date, then by series.
    test_data = []
    for test_id in range(num_test_dates):
        for i, series in enumerate(raw_dataset):
            train_end_index = _count_timesteps(series["start"], backtest_timestamp, timestep_delta)
            test_end_index = train_end_index + metadata.prediction_length * (test_id + 1)
            test_start_index = test_end_index - metadata.prediction_length - history_length

            s_test = series.copy()
            s_test["start"] = series["start"] + test_start_index * timestep_delta
            s_test["target"] = series["target"][test_start_index:test_end_index]
            s_test["item_id"] = len(test_data)
            test_data.append(s_test)

    train_grouper = MultivariateGrouper()
    test_grouper = MultivariateGrouper(num_test_dates=num_test_dates)

    return metadata, train_grouper(train_data), test_grouper(test_data)
