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

from typing import Iterator, Tuple

import pandas as pd
from copy import deepcopy

from gluonts.dataset.common import DataEntry, Dataset
from gluonts.model.forecast import Forecast
from gluonts.model.predictor import Predictor
from gluonts.transform import AdhocTransform


def make_evaluation_predictions(
    dataset: Dataset, predictor: Predictor, num_samples: int = 100
) -> Tuple[Iterator[Forecast], Iterator[pd.Series]]:
    """
    Returns predictions for the trailing prediction_length observations of the given
    time series, using the given predictor.

    The predictor will take as input the given time series without the trailing
    prediction_length observations.

    Parameters
    ----------
    dataset
        Dataset where the evaluation will happen. Only the portion excluding
        the prediction_length portion is used when making prediction.
    predictor
        Model used to draw predictions.
    num_samples
        Number of samples to draw on the model when evaluating. Only sampling-based
        models will use this.

    Returns
    -------
    Tuple[Iterator[Forecast], Iterator[pd.Series]]
        A pair of iterators, the first one yielding the forecasts, and the second
        one yielding the corresponding ground truth series.
    """

    prediction_length = predictor.prediction_length
    freq = predictor.freq
    lead_time = predictor.lead_time

    def add_ts_dataframe(
        data_iterator: Iterator[DataEntry],
    ) -> Iterator[DataEntry]:
        for data_entry in data_iterator:
            data = data_entry.copy()
            index = pd.date_range(
                start=data["start"],
                freq=freq,
                periods=data["target"].shape[-1],
            )
            data["ts"] = pd.DataFrame(index=index, data=data["target"].transpose())
            yield data

    def ts_iter(dataset: Dataset) -> pd.DataFrame:
        for data_entry in add_ts_dataframe(iter(dataset)):
            yield data_entry["ts"]

    def truncate_target(data):
        data = data.copy()
        target = data["target"]
        assert target.shape[-1] >= prediction_length  # handles multivariate case (target_dim, history_length)
        data["target"] = target[..., : -prediction_length - lead_time]
        return data

    dataset_as_is = deepcopy(dataset)
    dataset_trunc = AdhocTransform(truncate_target).apply(dataset)

    return (
        predictor.predict(dataset_trunc, num_samples=num_samples),
        ts_iter(dataset_as_is),
    )
