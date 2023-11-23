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

from typing import List, Tuple, Optional

import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from copy import deepcopy
from gluonts.model.forecast import Forecast


def plot_single_forecast(
    forecast: Forecast,
    target: pd.Series,
    axes: plt.Axes,
    locator: matplotlib.ticker.Locator,
) -> None:
    """
    Plot the forecast for a single series, on the given Axes object.

    Parameters:
    -----------
    forecast: Forecast
        The univariate forecast object generated through the GluonTS interface.
    target: pd.Series
        The ground truth for the series.
    axes: plt.Axes
        The Axes object on which to draw the forecast.
    locator: matplotlib.ticker.Locator
        An object defining how the horizontal ticks will be selected.
    """
    # Add the last point of historical data to the forecast, for a smoother transition.
    last_hist_index = target.index[-forecast.prediction_length - 1]
    last_hist_value = target[-forecast.prediction_length - 1]

    forecast_index = [last_hist_index] + [x for x in forecast.index]

    def quantile(q: float) -> np.ndarray:
        return np.append(last_hist_value, forecast.quantile(q))

    # Plot the forecasts with 3 different quantile ranges.
    axes.fill_between(
        forecast_index,
        quantile(0.05),
        quantile(0.95),
        facecolor=(0.75, 0.75, 1),
        interpolate=True,
        label="5%-95%",
        zorder=1,
    )
    axes.fill_between(
        forecast_index,
        quantile(0.1),
        quantile(0.9),
        facecolor=(0.25, 0.25, 1),
        interpolate=True,
        label="10%-90%",
        zorder=2,
    )
    axes.fill_between(
        forecast_index,
        quantile(0.25),
        quantile(0.75),
        facecolor=(0.0, 0.0, 0.75),
        interpolate=True,
        label="25%-75%",
        zorder=3,
    )

    # Plot the forecast median
    axes.plot(
        forecast_index,
        quantile(0.5),
        color=(0.5, 0.5, 0.5),
        linewidth=3,
        label="50%",
        zorder=4,
    )

    # Plot the ground truth with a white highlight
    target.plot(
        ax=axes,
        color="#FFFFFF",
        alpha=1,
        linewidth=2,
        zorder=5,
    )
    target.plot(
        ax=axes,
        color="#FC6A08",
        alpha=1,
        linewidth=1.5,
        label="Ground Truth",
        zorder=6,
    )

    # Highlight the forecasting regions
    axes.set_xlim(axes.get_xlim())
    xmin = forecast.index[0] - (forecast.index[1] - forecast.index[0]) / 2
    xmax = forecast.index[-1] + (forecast.index[-1] - forecast.index[0])  # Beyond the limit of the plot
    axes.axvspan(xmin=xmin, xmax=xmax, facecolor=(0.2, 0.5, 0.2), alpha=0.1)

    # Allows for nice time-based ticks
    axes.xaxis.set_major_locator(deepcopy(locator))
    axes.tick_params(
        axis="both",
        length=5,
        width=1,
        color=(0.7, 0.7, 0.7),
        left=True,
        bottom=True,
    )


def plot_four_forecasts(
    forecasts: List[Forecast],
    targets: List[pd.DataFrame],
    selection: List[Tuple[int, int]],
    tick_freq: str = "day",
    history_length: Optional[int] = None,
    savefile: Optional[str] = None,
) -> None:
    """
    Plot the forecast for four series, from potentially multiple forecasts.

    Parameters:
    -----------
    forecasts: List[Forecast]
        A list of multivariate forecasts generated through the GluonTS interface.
    targets: List[pd.DataFrame]
        A list of multivariate ground truths.
    selection: List[Tuple[int, int]]
        A list of 4 pairs of integers, to select which series to plot.
        The first element of the pairs selects which forecast to plot, while the second selects which series to plot.
    tick_freq: str, from "day", "6 hours", or "4 months end"
        The frequency of the horizontal tick marks.
    history_length: Optional[int], default to None
        If set, how much history to plot from the ground-truth (minimum = 1).
        If not set, default to the prediction length of the forecasts.
    """
    assert len(selection) == 4, "plot_four_forecasts() can only plot 4 series at once."

    # Seaborn nice default
    sns.set()
    sns.set_context("paper", font_scale=1)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))

    locator = {
        "day": matplotlib.dates.DayLocator(),
        "6 hours": matplotlib.dates.HourLocator(range(0, 24, 6)),
        "4 months end": matplotlib.dates.MonthLocator(range(1, 13, 4), bymonthday=30),
    }[tick_freq]

    counter = 0
    for forecast_num, series_num in selection:
        forecast = forecasts[forecast_num]
        target = targets[forecast_num]

        single_forecast = forecast.copy_dim(series_num)
        if history_length is None:
            window_length = 2 * single_forecast.prediction_length
        else:
            window_length = history_length + single_forecast.prediction_length
        single_target = target[series_num][-window_length:]
        plot_single_forecast(
            forecast=single_forecast,
            target=single_target,
            axes=axs.flat[counter],
            locator=locator,
        )
        counter += 1
    plt.subplots_adjust(hspace=0.3)

    # Reorder the legend, and have a unique one for all 4 plots.
    handles, labels = axs.flat[0].get_legend_handles_labels()
    order = [5, 0, 1, 2, 3]
    handles = [handles[o] for o in order]
    labels = [labels[o] for o in order]
    fig.legend(handles, labels, bbox_to_anchor=(0.91, 0.89), loc="upper left")
    fig.show()

    if savefile:
        fig.savefig(savefile, bbox_inches="tight", pad_inches=0)
