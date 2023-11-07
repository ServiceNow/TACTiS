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

import random
import torch
import numpy as np
import wandb
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class DummyConfig:
    def update(self, *args, **kwargs):
        return


class DummyLogger:
    def __init__(self, *args, **kwargs):
        self.name = "dummy_name"
        self.config = DummyConfig()
        return

    def log(self, *args, **kwargs):
        return

    def finish(self, *args, **kwargs):
        return


def log_parameter_difference(
    logger, initial_model, current_model, step=None, average_only=False
):
    initial_model_state_dict = initial_model.state_dict()

    for name, param in current_model.named_parameters():
        if not average_only:
            logger.log(
                {
                    "param_diff/"
                    + name: wandb.Histogram(
                        torch.abs(
                            initial_model_state_dict[name].flatten()
                            - param.data.flatten()
                        )
                    ),
                    "step": step,
                }
            )
        logger.log(
            {
                "average_param_diff/"
                + name: torch.abs(
                    initial_model_state_dict[name].flatten() - param.data.flatten()
                ).mean(),
                "step": step,
            }
        )
    return


def log_gradients(logger, model, objective_name="", step=None, average_only=False):
    if objective_name and not objective_name.endswith("/"):
        objective_name = objective_name + "/"
    for name, param in model.named_parameters():
        if param.grad != None:
            if not average_only:
                logger.log(
                    {
                        objective_name
                        + "gradients/"
                        + name: wandb.Histogram(param.grad.flatten()),
                        "step": step,
                    }
                )
            logger.log(
                {
                    objective_name
                    + "average-gradients/"
                    + name: param.grad.flatten().mean(),
                    "step": step,
                }
            )
    return


def save_checkpoint(state, checkpoint_dir, filename):
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    torch.save(state, os.path.join(checkpoint_dir, filename))


def get_correlation_as_pil(correlation, filepath):
    plt.figure()
    svm = sns.heatmap(
        correlation, robust=True, center=0, xticklabels=False, yticklabels=False
    )
    filename = os.path.join(filepath)
    figure = svm.get_figure()
    figure.savefig(filename, bbox_inches="tight")
    image = Image.open(filename)
    return image


def plot_matrices(targets, forecasts, wandb_log_prefix, epoch):
    corrcoef_v = 0
    corrcoef_ts = 0
    corrcoef_v_first_timestep = 0
    corrcoef_v_last_timestep = 0
    corrcoef_ts_first_variable = 0
    corrcoef_ts_last_variable = 0

    for i, (target, forecast) in enumerate(zip(targets, forecasts)):
        # # The targets should always end with the prediction window
        # assert target.index[-forecast.prediction_length] == forecast.start_date
        # target_data = target.iloc[-forecast.prediction_length :].to_numpy()
        samples = forecast.samples
        #     print(i, target_data.shape, samples.shape)

        for v in range(samples.shape[2]):
            spl = samples[:, :, v].transpose()
            corrcoef_ts = corrcoef_ts + np.corrcoef(spl)
            if i == 0 and v == 0:
                print(
                    "For correlations across timesteps, we take correlation across",
                    spl.shape,
                    "resulting in corr matrix of size",
                    corrcoef_ts.shape,
                )

        for ts in range(samples.shape[1]):
            spl = samples[:, ts, :].transpose()
            corrcoef_v = corrcoef_v + np.corrcoef(spl)
            if i == 0 and ts == 0:
                print(
                    "For correlations across variables, we take correlation across",
                    spl.shape,
                    "resulting in corr matrix of size",
                    corrcoef_v.shape,
                )

        corrcoef_v_first_timestep = corrcoef_v_first_timestep + np.corrcoef(
            samples[:, 0, :].transpose()
        )
        corrcoef_v_last_timestep = corrcoef_v_last_timestep + np.corrcoef(
            samples[:, -1, :].transpose()
        )

        corrcoef_ts_first_variable = corrcoef_ts_first_variable + np.corrcoef(
            samples[:, :, 0].transpose()
        )
        corrcoef_ts_last_variable = corrcoef_ts_last_variable + np.corrcoef(
            samples[:, :, -1].transpose()
        )

    corrcoef_ts /= len(targets) * 107
    corrcoef_v /= len(targets) * 12
    corrcoef_v_first_timestep /= len(targets)
    corrcoef_v_last_timestep /= len(targets)
    corrcoef_ts_first_variable /= len(targets)
    corrcoef_ts_last_variable /= len(targets)

    corrcoef_ts_fig = get_correlation_as_pil(
        corrcoef_ts,
        os.path.join(wandb.run.dir, "epoch_" + str(epoch) + "corrcoef_ts.png"),
    )
    wandb.log(
        {
            wandb_log_prefix + "-corrcoef/ts": wandb.Image(corrcoef_ts_fig),
            wandb_log_prefix + "-eval/" + wandb_log_prefix + "-eval-epoch": epoch,
        }
    )

    corrcoef_v_fig = get_correlation_as_pil(
        corrcoef_v,
        os.path.join(wandb.run.dir, "epoch_" + str(epoch) + "corrcoef_v.png"),
    )
    wandb.log(
        {
            wandb_log_prefix + "-corrcoef/v": wandb.Image(corrcoef_v_fig),
            wandb_log_prefix + "-eval/" + wandb_log_prefix + "-eval-epoch": epoch,
        }
    )

    corrcoef_v_first_timestep_fig = get_correlation_as_pil(
        corrcoef_v_first_timestep,
        os.path.join(
            wandb.run.dir, "epoch_" + str(epoch) + "corrcoef_v_first_timestep.png"
        ),
    )
    wandb.log(
        {
            wandb_log_prefix
            + "-corrcoef/v_first_timestep": wandb.Image(corrcoef_v_first_timestep_fig),
            wandb_log_prefix + "-eval/" + wandb_log_prefix + "-eval-epoch": epoch,
        }
    )

    corrcoef_v_last_timestep_fig = get_correlation_as_pil(
        corrcoef_v_last_timestep,
        os.path.join(
            wandb.run.dir, "epoch_" + str(epoch) + "corrcoef_v_last_timestep.png"
        ),
    )
    wandb.log(
        {
            wandb_log_prefix
            + "-corrcoef/v_last_timestep": wandb.Image(corrcoef_v_last_timestep_fig),
            wandb_log_prefix + "-eval/" + wandb_log_prefix + "-eval-epoch": epoch,
        }
    )

    corrcoef_ts_first_variable_fig = get_correlation_as_pil(
        corrcoef_ts_first_variable,
        os.path.join(
            wandb.run.dir, "epoch_" + str(epoch) + "corrcoef_ts_first_variable.png"
        ),
    )
    wandb.log(
        {
            wandb_log_prefix
            + "-corrcoef/ts_first_variable": wandb.Image(
                corrcoef_ts_first_variable_fig
            ),
            wandb_log_prefix + "-eval/" + wandb_log_prefix + "-eval-epoch": epoch,
        }
    )

    corrcoef_ts_last_variable_fig = get_correlation_as_pil(
        corrcoef_ts_last_variable,
        os.path.join(
            wandb.run.dir, "epoch_" + str(epoch) + "corrcoef_ts_last_variable.png"
        ),
    )
    wandb.log(
        {
            wandb_log_prefix
            + "-corrcoef/ts_last_variable": wandb.Image(corrcoef_ts_last_variable_fig),
            wandb_log_prefix + "-eval/" + wandb_log_prefix + "-eval-epoch": epoch,
        }
    )


def plot_single_series(
    samples,
    all_targets,
    prediction_timesteps,
    all_timesteps,
    wandb_log_prefix,
    wandb_log_suffix,
    epoch,
):
    """
    Shapes:
    samples: (pred_len, num_samples)
    all_targets: (pred_len + hist_len, )
    prediction_timesteps: (pred_len,)
    all_timesteps: (pred_len + hist_len, )
    """
    plt.figure()

    for zorder, quant, color, label in [
        [1, 0.05, (0.75, 0.75, 1), "5%-95%"],
        [2, 0.10, (0.25, 0.25, 1), "10%-90%"],
        [3, 0.25, (0, 0, 0.75), "25%-75%"],
    ]:
        plt.fill_between(
            prediction_timesteps,
            np.quantile(samples, quant, axis=1),
            np.quantile(samples, 1 - quant, axis=1),
            facecolor=color,
            interpolate=True,
            label=label,
            zorder=zorder,
        )

    plt.plot(
        prediction_timesteps,
        np.quantile(samples, 0.5, axis=1),
        color=(0.5, 0.5, 0.5),
        linewidth=3,
        label="50%",
        zorder=4,
    )

    plt.plot(
        all_timesteps,
        all_targets,
        color=(0, 0, 0),
        linewidth=2,
        zorder=5,
        label="ground truth",
    )

    xmin = prediction_timesteps[0]  # Hardcoded to have the same for both subplots
    xmax = prediction_timesteps[-1]  # Go to the right-end of the plot
    plt.axvspan(xmin=xmin, xmax=xmax, facecolor=(0.2, 0.5, 0.2), alpha=0.1)

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 2, 3, 4, 0]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    filename = os.path.join(
        wandb.run.dir,
        "epoch_"
        + str(epoch)
        + "_"
        + wandb_log_prefix
        + "_"
        + wandb_log_suffix
        + ".png",
    )
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    image = Image.open(filename)
    wandb.log(
        {
            wandb_log_prefix + "-forecasts/" + wandb_log_suffix: wandb.Image(image),
            wandb_log_prefix + "-eval/" + wandb_log_prefix + "-eval-epoch": epoch,
        }
    )

    plt.close()
