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

import torch
import argparse

from tactis.gluon.utils import set_seed
from tactis.gluon.estimator import TACTiSEstimator
from tactis.gluon.trainer import TACTISTrainer
from tactis.gluon.dataset import (
    generate_hp_search_datasets,
    generate_backtesting_datasets,
    generate_prebacktesting_datasets,
)
from tactis.model.utils import check_memory


def main(args):
    seed = args.seed
    num_workers = args.num_workers
    history_factor = args.history_factor
    epochs = args.epochs
    load_checkpoint = args.load_checkpoint
    activation_function = args.decoder_act
    dataset = args.dataset
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    clip_gradient = args.clip_gradient

    if args.use_cpu:
        print("Using CPU")

    checkpoint_path = args.checkpoint_path
    logger = None

    # Print memory avl.
    if not args.use_cpu:
        total, used = check_memory(0)
        print("Used/Total GPU memory:", used, "/", total)

    # Restrict memory to 12 GB if it greater than 12 GB
    # 12198 is the exact memory of a 12 GB P100
    if not args.do_not_restrict_memory and not args.use_cpu:
        if int(total) > 12198:
            fraction_to_use = 11598 / int(total)
            torch.cuda.set_per_process_memory_fraction(fraction_to_use, 0)
            print("Restricted memory to 12 GB")

    series_length_maps = {
        "solar_10min": 137,
        "electricity_hourly": 321,
        "kdd_cup_2018_without_missing": 270,
        "traffic": 862,
        "fred_md": 107,
    }

    prediction_length_maps = {
        "solar_10min": 72,
        "electricity_hourly": 24,
        "kdd_cup_2018_without_missing": 48,
        "traffic": 24,
        "fred_md": 12,
    }

    ### Decide the prediction factor for the dataloader
    prediction_length = prediction_length_maps[dataset]
    print("Using history factor:", history_factor)
    print("Prediction length of the dataset:", prediction_length_maps[dataset])

    if args.bagging_size:
        assert args.bagging_size < series_length_maps[dataset]

    encoder_dict = {
        "flow_temporal_encoder": {
            "attention_layers": args.flow_encoder_num_layers,
            "attention_heads": args.flow_encoder_num_heads,
            "attention_dim": args.flow_encoder_dim,
            "attention_feedforward_dim": args.flow_encoder_dim,
            "dropout": 0.0,
        },
        "copula_temporal_encoder": {
            "attention_layers": args.copula_encoder_num_layers,
            "attention_heads": args.copula_encoder_num_heads,
            "attention_dim": args.copula_encoder_dim,
            "attention_feedforward_dim": args.copula_encoder_dim,
            "dropout": 0.0,
        },
    }

    # num_series is passed separately by the estimator
    model_parameters = {
        "flow_series_embedding_dim": args.flow_series_embedding_dim,
        "copula_series_embedding_dim": args.copula_series_embedding_dim,
        "flow_input_encoder_layers": args.flow_input_encoder_layers,
        "copula_input_encoder_layers": args.copula_input_encoder_layers,
        "bagging_size": args.bagging_size,
        "input_encoding_normalization": True,
        "data_normalization": "standardization",
        "loss_normalization": args.loss_normalization,
        "positional_encoding": {
            "dropout": 0.0,
        },
        **encoder_dict,
        "copula_decoder": {
            # flow_input_dim and copula_input_dim are passed by the TACTIS module dynamically
            "min_u": 0.05,
            "max_u": 0.95,
            "attentional_copula": {
                "attention_heads": args.decoder_num_heads,
                "attention_layers": args.decoder_num_layers,
                "attention_dim": args.decoder_dim,
                "mlp_layers": args.decoder_mlp_layers,
                "mlp_dim": args.decoder_mlp_dim,
                "resolution": args.decoder_resolution,
                "attention_mlp_class": args.decoder_attention_mlp_class,
                "dropout": 0.0,
            },
            "dsf_marginal": {
                "mlp_layers": args.dsf_mlp_layers,
                "mlp_dim": args.dsf_mlp_dim,
                "flow_layers": args.dsf_num_layers,
                "flow_hid_dim": args.dsf_dim,
            },
            "activation_function": activation_function,
        },
        "experiment_mode": args.experiment_mode,
        "skip_copula": True,
    }

    set_seed(seed)
    if args.backtest_id >= 0 and args.backtest_id <= 5:
        backtesting = True
        args.compute_validation_metrics_split = False
        print("Using backtest dataset with ID", args.backtest_id)
        if not args.prebacktest:
            print("CAUTION: The validation set here is the actual test set.")
            metadata, train_data, valid_data = generate_backtesting_datasets(dataset, args.backtest_id, history_factor)
        else:
            print("Using the prebacktesting set.")
            backtesting = False
            metadata, train_data, valid_data = generate_prebacktesting_datasets(
                dataset, args.backtest_id, history_factor
            )
            _, _, test_data = generate_backtesting_datasets(dataset, args.backtest_id, history_factor)
    else:
        backtesting = False
        print("Using HP search dataset")
        metadata, train_data, valid_data = generate_hp_search_datasets(dataset, history_factor)

    set_seed(seed)
    estimator_custom = TACTiSEstimator(
        model_parameters=model_parameters,
        num_series=train_data.list_data[0]["target"].shape[0],
        history_length=history_factor * metadata.prediction_length,
        prediction_length=prediction_length,
        freq=metadata.freq,
        trainer=TACTISTrainer(
            epochs=epochs,
            batch_size=args.batch_size,
            training_num_batches_per_epoch=args.training_num_batches_per_epoch,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clip_gradient=clip_gradient,
            device=torch.device("cuda") if not args.use_cpu else torch.device("cpu"),
            log_subparams_every=args.log_subparams_every,
            checkpoint_path=checkpoint_path,
            seed=seed,
            load_checkpoint=load_checkpoint,
            logger=logger,
            early_stopping_epochs=args.early_stopping_epochs,
            do_not_restrict_time=args.do_not_restrict_time,
        ),
        cdf_normalization=False,
    )

    estimator_custom.train(
        train_data,
        valid_data,
        num_workers=num_workers,
        optimizer=args.optimizer,
        backtesting=backtesting,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of multiprocessing workers.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch Size.")
    parser.add_argument("--epochs", type=int, default=1000, help="Epochs.")

    parser.add_argument("--optimizer", type=str, default="adam", choices=["rmsprop", "adam"], help="Optimizer to be used.")
    parser.add_argument(
        "--checkpoint_path",
        type=str, help="Folder to store all checkpoints in. This folder will be created automatically if it does not exist."
    )
    parser.add_argument("--load_checkpoint", type=str, help="Checkpoint to start training from.")
    parser.add_argument("--training_num_batches_per_epoch", type=int, default=512, help="Number of batches in a single epoch of training.")
    parser.add_argument(
        "--backtest_id",
        type=int,
        default=-1,
        help="Backtest set to use. Use -1 to use the hyperparameter set."
    )
    parser.add_argument(
        "--prebacktest",
        action="store_true",
        help="When specified, uses the last few windows of the training set as the validation set. To be used only when training during backtesting.",
    )
    parser.add_argument(
        "--log_subparams_every",
        type=int,
        default=10000,
        help="Frequency of logging the epoch number and iteration number during training.",
    )
    parser.add_argument("--bagging_size", type=int, default=20, help="Bagging Size")

    parser.add_argument(
        "--dataset",
        type=str,
        default="fred_md",
        choices=[
            "fred_md",
            "kdd_cup_2018_without_missing",
            "solar_10min",
            "electricity_hourly",
            "traffic",
        ],
        help="Dataset to train on"
    )

    # compute_validation_metrics_split - True or False (default: False)
    # if True, all "hist_len+pred_len" windows are taken and sampling+evaluation is done (easy for datasets with less dimensions like FRED)
    # if False, the last window is the only window used (which itself is time-consuming for datasets like KDD)
    parser.add_argument("--compute_validation_metrics_split", action="store_true", help="If True, windows are sampled across the sample. Else, only the last window in each sample is used.")

    # Early stopping epochs based on total validation loss. -1 indicates no early stopping.
    parser.add_argument("--early_stopping_epochs", type=int, default=50, help="Early stopping patience")

    # HPARAMS
    # General ones
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight Decay")
    parser.add_argument("--clip_gradient", type=float, default=1e3, help="Gradient Clipping Magnitude")
    parser.add_argument("--history_factor", type=int, default=1, help="History Factor")
    # Series embedding
    parser.add_argument("--flow_series_embedding_dim", type=int, default=5, help="Embedding Dimension of the Flow Series Encoder")
    parser.add_argument("--copula_series_embedding_dim", type=int, default=5, help="Embedding Dimension of the Copula Series Encoder")
    # Input embedding
    parser.add_argument("--flow_input_encoder_layers", type=int, default=2, help="Embedding Dimension of the Flow Encoder")
    parser.add_argument("--copula_input_encoder_layers", type=int, default=2, help="Embedding Dimension of the Copula Encoder")
    # Shared encoder
    parser.add_argument("--flow_encoder_num_layers", type=int, default=2, help="Number of Layers in the Flow Encoder")
    parser.add_argument("--flow_encoder_num_heads", type=int, default=1, help="Number of Heads in the Flow Encoder")
    parser.add_argument("--flow_encoder_dim", type=int, default=16, help="Embedding Dimension of the Flow Encoder")
    # Shared encoder
    parser.add_argument("--copula_encoder_num_layers", type=int, default=2, help="Number of Layers in the Copula Encoder")
    parser.add_argument("--copula_encoder_num_heads", type=int, default=1, help="Number of Heads in the Copula Encoder")
    parser.add_argument("--copula_encoder_dim", type=int, default=16, help="Embedding Dimension of the Copula Encoder")
    # Attentional Copula Decoder
    parser.add_argument("--decoder_num_layers", type=int, default=1, help="Number of Layers in the Attentional Copula")
    parser.add_argument("--decoder_num_heads", type=int, default=3, help="Number of Heads in the Attentional Copula")
    parser.add_argument("--decoder_dim", type=int, default=8, help="Embedding Dimension of the Attentional Copula")
    parser.add_argument(
        "--decoder_attention_mlp_class",
        type=str,
        default="_simple_linear_projection",
        choices=["_easy_mlp", "_simple_linear_projection"], 
        help="MLP Type to be used in the Attentional Copula"
    )
    # Final layers in the decoder
    parser.add_argument("--decoder_resolution", type=int, default=20, help="Number of bins in the Attentional Copula")
    parser.add_argument("--decoder_mlp_layers", type=int, default=2, help="Number of layers in the final MLP in the Decoder")
    parser.add_argument("--decoder_mlp_dim", type=int, default=48, help="Embedding Dimension of the final MLP in the Decoder")
    parser.add_argument(
        "--decoder_act",
        type=str,
        default="relu",
        choices=["relu", "elu", "glu", "gelu"], 
        help="Activation Function to be used in the Decoder"
    )
    # DSF Marginal
    parser.add_argument("--dsf_num_layers", type=int, default=2, help="Number of layers in the deep sigmoidal flow")
    parser.add_argument("--dsf_dim", type=int, default=48, help="Embedding Dimension of the deep sigmoidal flow")
    parser.add_argument("--dsf_mlp_layers", type=int, default=2, help="Number of layers in the marginal conditioner MLP")
    parser.add_argument("--dsf_mlp_dim", type=int, default=48, help="Embedding Dimension of the marginal conditioner MLP")

    # Loss normalization
    parser.add_argument(
        "--loss_normalization",
        type=str,
        default="both",
        choices=["", "none", "series", "timesteps", "both"],
        help="Loss normalization type"
    )

    # Modify this argument to use interpolation
    parser.add_argument(
        "--experiment_mode",
        type=str,
        choices=["forecasting", "interpolation"],
        default="forecasting",
        help="Operation mode of the model"
    )

    # Don't restrict memory / time
    parser.add_argument("--do_not_restrict_memory", action="store_true", help="When enabled, memory is not restricted to 12 GB")
    parser.add_argument("--do_not_restrict_time", action="store_true", help="When enabled, total training time is not restricted to 3 days")

    # CPU
    parser.add_argument("--use_cpu", action="store_true", default=False, help="When enabled, CPU is used instead of GPU")

    args = parser.parse_args()

    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA version:", torch.version.cuda)

    main(args=args)
