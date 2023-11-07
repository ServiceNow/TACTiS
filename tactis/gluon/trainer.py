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

import time
from typing import Optional, Union

import torch
import os
import torch.nn as nn
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader

from gluonts.core.component import validated
from tactis.gluon.utils import (
    DummyLogger,
    save_checkpoint,
    set_seed,
)
from pts import Trainer

from gluonts.dataset.loader import ValidationDataLoader


class TACTISTrainer(Trainer):
    @validated()
    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        training_num_batches_per_epoch: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        clip_gradient: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        log_subparams_every=-1,
        checkpoint_path=None,
        seed=42,
        load_checkpoint=None,
        logger=None,
        early_stopping_epochs=-1,
        do_not_restrict_time=False,
        **kwargs,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.training_num_batches_per_epoch = training_num_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_gradient = clip_gradient
        self.device = device
        self.log_subparams_every = log_subparams_every
        self.checkpoint_path = checkpoint_path
        self.seed = seed
        self.load_checkpoint = load_checkpoint
        self.logger = logger

        self.early_stopping_epochs = early_stopping_epochs
        self.do_not_restrict_time = do_not_restrict_time

        if self.checkpoint_path == None:
            print("WARNING: Checkpoints will not be saved")
        else:
            print("Checkpoints will be saved at", self.checkpoint_path)
            os.makedirs(self.checkpoint_path, exist_ok=True)

    def set_load_checkpoint(self, load_checkpoint):
        self.load_checkpoint = load_checkpoint

    def set_logger(self, logger):
        self.logger = logger

    def disable_grads(self, net, disable_grads):
        total_enabled = 0
        total_disabled = 0
        total_grads = 0
        for j, p in enumerate(net.model.named_parameters()):
            for param in disable_grads:
                if param in p[0]:
                    print("Disabling gradient on", p[0])
                    p[1].requires_grad = False
            if p[1].requires_grad == True:
                total_enabled += 1
            if p[1].requires_grad == False:
                total_disabled += 1
            total_grads += 1

        print("Grads to disable this epoch:", disable_grads)
        print("Total disabled grads:", total_disabled, "/", total_grads)
        print("Total enabled grads:", total_enabled, "/", total_grads)

    def __call__(
        self,
        net: nn.Module,
        train_iter: DataLoader,
        validation_iter: Optional[DataLoader] = None,
        validation_iter_args=None,
        optimizer: str = "adam",
    ) -> None:
        set_seed(self.seed)
        if not self.logger:
            self.logger = DummyLogger()

        # A list of parameters related to each component for reference
        params_input_encoder_flow = [
            "flow_series_encoder",
            "flow_time_encoding",
            "flow_input_encoder",
        ]
        params_encoder_flow = ["flow_encoder"]
        params_decoder_flow = ["decoder.marginal"]

        disable_grads = []

        if self.load_checkpoint:
            set_seed(self.seed)
            assert os.path.isfile(self.load_checkpoint), "Checkpoint " + self.load_checkpoint + "is invalid"
            print("Loading from checkpoint", self.load_checkpoint)
            ckpt = torch.load(self.load_checkpoint, map_location=self.device)
            if ckpt["stage"] == 1:
                print("Loaded checkpoint has stage 1")
                net.model.set_stage(1)
                flow_loss_weight = 1.0
                copula_loss_weight = 0.0
                print("Loading model...")
                net.load_state_dict(ckpt["model"])
                print("Loaded model")
                if optimizer == "rmsprop":
                    optim = RMSprop(
                        net.parameters(),
                        lr=self.learning_rate,
                        weight_decay=self.weight_decay,
                    )
                elif optimizer == "adam":
                    optim = Adam(
                        net.parameters(),
                        lr=self.learning_rate,
                        weight_decay=self.weight_decay,
                    )
                print("Loading optim...")
                optim.load_state_dict(ckpt["optim"])
                print("Loaded optim...")
            else:
                print("Loaded checkpoint has stage 2")
                net.model.set_stage(2)
                net.model.initialize_stage2()
                net.to(self.device)

                # Create parameter groups for the specific parameters
                parameter_names_to_optimize = [
                    "copula_series_encoder",
                    "copula_time_encoding",
                    "copula_input_encoder",
                    "copula_encoder",
                    "decoder.copula",
                ]
                params_to_optimize_in_stage2 = []
                for name, param in net.model.named_parameters():
                    if any(pname in name for pname in parameter_names_to_optimize):
                        params_to_optimize_in_stage2.append(param)

                flow_loss_weight = 0.0
                copula_loss_weight = 1.0
                print("Loading model...")
                net.load_state_dict(ckpt["model"])
                print("Loaded model")
                if optimizer == "rmsprop":
                    optim = RMSprop(
                        params_to_optimize_in_stage2,
                        lr=self.learning_rate,
                        weight_decay=self.weight_decay,
                    )
                elif optimizer == "adam":
                    optim = Adam(
                        params_to_optimize_in_stage2,
                        lr=self.learning_rate,
                        weight_decay=self.weight_decay,
                    )
                print("Loading optim...")
                optim.load_state_dict(ckpt["optim"])
                print("Loaded optim...")

                disable_grads.extend(params_decoder_flow)
                disable_grads.extend(params_input_encoder_flow)
                disable_grads.extend(params_encoder_flow)
                self.disable_grads(net, disable_grads)

            start_epoch = ckpt["epoch"] + 1
            print("Start epoch set to", start_epoch)
            best_val_loss_unweighted = ckpt["best_val_loss_unweighted"]
            best_epoch = ckpt["best_epoch"]
            total_training_only_time = ckpt["total_training_only_time"]
        else:
            net.model.set_stage(1)
            flow_loss_weight = 1.0
            copula_loss_weight = 0.0
            if optimizer == "rmsprop":
                optim = RMSprop(
                    net.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )
            elif optimizer == "adam":
                optim = Adam(
                    net.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )

        step = -1
        best_epoch = -1

        # epoch:value dictionary
        best_val_loss_unweighted = None

        start_epoch = 0
        epoch_no = start_epoch
        batch_size = self.batch_size
        total_training_only_time = 0.0

        ## Log the start and total epochs
        print("Start epoch:", start_epoch)
        print("Total epochs:", self.epochs)
        print("Epochs:", self.epochs)

        print("Flow loss weight:", flow_loss_weight)
        print("Copula loss weight:", copula_loss_weight)

        ## Iterator for the epochs
        switch_to_stage_2 = False
        while epoch_no < self.epochs:
            # Stage switching
            if switch_to_stage_2:
                net.model.set_stage(2)
                net.model.initialize_stage2()
                net.to(self.device)

                # Create parameter groups for the specific parameters
                parameter_names_to_optimize = [
                    "copula_series_encoder",
                    "copula_time_encoding",
                    "copula_input_encoder",
                    "copula_encoder",
                    "decoder.copula",
                ]
                params_to_optimize_in_stage2 = []
                for name, param in net.model.named_parameters():
                    if any(pname in name for pname in parameter_names_to_optimize):
                        params_to_optimize_in_stage2.append(param)

                flow_loss_weight = 0.0
                copula_loss_weight = 1.0
                if optimizer == "rmsprop":
                    optim = RMSprop(
                        params_to_optimize_in_stage2,
                        lr=self.learning_rate,
                        weight_decay=self.weight_decay,
                    )
                elif optimizer == "adam":
                    optim = Adam(
                        params_to_optimize_in_stage2,
                        lr=self.learning_rate,
                        weight_decay=self.weight_decay,
                    )

                disable_grads.extend(params_decoder_flow)
                disable_grads.extend(params_input_encoder_flow)
                disable_grads.extend(params_encoder_flow)

                self.disable_grads(net, disable_grads)

                epochs_since_best_epoch = 0
                total_training_only_time = 0.0
                switch_to_stage_2 = False

            net.train()

            print("Epoch:", epoch_no, "/", self.epochs)
            print("Training...")
            set_seed(epoch_no + self.seed)
            cumm_epoch_loss = 0.0
            cumm_epoch_loss_unweighted = 0.0
            cumm_epoch_loss_unnormalized = 0.0
            cumm_marginal_logdet = 0.0  # Normalized
            cumm_copula_loss = 0.0  # Normalized

            print("Total number of training batches:", self.training_num_batches_per_epoch)

            # training loop
            training_num_windows_seen = 0
            training_epoch_start_time = time.time()
            for batch_no, data_entry in enumerate(train_iter, start=0):
                step = epoch_no * self.training_num_batches_per_epoch + batch_no
                if self.log_subparams_every != -1 and step % self.log_subparams_every == 0:
                    print("Iter:", batch_no, "/", self.training_num_batches_per_epoch)

                optim.zero_grad()

                inputs = [v.to(self.device)[:batch_size] for v in data_entry.values()]
                training_num_windows_seen += len(inputs[0])

                # Forward pass
                _ = net(*inputs)
                # Loss computation
                marginal_logdet, copula_loss = (
                    net.model.marginal_logdet,
                    net.model.copula_loss,
                )
                loss = copula_loss_weight * copula_loss - flow_loss_weight * marginal_logdet
                loss_avg = loss.mean()
                # Backward pass
                loss_avg.backward()
                if self.clip_gradient is not None:
                    nn.utils.clip_grad_norm_(net.parameters(), self.clip_gradient)
                optim.step()

                cumm_epoch_loss += loss.sum()
                cumm_epoch_loss_unweighted += torch.sum(copula_loss - marginal_logdet).item()
                cumm_epoch_loss_unnormalized += torch.sum(
                    net.model.unnormalized_copula_loss - net.model.marginal_logdet
                ).item()
                cumm_marginal_logdet += -torch.sum(marginal_logdet).item()
                cumm_copula_loss += torch.sum(copula_loss).item()

                if self.training_num_batches_per_epoch == batch_no:
                    break

            # Accumulate training time
            training_epoch_end_time = time.time()
            total_training_only_time += training_epoch_end_time - training_epoch_start_time

            avg_epoch_loss = cumm_epoch_loss / training_num_windows_seen
            print("Epoch:", epoch_no, "Average training loss:", avg_epoch_loss)

            ####### VALIDATION #########
            ####### VALIDATION #########
            ####### VALIDATION #########
            print("Validation...")
            if validation_iter is None and validation_iter_args is not None:
                print("Creating a validation dataloader with a batch size of", batch_size)
                validation_iter = ValidationDataLoader(**validation_iter_args, batch_size=batch_size)
            validation_length = None
            net.eval()
            cumm_epoch_loss_val = 0.0
            cumm_epoch_loss_val_unweighted = 0.0
            cumm_epoch_loss_val_unnormalized = 0.0
            cumm_marginal_loss_val = 0.0
            cumm_copula_loss_val = 0.0
            set_seed(
                epoch_no + self.seed
            )  # Change the validation set each epoch to ensure stability of minimum obtained through early stopping

            validation_num_windows_seen = 0
            validation_num_batches_per_epoch = len(validation_iter)
            for batch_no, data_entry in enumerate(validation_iter, start=0):
                if self.log_subparams_every != -1 and step % self.log_subparams_every == 0:
                    print("Iter:", batch_no, "/", validation_num_batches_per_epoch)
                inputs = [v.to(self.device) for v in data_entry.values()]
                validation_num_windows_seen += len(inputs[0])
                if validation_length:
                    inputs = [v[:, :validation_length] for v in inputs]
                with torch.no_grad():
                    _ = net(*inputs)

                    marginal_logdet, copula_loss = (
                        net.model.marginal_logdet,
                        net.model.copula_loss,
                    )
                    loss = copula_loss_weight * copula_loss - flow_loss_weight * marginal_logdet
                    loss_avg = loss.mean()

                cumm_epoch_loss_val += loss.sum()
                cumm_epoch_loss_val_unweighted += torch.sum(copula_loss - marginal_logdet).item()
                cumm_epoch_loss_val_unnormalized += torch.sum(
                    net.model.unnormalized_copula_loss - net.model.marginal_logdet
                ).item()
                cumm_marginal_loss_val += -torch.sum(marginal_logdet).item()
                cumm_copula_loss_val += torch.sum(copula_loss).item()

            avg_epoch_loss_val = cumm_epoch_loss_val / validation_num_windows_seen
            avg_epoch_loss_val_unweighted = cumm_epoch_loss_val_unweighted / validation_num_windows_seen

            print("Epoch:", epoch_no, "Average validation loss:", avg_epoch_loss_val)

            if best_val_loss_unweighted == None or avg_epoch_loss_val_unweighted < best_val_loss_unweighted:
                best_val_loss_unweighted = avg_epoch_loss_val_unweighted
                best_epoch = epoch_no
                epochs_since_best_epoch = 0
            else:
                epochs_since_best_epoch += 1

            print("Epochs since best epoch:", epochs_since_best_epoch)

            # If in stage 2, stop. If in stage 1, switch to stage 2 (taken care of in the )
            if self.early_stopping_epochs != -1 and epochs_since_best_epoch == self.early_stopping_epochs:
                if net.model.stage == 2:
                    print("Stopping criterion reached for stage 2. Stopping training.")
                    break
                else:
                    print("Stopping criterion reached for stage 1. Shifting to stage 2.")
                    switch_to_stage_2 = True

            ####### VALIDATION #########
            ####### VALIDATION #########
            ####### VALIDATION #########

            if self.checkpoint_path:
                state_dict = {
                    "model": net.state_dict(),
                    "epoch": epoch_no,
                    "step": step,
                    "best_val_loss_unweighted": best_val_loss_unweighted,
                    "best_epoch": best_epoch,
                    "stage": net.model.stage,
                    "total_training_only_time": total_training_only_time,
                }
                state_dict["optim"] = optim.state_dict()
                filename = "last.pth.tar"
                save_checkpoint(state_dict, self.checkpoint_path, filename=filename)
                print(
                    "Checkpoint of epoch",
                    epoch_no,
                    "saved at",
                    os.path.join(self.checkpoint_path, filename),
                )

                if epochs_since_best_epoch == 0:
                    filename = "best.pth.tar"
                    save_checkpoint(state_dict, self.checkpoint_path, filename=filename)
                    print(
                        "Checkpoint of epoch",
                        epoch_no,
                        "saved at",
                        os.path.join(self.checkpoint_path, filename),
                    )

            # Check if the training time criterion is reached
            if not self.do_not_restrict_time and total_training_only_time >= 129600:
                if net.model.stage == 2:
                    print("Time limit reached for stage 2. Stopping training.")
                    break
                else:
                    print("Time limit reached for stage 1. Shifting to stage 2.")
                    switch_to_stage_2 = True

            # Increase epoch_no
            epoch_no += 1

        return best_val_loss_unweighted
