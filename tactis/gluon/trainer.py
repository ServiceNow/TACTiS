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
import gc
import torch.nn as nn
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader

from gluonts.core.component import validated
from tactis.gluon.utils import (
    save_checkpoint,
    load_checkpoint,
    set_seed,
)
from pts import Trainer

from gluonts.dataset.loader import ValidationDataLoader


class TACTISTrainer(Trainer):
    @validated()
    def __init__(
        self,
        epochs: int = 100,
        epochs_phase_1=None,
        epochs_phase_2=None,
        batch_size: int = 32,
        training_num_batches_per_epoch: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        clip_gradient: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        log_subparams_every=-1,
        checkpoint_dir=None,
        seed=42,
        load_checkpoint=None,
        early_stopping_epochs=-1,
        do_not_restrict_time=False,
        skip_batch_size_search=False,
    ) -> None:
        """
        When `epochs_phase_1` and `epochs_phase_2` are not specified, training is done for a total of `epochs` epochs, and `early_stopping_epochs` is used to switch from phase 1 to phase 2.
        When `epochs_phase_1` and `epochs_phase_2` are specified, `early_stopping_epochs` is ignored, and phase 1 and phase 2 are trained for `epochs_phase_1` and `epochs_phase_2` epochs respectively.
        """
        self.epochs = epochs
        self.epochs_phase_1 = epochs_phase_1
        self.epochs_phase_2 = epochs_phase_2
        self.batch_size = batch_size
        self.training_num_batches_per_epoch = training_num_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_gradient = clip_gradient
        self.device = device
        self.log_subparams_every = log_subparams_every
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed
        self.load_checkpoint = load_checkpoint

        self.early_stopping_epochs = early_stopping_epochs
        self.do_not_restrict_time = do_not_restrict_time
        self.skip_batch_size_search = skip_batch_size_search

        if self.checkpoint_dir == None:
            raise Exception("Checkpoint directory (checkpoint_dir) is required to be specified for training.")
        else:
            print("Checkpoints will be saved at", self.checkpoint_dir)
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def set_load_checkpoint(self, load_checkpoint):
        self.load_checkpoint = load_checkpoint

    @staticmethod
    def disable_grads(net, disable_grads):
        for j, p in enumerate(net.model.named_parameters()):
            for param in disable_grads:
                if param in p[0]:
                    p[1].requires_grad = False

    def initialize_stage_1(self, net, optimizer_name, ckpt=None):
        net.model.set_stage(1)
        if optimizer_name == "rmsprop":
            optim = RMSprop(
                net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "adam":
            optim = Adam(
                net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        if ckpt:
            print("Loading model...")
            net.load_state_dict(ckpt["model"])
            print("Loaded model")

            print("Loading optim...")
            optim.load_state_dict(ckpt["optim"])
            print("Loaded optim...")

        return net, optim

    def switch_to_stage_2(self, net, optimizer_name, ckpt=None):
        disable_grads = []

        net.model.set_stage(2)
        net.model.initialize_stage2()
        net.to(self.device)

        # Create parameter groups for the specific parameters
        # A list of parameters related to each component for reference
        params_input_encoder_flow = [
            "flow_series_encoder",
            "flow_time_encoding",
            "flow_input_encoder",
        ]
        params_encoder_flow = ["flow_encoder"]
        params_decoder_flow = ["decoder.marginal"]

        parameter_names_to_optimize = [
            "copula_series_encoder",
            "copula_time_encoding",
            "copula_input_encoder",
            "copula_encoder",
            "decoder.copula",
        ]
        params_to_optimize_in_stage2 = []
        for name, param in net.named_parameters():
            if any(pname in name for pname in parameter_names_to_optimize):
                params_to_optimize_in_stage2.append(param)

        if optimizer_name == "rmsprop":
            optim = RMSprop(
                params_to_optimize_in_stage2,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "adam":
            optim = Adam(
                params_to_optimize_in_stage2,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        if ckpt:
            print("Loading model...")
            net.load_state_dict(ckpt["model"])
            print("Loaded model")

            print("Loading optim...")
            optim.load_state_dict(ckpt["optim"])
            print("Loaded optim...")

        disable_grads.extend(params_decoder_flow)
        disable_grads.extend(params_input_encoder_flow)
        disable_grads.extend(params_encoder_flow)
        TACTISTrainer.disable_grads(net, disable_grads)

        return net, optim

    def __call__(
        self,
        net: nn.Module,
        train_iter: DataLoader,
        validation_iter_args=None,
        optimizer: str = "adam",
    ) -> None:
        set_seed(self.seed)

        # Load checkpoint automatically detecting the stage
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
                net, optim = self.initialize_stage_1(net, optimizer, ckpt)
                current_stage = 1
            else:
                print("Loaded checkpoint has stage 2")
                flow_loss_weight = 0.0
                copula_loss_weight = 1.0
                net, optim = self.switch_to_stage_2(net, optimizer, ckpt)
                current_stage = 2
            start_epoch = ckpt["epoch"] + 1
            print("Start epoch set to", start_epoch)
            best_val_loss_unweighted = ckpt["best_val_loss_unweighted"]
            best_epoch = ckpt["best_epoch"]
            total_training_only_time = ckpt["total_training_only_time"]
        else:
            flow_loss_weight = 1.0
            copula_loss_weight = 0.0
            net, optim = self.initialize_stage_1(net, optimizer)
            current_stage = 1

        ### The following code contains a batch size search.
        ### It can be used to maximize the batch size that one can use with the available GPU memory.
        ### It is however completely optional and may be skipped using the "skip_batch_size_search" argument
        ### If skipped, the batch size can be manually set appropriately
        if not self.skip_batch_size_search:
            print("Performing a batch size search with 10 iterations.")
            net.train()
            batch_size = self.batch_size
            print("Initial batch size:", batch_size)
            iters_since_batch_size = 0
            for batch_no, data_entry in enumerate(train_iter, start=0):
                try:
                    inputs = [v.to(self.device)[:batch_size] for v in data_entry.values()]
                    _ = net(*inputs)
                    # Loss computation
                    marginal_logdet, copula_loss = (
                        net.model.marginal_logdet,
                        net.model.copula_loss,
                    )
                    loss = copula_loss_weight * copula_loss - flow_loss_weight * marginal_logdet
                    loss_avg = loss.mean()
                    loss_avg.backward()
                    iters_since_batch_size += 1
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        gc.collect()
                        torch.cuda.empty_cache()
                        print("Out of memory error encountered. Current batch size is", batch_size)
                        if batch_size == 1:
                            print("Batch is already at the minimum. Cannot reduce further. Exiting...")
                            return None
                        else:
                            print("Caught OutOfMemoryError. Reducing batch size...")
                            batch_size //= 2
                            print("New batch size:", batch_size)
                            iters_since_batch_size = 0
                    else:
                        print(e)
                        exit(1)
                if iters_since_batch_size == 10:
                    break
            print("Using a batch size of:", batch_size)

        # Initialise variables
        step = -1
        best_epoch = -1
        best_val_loss_unweighted = None

        start_epoch = 0
        epoch_no = start_epoch
        batch_size = self.batch_size
        total_training_only_time = 0.0

        ## Iterator for the epochs
        switch_to_stage_2 = False
        if self.epochs_phase_1 and self.epochs_phase_2:
            total_epochs = self.epochs_phase_1 + self.epochs_phase_2
        else:
            total_epochs = self.epochs

        ## Log the start and total epochs
        print("Start epoch:", start_epoch)
        print("Total Epochs:", total_epochs)

        print("Flow loss weight:", flow_loss_weight)
        print("Copula loss weight:", copula_loss_weight)

        while epoch_no < total_epochs:
            # Stage switching
            if switch_to_stage_2:
                print("Switching to stage 2")
                flow_loss_weight = 0.0
                copula_loss_weight = 1.0
                net, optim = self.switch_to_stage_2(net, optimizer)
                # Reset epochs, time
                epochs_since_best_epoch = 0
                total_training_only_time = 0.0
                # Set current stage
                current_stage = 2
                # Reset best validation loss
                best_val_loss_unweighted = None
                # Reset flag
                switch_to_stage_2 = False

            net.train()

            print("Epoch:", epoch_no, "/", total_epochs)
            set_seed(epoch_no + self.seed)
            cumm_epoch_loss = 0.0
            cumm_epoch_loss_unweighted = 0.0
            cumm_epoch_loss_unnormalized = 0.0
            cumm_marginal_logdet = 0.0  # Normalized
            cumm_copula_loss = 0.0  # Normalized

            print("Training.")
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

            avg_epoch_loss_unweighted = cumm_epoch_loss_unweighted / training_num_windows_seen
            print("Epoch:", epoch_no, "Average training loss:", avg_epoch_loss_unweighted)

            ####### VALIDATION #########
            ####### VALIDATION #########
            ####### VALIDATION #########
            if validation_iter_args is not None:
                validation_iter = ValidationDataLoader(**validation_iter_args, batch_size=batch_size)
            validation_length = None
            net.eval()
            cumm_epoch_loss_val = 0.0
            cumm_epoch_loss_val_unweighted = 0.0
            cumm_epoch_loss_val_unnormalized = 0.0
            cumm_marginal_loss_val = 0.0
            cumm_copula_loss_val = 0.0
            set_seed(epoch_no + self.seed)

            validation_num_windows_seen = 0
            validation_num_batches_per_epoch = len(validation_iter)

            print("Validation.")
            print("Total number of validation batches:", validation_num_batches_per_epoch)

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

            avg_epoch_loss_val_unweighted = cumm_epoch_loss_val_unweighted / validation_num_windows_seen

            print("Epoch:", epoch_no, "Average validation loss:", avg_epoch_loss_val_unweighted)

            if best_val_loss_unweighted == None or avg_epoch_loss_val_unweighted < best_val_loss_unweighted:
                best_val_loss_unweighted = avg_epoch_loss_val_unweighted
                best_epoch = epoch_no
                epochs_since_best_epoch = 0
            else:
                epochs_since_best_epoch += 1
            print("Epochs since best epoch:", epochs_since_best_epoch)

            ####################################################
            # All three stage-switching/early stopping criteria are below #
            ####################################################

            # Early stopping based shifting/stopping
            if (
                self.epochs_phase_1 is None
                and self.epochs_phase_2 is None
                and self.early_stopping_epochs != -1
                and epochs_since_best_epoch == self.early_stopping_epochs
            ):
                if net.model.stage == 2:
                    print("Stopping criterion reached for stage 2. Stopping training.")
                    # Load the best model
                    print("Loading the best model so far.")
                    checkpoint_path = os.path.join(self.checkpoint_dir, "best_stage_2.pth.tar")
                    net = load_checkpoint(checkpoint_path, net, self.device)
                    # Break
                    break
                else:
                    print("Stopping criterion reached for stage 1. Shifting to stage 2.")
                    # Load the best model
                    print("Loading the best model so far.")
                    checkpoint_path = os.path.join(self.checkpoint_dir, "best_stage_1.pth.tar")
                    net = load_checkpoint(checkpoint_path, net, self.device)
                    # Set flag
                    switch_to_stage_2 = True
                    # Increase epoch_no here since we are continuing
                    epoch_no += 1
                    # Continue
                    continue

            # Epochs based shifting/stopping
            if (
                net.model.stage == 2
                and self.epochs_phase_2
                and epoch_no - self.epochs_phase_1 == self.epochs_phase_2 - 1
            ):
                print(self.epochs_phase_2, "epochs completed in stage 2. Stopping training.")
                # Load the best model
                print("Loading the best model so far.")
                checkpoint_path = os.path.join(self.checkpoint_dir, "best_stage_2.pth.tar")
                net = load_checkpoint(checkpoint_path, net, self.device)
                # Break
                break
            elif net.model.stage == 1 and self.epochs_phase_1 and epoch_no == self.epochs_phase_1 - 1:
                print(self.epochs_phase_1, "epochs completed in stage 1. Shifting to stage 2.")
                # Load the best model
                print("Loading the best model so far.")
                checkpoint_path = os.path.join(self.checkpoint_dir, "best_stage_1.pth.tar")
                net = load_checkpoint(checkpoint_path, net, self.device)
                # Set flag
                switch_to_stage_2 = True
                # Increase epoch_no here since we are continuing
                epoch_no += 1
                # Continue
                continue

            # Training time based shifting / stopping
            if not self.do_not_restrict_time and total_training_only_time >= 129600:
                if net.model.stage == 2:
                    print("Time limit reached for stage 2. Stopping training.")
                    # Load the best model
                    print("Loading the best model so far.")
                    checkpoint_path = os.path.join(self.checkpoint_dir, "best_stage_2.pth.tar")
                    net = load_checkpoint(checkpoint_path, net, self.device)
                    # Break
                    break
                else:
                    print("Time limit reached for stage 1. Shifting to stage 2.")
                    # Load the best model
                    print("Loading the best model so far.")
                    checkpoint_path = os.path.join(self.checkpoint_dir, "best_stage_1.pth.tar")
                    net = load_checkpoint(checkpoint_path, net, self.device)
                    # Set flag
                    switch_to_stage_2 = True
                    # Increase epoch_no here since we are continuing
                    epoch_no += 1
                    # Continue
                    continue

            ####### VALIDATION #########
            ####### VALIDATION #########
            ####### VALIDATION #########

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
            filename = "last_stage_" + str(current_stage) + ".pth.tar"
            save_checkpoint(state_dict, self.checkpoint_dir, filename=filename)
            print(
                "Checkpoint of epoch",
                epoch_no,
                "saved at",
                os.path.join(self.checkpoint_dir, filename),
            )

            if epochs_since_best_epoch == 0:
                filename = "best_stage_" + str(current_stage) + ".pth.tar"
                save_checkpoint(state_dict, self.checkpoint_dir, filename=filename)
                print(
                    "Checkpoint of epoch",
                    epoch_no,
                    "saved at",
                    os.path.join(self.checkpoint_dir, filename),
                )

            # Increase epoch_no
            epoch_no += 1

            # Print a new line for better readability
            print("\n")

        return best_val_loss_unweighted

    def validate(
        self,
        net: nn.Module,
        validation_iter_args=None,
    ):
        if self.load_checkpoint:
            assert os.path.isfile(self.load_checkpoint), "Checkpoint " + self.load_checkpoint + "is invalid"
            print("Loading from checkpoint", self.load_checkpoint)
            ckpt = torch.load(self.load_checkpoint, map_location=self.device)
            net.load_state_dict(ckpt["model"])

        # Note that the NLL you obtain from this function is only comparable to TACTiS-style architecture
        # To make it comparable to non-TACTiS style architectures, you need to denormalize the loss of every sample/batch which is not currently implemented
        print("Validation...")
        batch_size = self.batch_size

        if validation_iter_args is not None:
            print("Creating a validation dataloader with a batch size of", batch_size)
            validation_iter = ValidationDataLoader(**validation_iter_args, batch_size=batch_size)
        validation_length = None
        net.eval()
        cumm_epoch_loss_val = 0.0

        validation_num_windows_seen = 0
        for batch_no, data_entry in enumerate(validation_iter, start=0):
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
                loss = copula_loss - marginal_logdet

            cumm_epoch_loss_val += loss.sum()

        avg_epoch_loss_val = cumm_epoch_loss_val / validation_num_windows_seen

        return avg_epoch_loss_val
