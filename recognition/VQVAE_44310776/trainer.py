"""
Generic PyTorch model trainer I wrote. Subclass and override the step function to use.
A python package of this script can be found on my GitHub: https://github.com/lion-software/pytorch-trainer
"""


from __future__ import annotations
from typing import List, Dict, Callable, Tuple
import os
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch import Tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm

class Trainer(ABC):
    def __init__(self, 
                 model: nn.Module, 
                 train_loader: DataLoader, 
                 val_loader: DataLoader, 
                 optimizer: Optimizer,
                 device: str = "cpu",
                 ddp: bool = False,
                 rank: int = None
                 ) -> Trainer:
                 
        self._model = model
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._optimizer = optimizer
        self._device = device

        if ddp and rank == None:
            raise ValueError("DistributedDataParallel enabled, but no rank supplied to Trainer. Please set rank when initializing the Trainer.")

        self._ddp = ddp
        self._rank = rank

        self._train_losses, self._train_accuracy = {}, []
        self._val_losses, self._val_accuracy = {}, []
        self._model_state = []

        self.tqdm = tqdm
        self.tqdm_kwargs = {}

        self._callbacks = {}

    def train(self, epochs: int, quiet: bool = False) -> None:
        disable_tqdm = False
        if (quiet or (self._rank is not None and self._rank != 0)):
            disable_tqdm = True
        for i in self.tqdm(range(epochs), disable=disable_tqdm, position=0, **self.tqdm_kwargs):
            if self._ddp:
                self._train_loader.sampler.set_epoch(i)
            if i in self._callbacks:
                for callback in self._callbacks[i]:
                    args = self._callbacks[i][callback]["args"]
                    kwargs = self._callbacks[i][callback]["kwargs"]
                    callback(*args, **kwargs)
            train_epoch_losses, train_epoch_accuracy = self.fit(disable_tqdm=disable_tqdm)
            val_epoch_losses, val_epoch_accuracy = self.validate(disable_tqdm=disable_tqdm)
            for k, v in train_epoch_losses.items():
                if k not in self._train_losses:
                    self._train_losses[k] = []
                self._train_losses[k].append(v)
            self._train_accuracy.append(train_epoch_accuracy)
            for k, v in val_epoch_losses.items():
                if k not in self._val_losses:
                    self._val_losses[k] = []
                self._val_losses[k].append(v)
            self._val_accuracy.append(val_epoch_accuracy)
            self._model_state.append(self._model.state_dict())

    def fit(self, disable_tqdm: bool = False) -> Tuple[Dict[str, float], float]:
        # Set model to training mode.
        self._model.train()
        running_loss = {}
        running_correct = 0
        counter = 0
        total = 0
        batches = enumerate(self._train_loader)
        for i, batch in self.tqdm(batches, total=len(self._train_loader), 
                                  disable=disable_tqdm, position=1, leave=False):
            counter += 1
            total += batch[0].size(0)
            self._optimizer.zero_grad()
            loss, correct = self.step(batch)
            for k in loss:
                if k not in running_loss:
                    running_loss[k] = 0.0
                running_loss[k] += loss[k].item()
            running_correct += correct
            torch.stack(list(loss.values())).sum().backward()
            self._optimizer.step()

        loss = {k: v / counter for k, v in running_loss.items()}
        accuracy = 100. * running_correct / total
        return loss, accuracy


    def validate(self, disable_tqdm: bool = False) -> None:
        # Set model to evaluation mode.
        self._model.eval()
        running_loss = {}
        running_correct = 0
        counter = 0
        total = 0
        batches = enumerate(self._val_loader)
        with torch.no_grad():
            for i, batch in self.tqdm(batches, total=len(self._val_loader), 
                                  disable=disable_tqdm, position=1, leave=False):
                counter += 1
                total += batch[0].size(0)
                loss, correct = self.step(batch)
                for k in loss:
                    if k not in running_loss:
                        running_loss[k] = 0.0
                    running_loss[k] += loss[k].item()
                running_correct += correct

            loss = {k: v / counter for k, v in running_loss.items()}
            accuracy = 100. * running_correct / total
            return loss, accuracy

    @abstractmethod
    def step(self, batch: Tensor) -> Tuple(Tensor, int):
        """
        Override this method for the model being trained.
        Template step method {
            data, target = batch[0].to(self._device), batch[1].to(self._device)
            outputs = self._model(data)
            loss = self._loss_fn(outputs, target)
            predictions = self._pred_fn(outputs)
            correct = (predictions == target).sum().item()

            return loss, correct
        }
        """
        pass

    ####################
    # Save methods
    ####################

    def save_model(self, path) -> None:
        if self._ddp:
            torch.save(self._model.module.state_dict(), path)
        else:
            torch.save(self._model.state_dict(), path)

    def plot_loss(self, save_path: str = None, quiet: bool = False, smooth: bool = False, yscale: str = 'linear') -> None:
        for k in self._train_losses:
            train_loss = self._train_losses[k]
            val_loss = self._val_losses[k]
            if smooth:
                train_loss = savgol_filter(train_loss, 201, 7)
                val_loss = savgol_filter(val_loss, 201, 7)

            plt.figure(figsize=(10, 7))
            plt.grid(axis='y', color='gray', linestyle='solid')
            plt.plot(train_loss, color='blue', label='train loss')
            plt.plot(val_loss, color='red', label='validataion loss')
            plt.yscale(yscale)
            plt.tick_params(axis='y', which='minor')
            plt.title(f"Training and Validation {k} Loss")
            plt.xlabel('Epochs')
            plt.ylabel(f"{k} Loss")
            plt.legend()
            if save_path:
                plt.savefig(os.path.join(save_path, f"loss_{k}.png"))
            if not quiet:
                plt.show()
            plt.close()
    
    def plot_accuracy(self, save_path: str = None, quiet: bool = False, smooth: bool = False) -> None:
        train_acc = self._train_accuracy
        val_acc = self._val_accuracy
        if smooth:
            train_acc = savgol_filter(train_acc, 21, 5)
            val_acc = savgol_filter(val_acc, 21, 5)
        
        plt.figure(figsize=(10, 7))
        plt.grid(axis='y', color='gray', linestyle='solid')
        plt.plot(train_acc, color='blue', label='train accuracy')
        plt.plot(val_acc, color='red', label='validataion accuracy')
        plt.title("Training and Validation Classification Accuracy")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        if save_path:
            plt.savefig(os.path.join(save_path, f"acc.png"))
        if not quiet:
            plt.show()
        plt.close()

    def set_callback(self, fn: Callable, epochs: List[int], *args, **kwargs) -> None:
        """
        Set fn to be called at the beginning of the specified epochs.
        """
        for i in epochs:
            if i not in self._callbacks:
                self._callbacks[i] = {}
            self._callbacks[i][fn] = {"args": args, "kwargs": kwargs}