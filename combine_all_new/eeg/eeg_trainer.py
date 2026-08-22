"""
EEG 訓練 Pipeline。

對應原 EEG_DL/EEG/EEG_Train.py。
包裝 BraindecodeTrainer，提供 train / fine-tune 的高階介面。
"""
import os
import numpy as np
import random
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from braindecode.models import ShallowFBCSPNet

from eeg.eeg_model import EEGNet_our_ver, load_shallowfbcsp_params, load_EEGNet_our_ver_params
from eeg.braindecode_trainer import BraindecodeTrainer

import config
import global_state
import eeg.lsl_utils as lsl_utils


class EEGTrainingPipeline:
    def __init__(self, epochs=150, freeze_layers=False, ft=False, tcp_client=None):
        self.tcp_client = tcp_client
        self.epochs = epochs
        self.freeze_layers = freeze_layers
        self.ft = ft

        self.batch_size = 8
        self.lr = 1e-5
        self.seed = 42
        self.patience = 30

    def _run_braindecode_training(self,
                                  model_class, model_kwargs,
                                  dataset, dataset_valid,
                                  save_path=None):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        trainer = BraindecodeTrainer(
            model_class=model_class,
            model_kwargs=model_kwargs,
            dataset=dataset,
            val_dataset=dataset_valid,
            batch_size=self.batch_size,
            num_epochs=self.epochs,
            lr=self.lr,
            ft=self.ft,
            tcp_client=self.tcp_client,
        )

        name = "train"

        if self.ft is True:
            trainer.load_checkpoint(global_state.NOW_FT_CHECKPOINT + ".pth")
            name = "ft"

        hist = trainer.train(freeze_layer=self.freeze_layers, patience=self.patience)

        best_loss_epoch = np.argmin(hist["val_loss"])
        best_loss_epoch_path = os.path.join(
            global_state.EEG_CHECKPOINT_TMP_BASE_DIR,
            f"{name}-epoch{best_loss_epoch}.pth"
        )

        val_acc = hist['val_acc'][best_loss_epoch]
        val_loss = hist['val_loss'][best_loss_epoch]

        if save_path:
            shutil.copyfile(best_loss_epoch_path, save_path)

        print(f"\n{config.TAGS.INFO.value} Calibration done, "
              f"copy model {name}-epoch{best_loss_epoch}.pth to the {save_path}")
        print(f"[DEBUG] val_acc={val_acc:.4f}, val_loss={val_loss:.4f}")

        return val_acc, val_loss

    def run_training(self,
                     dataset,
                     dataset_val,
                     save_path,
                     model_class="EEGNet_our_ver"):

        if model_class == "EEGNet_our_ver":
            model_kwargs = load_EEGNet_our_ver_params(dataset)
            if model_kwargs is None:
                raise RuntimeError(f"please set the model_kwargs of the {model_class}")
            val_acc, val_loss = self._run_braindecode_training(
                EEGNet_our_ver, model_kwargs, dataset, dataset_val, save_path=save_path)

        elif model_class == "shallownet":
            model_kwargs = load_shallowfbcsp_params(dataset)
            if model_kwargs is None:
                raise RuntimeError(f"please set the model_kwargs of the {model_class}")
            val_acc, val_loss = self._run_braindecode_training(
                ShallowFBCSPNet, model_kwargs, dataset, dataset_val, save_path=save_path)

        return val_acc, val_loss


# ─── Debug ───
if __name__ == "__main__":
    print("=== eeg.eeg_trainer standalone test ===")
    pipeline = EEGTrainingPipeline(epochs=2, ft=False)
    print(f"epochs     = {pipeline.epochs}")
    print(f"batch_size = {pipeline.batch_size}")
    print(f"lr         = {pipeline.lr}")
    print("✅ eeg_trainer OK")
