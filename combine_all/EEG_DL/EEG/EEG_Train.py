"""
用於 Calibration 場景 蒐集出來的資料做 fine-tune
和 training


tsai: 2026 / 01 / 26 修正註解
"""

import os
import numpy as np
import random
import shutil                                                                       # use for copy model




import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset


from braindecode.models import ShallowFBCSPNet

from EEG_DL.EEG.MI_train import BraindecodeTrainer, load_shallowfbcsp_params, load_EEGNet_our_ver_params, EEGNet_our_ver



import EEG_DL.Utils.config as config

import EEG_DL.Utils.global_value as global_value
import EEG_DL.Utils.LSL as LSL





class EEGTrainingPipeline:

    def __init__(self, epochs=150, freeze_layers=False, ft=False, tcp_client=None):

        self.tcp_client = tcp_client

        self.epochs = epochs
        self.freeze_layers = freeze_layers
        self.ft = ft

        self.batch_size = 8
        self.lr = 1e-5
        self.seed = 42

        # patience 代表 early stop 的判斷次數
        self.patience = 30
        

    def _run_braindecode_training(self,
                                 model_class, model_kwargs,
                                 dataset, dataset_valid,
                                 save_path=None # .pth
                                 ):

        # print(f"seed is {self.seed}")
        # print(f"model {self.model_class}, epochs {self.epochs}, batch_size {self.batch_size}, lr {self.lr}, freeze_layers {self.freeze_layers}")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        trainer = BraindecodeTrainer(
            model_class=model_class,            # 使用模型
            model_kwargs=model_kwargs,          # 對應模型參數
            dataset=dataset,
            val_dataset=dataset_valid,
            
            batch_size=self.batch_size,
            num_epochs=self.epochs,
            lr=self.lr,
            ft=self.ft,

            tcp_client=self.tcp_client
        )

        name = "train"

        # fine-tune
        if self.ft is True:                  
            trainer.load_checkpoint(global_value.NOW_FT_CHECKPOINT+".pth")
            name = "ft"

        # patience 代表 early stop 的判斷次數
        hist = trainer.train(freeze_layer=self.freeze_layers,
                             patience=self.patience)
        
        best_loss_epoch = np.argmin(hist["val_loss"])
        best_loss_epoch_path = os.path.join(global_value.EEG_CHECKPOINT_TMP_BASE_DIR, f"{name}-epoch{best_loss_epoch}.pth")
        
        # 取 valid set acc 和 loss
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

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ EEGNet

        if model_class == "EEGNet_our_ver":
            model_kwargs = load_EEGNet_our_ver_params(dataset)

            if model_kwargs is None:
                raise RuntimeError(f"please set the model_kwargs of the {model_class}")

            val_acc, val_loss = self._run_braindecode_training(EEGNet_our_ver, model_kwargs,
                                          dataset, dataset_val,
                                          save_path=save_path,
                                        )

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ shallownet

        
        
        elif model_class == "shallownet":
            model_kwargs = load_shallowfbcsp_params(dataset)

            if model_kwargs is None:
                raise RuntimeError(f"please set the model_kwargs of the {model_class}")

            val_acc, val_loss = self._run_braindecode_training(ShallowFBCSPNet, model_kwargs,
                                        dataset, dataset_val, 
                                        save_path=save_path,
                                        )
        
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 

        return val_acc, val_loss


def main():
    print("You are in EEG_Train, do nothing")
    return



if __name__ == "__main__":
    main()
