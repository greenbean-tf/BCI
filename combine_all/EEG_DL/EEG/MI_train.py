"""
20251015 從之前的 MI_train 改過來，把多餘內容刪除


tsai: 用於 pipeline 中
"""
import os
import time
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


from braindecode.models import ShallowFBCSPNet
import EEG_DL.Utils.config as config
import EEG_DL.Utils.global_value as global_value

class EEGNet_our_ver(nn.Module):
    def __init__(self, 
                    n_chans=None,
                    n_outputs=None,
                    n_times=None,
                    F1=4, 
                    D=2, 
                    F2=8, 
                    T_kernSizes=[32, 64, 128, 200, 256, 512], 
                    S_kernSizes=[], 
                    dropout=0.55
                    ):

        super(EEGNet_our_ver, self).__init__()
        ################################################################################
        # 1. Temporal Conv
        self.T_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, F1, (1, k), padding=(0, k//2), bias=False),
                nn.BatchNorm2d(F1),
                nn.ELU()
            )
            for k in T_kernSizes
        ])
        ## 將多kernel concat在channel維度上，因此總輸出通道數 = F1 * len(kernSizes)
        F1 = F1 * len(T_kernSizes)

        ################################################################################
        # 2. Spatial Conv

        # S multi kernel
        S_kernSizes = [n_chans, n_chans//2+1, 3]  # 先固定用這2, n_chans//2 再說
        # print(f"S_kernSizes: {S_kernSizes}")

        self.S_kernSizes = S_kernSizes
        
        self.S_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(F1, F1 * D, (ks, 1), groups=F1, padding=(ks//2, 0), bias=False),
                nn.BatchNorm2d(F1 * D),
                nn.ELU()
            )
            for ks in S_kernSizes
        ])
        # concat 後的總通道數
        # F1D_total = F1 * D * len(S_kernSizes)
        F1 = F1 * len(S_kernSizes)


        # 3. pooling + dropout
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        
        # 4. Separable Conv
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )

        # flatten feature_dim
        self.feature_dim = self._get_feature_dim(n_chans, n_times)

        # classifier
        self.classifier = nn.Linear(self.feature_dim, n_outputs)
        



    def _get_feature_dim(self, n_chans, n_times):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_chans, n_times)
            
            # 1. T conv
            x = torch.cat([b(x) for b in self.T_branches], dim=1)

            # 2. S conv
            '''
            x = self.depthwise(x)
            x = self.bn2(x)
            '''
            x = torch.cat([b(x) for b in self.S_branches], dim=1)
            


            # 3. pool + dropout
            x = self.pool1(x)
            x = self.dropout1(x)

            # 4. Separable
            x = self.separable(x)
            x = x.flatten(1)
        
        return x.shape[1]

    def forward(self, x):
        # x: (batch, 1, n_chans, n_times)

        if x.dim() == 3:
            x = x.unsqueeze(1)  # -> (B, 1, C, T)

        # 1. T conv
        x = torch.cat([b(x) for b in self.T_branches], dim=1)

        # 2. S conv
        """
        x = self.depthwise(x)
        x = self.bn2(x)
        """
        x = torch.cat([b(x) for b in self.S_branches], dim=1)  
        x = F.elu(x)


        # 3. pool + dropout
        x = self.pool1(x)
        x = self.dropout1(x)

        # 4. Separable
        x = self.separable(x)

        x = x.flatten(1)
        return self.classifier(x)
    

class BraindecodeTrainer:
    def __init__(self,
                model_class, model_kwargs,
                dataset, val_dataset,

                batch_size, 
                num_epochs, 
                lr,
                ft,

                tcp_client
                ):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f"device: {self.device}")

        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.dataset = dataset
        self.val_dataset = val_dataset


        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.ft = ft

        self.tcp_client = tcp_client
        
        self._load_data()
        self._init_model()

    def _load_data(self):
        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)


    def _init_model(self):
        """初始化可替換的 Braindecode 模型"""
        self.model = self.model_class(**self.model_kwargs).to(self.device)
        # summary(self.model)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))


    def load_checkpoint(self, checkpoint_path):

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        print(f"\n{config.TAGS.INFO.value} Loaded checkpoint from {checkpoint_path}")

    def _evaluate(self, loader):
        self.model.eval()
        total_correct, total_samples, losses = 0, 0, []

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device).float(), y_batch.to(self.device).float()
                
                # 因為輸入的維度是 4 維 (之前自己寫的 class 有多一維)，所以這邊要減 1 維
                output = self.model(x_batch.squeeze(1)) 

                # output = self.model(x_batch)
                loss = self.criterion(output, y_batch)
                losses.append(loss.item())
                preds = output.argmax(dim=1)
                total_correct += (preds == y_batch.argmax(dim=1)).sum().item()
                total_samples += y_batch.size(0)

        return np.mean(losses), total_correct / total_samples

    def train(self, freeze_layer=False, patience=10):

        
        history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

        if self.ft:
            tag = "FT"
        else:
            tag = "Train"

        # --- Early Stopping 初始化 ---
        best_val_loss = float('inf')        # 初始最佳 loss 為無限大
        early_stop_counter = 0              # 計數器

        # ---------------------------
        # 只訓練 全連接層 fine tune
        if freeze_layer:  
            if hasattr(self.model, 'classifier'):
                final_layer = self.model.classifier

            elif hasattr(self.model, 'final_layer'):
                final_layer = self.model.final_layer

            else:
                raise AttributeError("Model does not have 'classifier' or 'final_layer' attribute.")
            
            # for name, param in self.model.named_parameters():
            #     if "patch_embedding" in name or "transformer" in name:
            #         param.requires_grad = False
            #         print(f"Froze layer: {name}")

            for param in self.model.parameters():
                param.requires_grad = False

            for param in final_layer.parameters():
                param.requires_grad = True

            # filter(lambda p: p.requires_grad, model.parameters())
            self.optimizer = torch.optim.Adam(final_layer.parameters(), lr=self.lr, betas=(0.5, 0.999))


        epochBar = tqdm(range(self.num_epochs), desc=f"[{tag}] Training", unit="epoch")
        for epoch in epochBar:
            self.model.train()

            # --- 如果是 Fine-tuning，強制鎖定 BatchNorm ---
            if freeze_layer:
                
                # 即使在 model.train() 下，也要讓 BN 保持在 eval 模式，running_mean/var 才不會被新資料洗掉
                for module in self.model.modules():

                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
                    else:
                        # print(f"Skipping layer: ({type(module).__name__})")
                        continue

            running_loss, correct, total = 0.0, 0, 0

            batchBar = tqdm(self.train_loader, desc=f"  Epoch {epoch+1}/{self.num_epochs}", 
                           leave=False, unit="batch")
            for inputs, labels in batchBar:
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float()

                self.optimizer.zero_grad()
                outputs = self.model(inputs.squeeze(1))

                # outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.argmax(dim=1)).sum().item()

                # 更新 batch progress bar
                batchBar.set_postfix(loss=f"{loss.item():.4f}")
            

            train_loss = running_loss / len(self.dataset)
            train_acc = correct / total
            val_loss, val_acc = self._evaluate(self.val_loader)

            history['loss'].append(train_loss)
            history['acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': train_loss,
            }, os.path.join(global_value.EEG_CHECKPOINT_TMP_BASE_DIR, f"{tag.lower()}-epoch{epoch}.pth"))
            
            # 更新 epoch progress bar
            epochBar.set_postfix(
                loss=f"{train_loss:.4f}", acc=f"{train_acc:.4f}",
                val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}"
            )

            if self.tcp_client:
                self.tcp_client.send_messages(f"training{config.TCP_STR_SPLIT}"
                                                f"Epoch {epoch}/{self.num_epochs} - "
                                                f"loss: {train_loss:.4f}, acc: {train_acc:.4f}, "
                                                f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
            


            # --- Early Stopping 邏輯 ---

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0  # 重置計數器
            else:
                early_stop_counter += 1

            # 觸發 Early Stopping
            if early_stop_counter >= patience:

                stop_msg = f"Early stopping triggered at epoch {epoch}. Best val_loss: {best_val_loss:.4f}"
                print(stop_msg)

                break

            """
            print(f"[{tag}] Epoch {epoch}/{self.num_epochs} - "
                  f"loss: {train_loss:.4f}, acc: {train_acc:.4f}, "
                  f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
            """

        time.sleep(0.5)  # 讓系統可以判斷

        return history







def load_shallowfbcsp_params(dataset):
    data_shape = [dataset.__getitem__(0)[0].shape, dataset.__getitem__(0)[1].shape]

    # model input info
    n_channels = data_shape[0][1]
    input_window_samples = data_shape[0][2]
    n_classes = data_shape[1][0]
    params = dict(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=input_window_samples,
        # n_filters_time=40,
        # filter_time_length=25,
        # n_filters_spat=40,
        # pool_time_length=75,
        # pool_time_stride=15,
        # final_conv_length="auto",
        # conv_nonlin=torch.square,  # 常用 square nonlinearity
        # pool_mode="mean",
        # activation_pool_nonlin=torch.log,  # 常用 log nonlinearity
        # split_first_layer=True,
        # batch_norm=True,
        # batch_norm_alpha=0.1,
        # drop_prob=0.5,
        # chs_info=info["chs"],
        # input_window_seconds=input_window_sec,
        # sfreq=sfreq
    )
    return params


def load_EEGNet_our_ver_params(dataset):
    data_shape = [dataset.__getitem__(0)[0].shape, dataset.__getitem__(0)[1].shape]

    # model input info
    n_channels = data_shape[0][1]
    input_window_samples = data_shape[0][2]
    n_classes = data_shape[1][0]
    
    params = dict(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=input_window_samples,
        # F1=4, 
        # D=2, 
        # F2=8, 
        # T_kernSizes=[32, 64, 128, 200, 256, 512], 
        # S_kernSizes=[], 
        # dropout=0.55
    )
    return params
