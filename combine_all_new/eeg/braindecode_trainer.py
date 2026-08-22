"""
通用 Braindecode 訓練器。

拆自原 EEG_DL/EEG/MI_train.py 中的 class BraindecodeTrainer。
"""
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import global_state


class BraindecodeTrainer:
    def __init__(self,
                 model_class, model_kwargs,
                 dataset, val_dataset,
                 batch_size,
                 num_epochs,
                 lr,
                 ft,
                 tcp_client):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        """初始化模型"""
        self.model = self.model_class(**self.model_kwargs).to(self.device)
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
                x_batch = x_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).float()

                output = self.model(x_batch.squeeze(1))
                loss = self.criterion(output, y_batch)
                losses.append(loss.item())

                preds = output.argmax(dim=1)
                total_correct += (preds == y_batch.argmax(dim=1)).sum().item()
                total_samples += y_batch.size(0)

        return np.mean(losses), total_correct / total_samples

    def train(self, freeze_layer=False, patience=10):
        history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
        tag = "FT" if self.ft else "Train"

        # Early Stopping
        best_val_loss = float('inf')
        early_stop_counter = 0

        # 凍結層（只訓練 classifier）
        if freeze_layer:
            if hasattr(self.model, 'classifier'):
                final_layer = self.model.classifier
            elif hasattr(self.model, 'final_layer'):
                final_layer = self.model.final_layer
            else:
                raise AttributeError("Model does not have 'classifier' or 'final_layer' attribute.")

            for param in self.model.parameters():
                param.requires_grad = False
            for param in final_layer.parameters():
                param.requires_grad = True

            self.optimizer = torch.optim.Adam(final_layer.parameters(), lr=self.lr, betas=(0.5, 0.999))

        epochBar = tqdm(range(self.num_epochs), desc=f"[{tag}] Training", unit="epoch")
        for epoch in epochBar:
            self.model.train()

            # Fine-tuning 時鎖定 BatchNorm
            if freeze_layer:
                for module in self.model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()

            running_loss, correct, total = 0.0, 0, 0

            batchBar = tqdm(self.train_loader, desc=f"  Epoch {epoch + 1}/{self.num_epochs}",
                            leave=False, unit="batch")
            for inputs, labels in batchBar:
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device).float()

                self.optimizer.zero_grad()
                outputs = self.model(inputs.squeeze(1))
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.argmax(dim=1)).sum().item()

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
            }, os.path.join(global_state.EEG_CHECKPOINT_TMP_BASE_DIR, f"{tag.lower()}-epoch{epoch}.pth"))

            epochBar.set_postfix(
                loss=f"{train_loss:.4f}", acc=f"{train_acc:.4f}",
                val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}"
            )

            if self.tcp_client:
                self.tcp_client.send_messages(
                    f"training{config.TCP_STR_SPLIT}"
                    f"Epoch {epoch+1}/{self.num_epochs} - "
                    f"loss: {train_loss:.4f}, acc: {train_acc:.4f}, "
                    f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
                )

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best val_loss: {best_val_loss:.4f}")
                break

        time.sleep(0.5)
        return history


# ─── Debug ───
if __name__ == "__main__":
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset
    from eeg.eeg_model import EEGNet_our_ver

    print("=== eeg.braindecode_trainer standalone test ===")

    # 製造假資料
    n_samples, n_chans, n_times, n_classes = 32, 13, 500, 4
    X = torch.randn(n_samples, 1, n_chans, n_times)
    y = F.one_hot(torch.randint(0, n_classes, (n_samples,)), num_classes=n_classes)
    dataset = TensorDataset(X, y)

    X_val = torch.randn(8, 1, n_chans, n_times)
    y_val = F.one_hot(torch.randint(0, n_classes, (8,)), num_classes=n_classes)
    val_dataset = TensorDataset(X_val, y_val)

    model_kwargs = dict(n_chans=n_chans, n_outputs=n_classes, n_times=n_times)

    trainer = BraindecodeTrainer(
        model_class=EEGNet_our_ver,
        model_kwargs=model_kwargs,
        dataset=dataset,
        val_dataset=val_dataset,
        batch_size=8,
        num_epochs=2,
        lr=1e-3,
        ft=False,
        tcp_client=None,
    )
    print(f"Model device: {trainer.device}")
    print("✅ braindecode_trainer OK (skipping actual train in debug)")
