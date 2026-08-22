"""
EEG 模型定義（standalone 複本）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
                 dropout=0.55):

        super(EEGNet_our_ver, self).__init__()

        # 1. Temporal Conv (multi-kernel)
        self.T_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, F1, (1, k), padding=(0, k // 2), bias=False),
                nn.BatchNorm2d(F1),
                nn.ELU()
            )
            for k in T_kernSizes
        ])
        F1 = F1 * len(T_kernSizes)

        # 2. Spatial Conv (multi-kernel)
        S_kernSizes = [n_chans, n_chans // 2 + 1, 3]
        self.S_kernSizes = S_kernSizes

        self.S_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(F1, F1 * D, (ks, 1), groups=F1, padding=(ks // 2, 0), bias=False),
                nn.BatchNorm2d(F1 * D),
                nn.ELU()
            )
            for ks in S_kernSizes
        ])
        F1 = F1 * len(S_kernSizes)

        # 3. Pooling + Dropout
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

        # Flatten feature dim
        self.feature_dim = self._get_feature_dim(n_chans, n_times)

        # Classifier
        self.classifier = nn.Linear(self.feature_dim, n_outputs)

    def _get_feature_dim(self, n_chans, n_times):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_chans, n_times)
            x = torch.cat([b(x) for b in self.T_branches], dim=1)
            x = torch.cat([b(x) for b in self.S_branches], dim=1)
            x = self.pool1(x)
            x = self.dropout1(x)
            x = self.separable(x)
            x = x.flatten(1)
        return x.shape[1]

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # -> (B, 1, C, T)

        x = torch.cat([b(x) for b in self.T_branches], dim=1)
        x = torch.cat([b(x) for b in self.S_branches], dim=1)
        x = F.elu(x)

        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.separable(x)

        x = x.flatten(1)
        return self.classifier(x)
