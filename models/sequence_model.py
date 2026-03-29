import torch
import torch.nn as nn
from torchvision import models

class CNNLSTM(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()

        # CNN backbone (ResNet18)
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # remove FC
        self.feature_dim = 512

        # Freeze CNN (important!)
        for param in self.cnn.parameters():
            param.requires_grad = False

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Classifier
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x shape: [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape

        # merge batch & time
        x = x.view(B * T, C, H, W)

        # CNN feature extraction
        features = self.cnn(x)          # [B*T, 512, 1, 1]
        features = features.view(B, T, self.feature_dim)

        # LSTM over time
        _, (hn, _) = self.lstm(features)

        # Last hidden state
        out = self.fc(hn[-1])
        return out
