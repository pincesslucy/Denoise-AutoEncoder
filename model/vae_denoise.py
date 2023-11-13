import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VAE_denoise(nn.Module):
    def __init__(self):
        super(VAE_denoise, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Sigmoid()  # normalize to [0, 1]
        )

    def forward(self, img):
        enc = self.encoder(img)
        dc = self.decoder(enc)
        return enc, dc
