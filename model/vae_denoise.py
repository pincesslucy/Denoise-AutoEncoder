import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VAE_denoise(nn.Module):
    def __init__(self):
        super(VAE_denoise, self).__init__()

        # 인코더
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(in_features=128*8*8, out_features=1024)
        self.fc21 = nn.Linear(in_features=1024, out_features=32)  # 평균을 출력하는 layer
        self.fc22 = nn.Linear(in_features=1024, out_features=32)  # 로그 분산을 출력하는 layer
        
        # 디코더
        self.fc3 = nn.Linear(in_features=32, out_features=1024)
        self.fc4 = nn.Linear(in_features=1024, out_features=128*8*8)
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
        
    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3)).view(-1, 128, 8, 8)
        deconv1 = F.relu(self.deconv1(h4))
        deconv2 = F.relu(self.deconv2(deconv1))
        return torch.sigmoid(self.conv4(deconv2))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var