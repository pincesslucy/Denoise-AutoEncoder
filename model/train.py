from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dataset.custumdataset import CustumDataset
import torch
import torch.nn as nn
from dae import Denoise
from tqdm import tqdm
from keras.datasets import cifar10
from torchvision.utils import save_image
import numpy as np

# CIFAR-10 데이터셋 불러오기
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_images = train_images.astype(np.float32)/255.0

transform = transforms.ToTensor()
dataset = CustumDataset(train_images, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

img_size = (3, 32, 32)
lr = 0.0001

cuda = True if torch.cuda.is_available() else False

criterion = nn.MSELoss()

dea = Denoise()

if cuda:
    dea.cuda()
    criterion.cuda()


# Optimizers
optimizer = torch.optim.Adam(dea.parameters(), lr=lr)


all_gen_data = []
for epoch in range(10):
    for real, noise in tqdm(iter(dataloader)):
        real_imgs = real
        noise_imgs = noise

        if cuda:
            real_imgs = real_imgs.cuda()
            noise_imgs = noise_imgs.cuda()

        enc, dc = dea(noise_imgs)

        loss = criterion(dc, real_imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    save_image(noise_imgs[0], f'/mnt/c/Users/lee/desktop/denoise/experiments/noise{epoch}.jpg')        
    save_image(dc[0], f'/mnt/c/Users/lee/desktop/denoise/experiments/train{epoch}.jpg')

    print("[Epoch %d/%d] [loss: %f]"% (epoch, 10, loss.item()))
torch.save(dea.state_dict(), f"dea.pt")
