from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dataset.custumdataset import CustumDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from dae import Denoise
from vae_denoise import VAE_denoise
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
vae = VAE_denoise()

if cuda:
    dea.cuda()
    vae.cuda()
    criterion.cuda()


# Optimizers
optimizer_dea = torch.optim.Adam(dea.parameters(), lr=lr)
optimizer_vae = torch.optim.Adam(vae.parameters(), lr=lr)

def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(model, model_name, ephochs=100):
    for epoch in range(ephochs):
        for real, noise in tqdm(iter(dataloader)):
            real_imgs = real
            noise_imgs = noise

            if cuda:
                real_imgs = real_imgs.cuda()
                noise_imgs = noise_imgs.cuda()

            if model == dea:
                enc, output = model(noise_imgs)

                loss = criterion(output, real_imgs)
                optimizer_dea.zero_grad()
                loss.backward()
                optimizer_dea.step()

            if model == vae:
                output, mu, log_var = model(noise_imgs)
                loss = loss_function(output, real_imgs, mu, log_var)
                optimizer_vae.zero_grad()
                loss.backward()
                optimizer_vae.step()



        print("[Epoch %d/%d] [loss: %f]"% (epoch, ephochs, loss.item()))

    save_image(noise_imgs[0], f'/mnt/c/Users/lee/desktop/denoise/experiments/{model_name}_noise{epoch}.jpg')        
    save_image(output[0], f'/mnt/c/Users/lee/desktop/denoise/experiments/{model_name}_train{epoch}.jpg')
    torch.save(model.state_dict(), f"{model_name}.pt")

train(vae ,'vae', 100)