from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from glob import glob
import numpy as np

class CustumDataset(Dataset):
    def __init__(self, img_dir, transform):
        # self.img = glob(img_dir+"/*.jpg")
        self.img = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
        # image = Image.open(self.img[idx])
        # resized = image.resize((32, 32))
        imgs = self.transform(self.img[idx])
        # 노이즈 추가
        noise = np.random.normal(loc=0.0, scale=0.1, size=imgs.shape)
        noise = torch.from_numpy(noise).float()
        noisy_imgs = imgs + noise
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)
        return imgs, noisy_imgs
    
    