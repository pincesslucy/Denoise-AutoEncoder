import torch
from PIL import Image
from model.dae import Denoise
from model.vae_denoise import VAE_denoise
from torchvision import transforms
from torchvision.utils import save_image

img_size = (3, 32, 32)
model = Denoise()
model.load_state_dict(torch.load('/mnt/c/Users/lee/desktop/denoise/model/dea.pt'))
img = Image.open('/mnt/c/Users/lee/desktop/denoise/experiments/test_img.jpg')
img_resize = img.resize((32,32))

transform = transforms.ToTensor()
img_tensor = transform(img_resize)


img_tensor = img_tensor.unsqueeze(0)
img_tensor = img_tensor[:, :3, :, :]

noise = torch.randn_like(img_tensor) * 0.1
noisy_imgs = img_tensor + noise

save_image(noisy_imgs, '/mnt/c/Users/lee/desktop/denoise/experiments/noise_img.jpg')

model.eval()  # 모델을 평가 모드로 설정합니다.
with torch.no_grad():  # 추론 과정에서는 기울기를 계산할 필요가 없으므로, 기울기 계산을 비활성화합니다.
    _, output = model(noisy_imgs)



# output은 (배치, 채널, 높이, 너비) 형태의 Tensor입니다.
# save_image 함수는 (채널, 높이, 너비) 형태의 Tensor를 입력으로 받으므로, 배치 차원을 제거합니다.
output_img = output.squeeze(0)

# Tensor를 이미지로 저장합니다.
# save_image 함수는 0과 1 사이의 값을 가진 Tensor를 입력으로 받아, 0과 255 사이의 값을 가진 이미지로 변환하여 저장합니다.
save_image(output_img, '/mnt/c/Users/lee/desktop/denoise/experiments/output_img.jpg')
