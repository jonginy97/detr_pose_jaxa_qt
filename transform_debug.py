import os
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch

# 이미지 경로 설정
img_dir = r'F:\YOON\JAXA\datasets\synthetic\new\1st_trajec\detr_dataset\images'
img_name = '4457.png'
img_path = os.path.join(img_dir, img_name)

# 이미지 로드
img = Image.open(img_path).convert("RGB")

# 원본 이미지 시각화
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

# 각각의 트랜스폼 적용
# 1. 더 강한 ColorJitter 변환
color_jitter = T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.4)
img_color_jitter = color_jitter(img)

plt.subplot(2, 3, 2)
plt.imshow(img_color_jitter)
plt.title("Stronger ColorJitter Applied")
plt.axis("off")

# 2. Grayscale 변환
random_grayscale = T.RandomGrayscale(p=1.0)
img_grayscale = random_grayscale(img)

plt.subplot(2, 3, 3)
plt.imshow(img_grayscale)
plt.title("RandomGrayscale Applied")
plt.axis("off")

# 3. GaussianBlur 변환
gaussian_blur = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0))
img_gaussian_blur = gaussian_blur(img)

plt.subplot(2, 3, 4)
plt.imshow(img_gaussian_blur)
plt.title("GaussianBlur Applied")
plt.axis("off")

# 4. Normalize 변환
normalize = T.Compose([
    T.ToTensor(),
    T.Normalize([0.3369, 0.3866, 0.4526], [0.1927, 0.2150, 0.2402])
])

img_normalized = normalize(img)
img_normalized_denorm = img_normalized.permute(1, 2, 0).numpy()

# Normalize 풀어주기
mean = [0.3369, 0.3866, 0.4526]
std = [0.1927, 0.2150, 0.2402]
img_normalized_denorm = img_normalized_denorm * std + mean

# NumPy로 clip 함수 적용
img_normalized_denorm = np.clip(img_normalized_denorm, 0, 1)

plt.subplot(2, 3, 5)
plt.imshow(img_normalized_denorm)
plt.title("ColorJitter + Normalized Image")
plt.axis("off")

plt.tight_layout()
plt.show()
