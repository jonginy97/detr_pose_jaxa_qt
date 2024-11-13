from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# 계산된 평균과 표준편차
mean = [0.3369, 0.3866, 0.4526]
std = [0.1927, 0.2150, 0.2402]

# 이미지 경로 설정
image_base_path = r'F:\YOON\JAXA\datasets\jaxa_1000samples\images'

# 변환 정의
transform_1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

transform_2 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),        # 50% 확률로 좌우 반전
    transforms.RandomVerticalFlip(p=0.5),          # 50% 확률로 상하 반전
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 및 밝기, 대비, 채도 변경
    transforms.RandomRotation(degrees=20),         # 최대 20도까지 회전
    transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),  # 임의 크기로 자르고 256x256 크기로 변경
    transforms.RandomAffine(
        degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10,  # 이동, 스케일링 및 시어 변환
    ),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),  # 원근 변환 적용
    transforms.RandomGrayscale(p=0.2),           # 20% 확률로 흑백 변환
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # 가우시안 블러 적용
    transforms.RandomInvert(p=0.5),              # 색상 반전
    transforms.RandomPosterize(bits=4, p=0.5),   # 포스터라이즈 변환
    transforms.RandomSolarize(threshold=128, p=0.5),  # 솔러라이즈 변환
    transforms.RandomEqualize(p=0.5),            # 히스토그램 평활화 적용
    transforms.ToTensor(),                        # 이미지를 텐서로 변환
    transforms.Normalize(mean=mean, std=std)     # 정규화
])




transform_3 = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 및 밝기, 대비, 채도 변경
    transforms.RandomGrayscale(p=0.2),           # 20% 확률로 흑백 변환
    transforms.RandomInvert(p=0.5),              # 색상 반전
    transforms.RandomPosterize(bits=4, p=0.5),   # 포스터라이즈 변환
    transforms.RandomSolarize(threshold=128, p=0.5),  # 솔러라이즈 변환
    transforms.RandomEqualize(p=0.5),            # 히스토그램 평활화 적용
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # 가우시안 블러 적용
    transforms.ToTensor(),                        # 이미지를 텐서로 변환
    transforms.Normalize(mean=mean, std=std)     # 정규화
])




# 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(image_base_path) if f.endswith('.png')]

# 이미지 순차적으로 처리
for image_file in image_files:
    image_path = os.path.join(image_base_path, image_file)
    
    # 이미지 로드 및 변환
    image = Image.open(image_path).convert('RGB')
    t1_image = transform_1(image)
    t2_image = transform_2(image)

    # 변환된 이미지 시각화 (이미지를 [0, 1] 범위로 변환 필요)
    t1_image = t1_image.permute(1, 2, 0).numpy()
    t1_image = t1_image.clip(0, 1)  # [0, 1] 범위로 클리핑

    t2_image = t2_image.permute(1, 2, 0).numpy()
    t2_image = t2_image.clip(0, 1)  # [0, 1] 범위로 클리핑
    
    t3_image = transform_3(image)
    t3_image = t3_image.permute(1, 2, 0).numpy()
    t3_image = t3_image.clip(0, 1)  # [0, 1] 범위로 클리핑
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.subplot(1, 4, 2)
    plt.imshow(t1_image)
    plt.title('Normalized Image')
    plt.subplot(1, 4, 3)
    plt.imshow(t2_image)
    plt.title('Augmented Image')
    plt.subplot(1, 4, 4)
    plt.imshow(t3_image)
    plt.title('Augmented Image 2')
    plt.show()
    
    plt.close
    
    
