import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

def calculate_mean_std(image_dir):
    transform = transforms.Compose([transforms.ToTensor()])
    
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

    mean = torch.zeros(3)
    std = torch.zeros(3)
    for img_file in tqdm(image_files, desc='Calculating mean and std'):
        img = Image.open(img_file).convert('RGB')
        img_tensor = transform(img)
        mean += img_tensor.mean(dim=(1, 2))
        std += img_tensor.std(dim=(1, 2))
           
    mean /= len(image_files)
    std /= len(image_files)
    return mean, std


image_path = r'F:\YOON\JAXA\datasets\synthetic\new\random_30000\detr_dataset\images'

mean, std = calculate_mean_std(image_path)
print(f"Mean: {mean}")
print(f"Standard Deviation: {std}")

# mean = torch.tensor([0.3360, 0.3861, 0.4524])
# std = torch.tensor([0.1926, 0.2152, 0.2407])