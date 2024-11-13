import os
import random
import util.misc as utils
import numpy as np
import torch
import pathlib
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import math
from dqrobotics import *
from detect.detection import detect, output_to_dic
from detect.misc import plot_results
from detect import build_model, get_args_parser
from detect import SyntheticDataModule

# 클래스 정의
CLASSES = ['forceps', 'scissors']

def DQfromTxTyTzRxRyRz(tx, ty, tz, rx, ry, rz):
    rx = math.radians(rx)
    ry = math.radians(ry)
    rz = math.radians(rz)
    Rx = math.cos(0.5 * rx) + math.sin(0.5 * rx) * i_
    Ry = math.cos(0.5 * ry) + math.sin(0.5 * ry) * j_
    Rz = math.cos(0.5 * rz) + math.sin(0.5 * rz) * k_
    R = Rx * Ry * Rz
    T = 1 + E_ * 0.5 * (tx * i_ + ty * j_ + tz * k_)
    return T * R

# 체크포인트 로드 함수
def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model'])
    print("Checkpoint loaded successfully.")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_model(args)
    model.to(device)
    model.eval()
    
    # pathlib 설정 변경
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    load_checkpoint(model, args.checkpoint)
    
    transform = T.Compose([
        T.ToTensor(),
        T.Resize([1200, 1200]),
        T.Normalize([0.3369, 0.3866, 0.4526], [0.1927, 0.2150, 0.2402])
    ])
    
    img_dir = args.img_dir
    img_files = sorted(os.listdir(img_dir), key=lambda x: int(x.split('.')[0]))
    
    # random seed
    # random.seed(0)
    # mix img_files
    # random.shuffle(img_files)
    

    for idx, img_name in enumerate(img_files):
        if idx % 20 == 0:
            img_path = os.path.join(img_dir, img_name)
            print(f"Opening image: {img_name}")
            
            im = Image.open(img_path)
            scores, boxes, poses = detect(im, model, transform, device=device)
            results = output_to_dic(scores.argmax(dim=1), scores.max(dim=1).values, boxes, poses, class_names={0: 'forceps', 1: 'scissors'})
            print(results)

            # forceps_pose와 scissors_pose 초기화
            forceps_pose = None
            scissors_pose = None

            if 0 in results:
                forceps_pose = results[0]['pose']
                print("Forceps pose:", forceps_pose)
            else:
                print("Forceps not detected.")

            if 1 in results:
                scissors_pose = results[1]['pose']
                print("Scissors pose:", scissors_pose)
            else:
                print("Scissors not detected.")

            sdm = SyntheticDataModule.SyntheticDataModule()
            sdm.connect()
            if forceps_pose is not None:
                sdm.set_forceps_state(DQfromTxTyTzRxRyRz(*forceps_pose), 1)
            if scissors_pose is not None:
                sdm.set_scissors_state(DQfromTxTyTzRxRyRz(*scissors_pose), 1)
            
            plot_results(im, scores, boxes)
            plt.close()

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    args.img_dir = r'F:\YOON\Datasets\JAXA_dataset\synthetic\new\2nd_trajec\detr_dataset\images'
    args.checkpoint = r'F:\YOON\research\detr_pose_jaxa\output\query_40\resolution_1200_30epoch\checkpoint.pth'
    args.num_queries = 40
    main(args)
