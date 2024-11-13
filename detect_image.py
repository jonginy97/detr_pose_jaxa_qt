import os
import random
import numpy as np
import torch
import orjson
import re

import util.misc as utils

from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as T

from detect.detection import detect, output_to_dic
from detect.misc import plot_results
from detect import build_model, get_args_parser

# Class names for detection
CLASSES = ['forceps', 'scissors']

def load_checkpoint(model, checkpoint_path):
    """Loads model checkpoint from the specified path."""
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model'])
    print("Checkpoint loaded successfully.")

def save_results_to_json(results, output_path):
    """
    모든 키를 문자열로 변환한 후 JSON 형식으로 저장합니다.
    배열은 한 줄로 표시합니다.
    """
    # 최상위 딕셔너리와 내부의 모든 키를 문자열로 변환
    results_str_keys = {str(k): v for k, v in results.items()}
    
    # JSON 직렬화 (들여쓰기 포함)
    json_bytes = orjson.dumps(results_str_keys, option=orjson.OPT_INDENT_2)
    json_str = json_bytes.decode('utf-8')
    
    # 배열 내부의 줄바꿈과 공백 제거 (배열을 한 줄로 만듦)
    json_str = re.sub(r'\[\s+([^\]]+?)\s+\]', lambda m: '[' + ' '.join(m.group(1).strip().split()) + ']', json_str)
    
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json_str)
    print(f"결과가 {output_path}에 저장되었습니다.")



def main(args):
    # Set device and seed for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build and load the model
    model = build_model(args)
    model.to(device)
    model.eval()

    load_checkpoint(model, args.checkpoint)

    # Define image transformation pipeline
    transform = T.Compose([
        T.ToTensor(),
        T.Resize([1200, 1200]),
        T.Normalize([0.3369, 0.3866, 0.4526], [0.1927, 0.2150, 0.2402])
    ])

    img_dir = args.img_dir
    all_results = {}

    # Sort image files by numerical order using lambda function
    img_files = sorted(os.listdir(img_dir), key=lambda x: int(os.path.splitext(x)[0]))

    # Process each image file
    for idx, img_name in enumerate(tqdm(img_files)):
        img_path = os.path.join(img_dir, img_name)
        # print(f"Opening image: {os.path.basename(img_path)}")

        im = Image.open(img_path)
        scores, boxes, poses = detect(im, model, transform, device=device)

        # Convert detection output to a dictionary format
        results = output_to_dic(scores.argmax(dim=1), scores.max(dim=1).values, boxes, poses, class_names=CLASSES)
        all_results[idx] = results

        # print(results)

        # Plot and show the detection results
        if args.plot:
            plot_results(im, scores, boxes)
            plt.close()

    # Save all detection results to JSON file
    output_json_path = os.path.join(args.output_dir, "results.json")
    save_results_to_json(all_results, output_json_path)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    args.plot = False
    args.checkpoint = r'F:\YOON\research\detr_pose_jaxa\output\query_40\20241106-212452\checkpoint.pth'
    args.num_queries = 40
    args.img_dir = r'F:\YOON\Datasets\JAXA_dataset\synthetic\new\1st_trajec\detr_dataset\images'
    args.output_dir = r'F:\YOON\research\detr_pose_jaxa\output\query_40\20241106-212452\1st_trajec'

    # Create output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
