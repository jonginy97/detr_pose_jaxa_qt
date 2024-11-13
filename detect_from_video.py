import argparse
import datetime
import random
import time
from pathlib import Path
import os
import cv2

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image
import matplotlib.pyplot as plt

from torchvision.ops import nms

import datasets
import util.misc as utils
import torchvision.transforms as T

from models import build_model

from datasets.JAXA import make_JAXA_transforms

# 클래스 정의
CLASSES = ['forceps', 'scissors']

# 박스 변환 함수
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size, device):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
    return b


def detect(im, model, transform, device): 
    start_time = time.time()

    img = transform(im).unsqueeze(0).to(device)
    outputs = model(img)
    probas = outputs['pred_logits'].softmax(-1)[0, :, 1:]

    # 각 클래스별로 최대 확률과 그에 해당하는 인덱스를 찾음
    max_scores, max_indices = probas.max(dim=0)
    
    # 각 클래스별로 최대 확률이 0.8 이상인 경우만 선택
    keep = (max_scores > 0.8).nonzero(as_tuple=True)[0]
    
    # 선택된 인덱스에 따라 최대 클래스 인덱스와 확률을 유지
    max_indices = max_indices[keep]
    max_scores = max_scores[keep]
    
    # bounding box 스케일링
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size, device)

    # Bounding box 값을 0 이상으로 클리핑
    bboxes_scaled = torch.clamp(bboxes_scaled, min=0)

    # NMS 적용
    if keep.numel() > 0:
        print("bboxes_scaled:", bboxes_scaled)
        print("max_scores:", max_scores)
        keep = nms(bboxes_scaled, max_scores, iou_threshold=0.5)
    else:
        keep = torch.tensor([], dtype=torch.int64, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print(f'Inference time {total_time_str}')
    
    # 선택된 박스에 대한 클래스 확률 반환
    return probas[keep], bboxes_scaled[keep]



# 결과를 시각화하는 함수
def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color='blue', linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    
    plt.show(block=False)
    plt.pause(0.5)
    
    # 창을 자동으로 닫습니다.
    plt.close()

# 체크포인트 로드 함수
def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model'])
    print("Checkpoint loaded successfully.")

def overlay_results(frame, prob, boxes):
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        # 파란색 테두리를 그립니다 (굵기를 줄임).
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(255, 0, 0), thickness=2)
        
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        
        # 텍스트 사이즈를 줄이기 위해 fontScale 값을 줄임
        font_scale = 0.4
        font_thickness = 1  # 글자 굵기를 줄임
        
        # 노란색 반투명한 텍스트 배경을 추가합니다. alpha 값으로 반투명도를 조절
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        overlay = frame.copy()
        cv2.rectangle(overlay, (int(xmin), int(ymin) - text_height - 10), 
                      (int(xmin) + text_width, int(ymin)), (0, 255, 255), -1)
        alpha = 0.7  # 투명도
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 텍스트를 이미지에 추가합니다.
        cv2.putText(frame, text, (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

def detect_from_video(video_path, model, transform, device, output_video_path=None, save_output=False):
    cap = cv2.VideoCapture(video_path)
    
    if save_output and output_video_path is not None:
        # 비디오의 프레임 속도, 높이, 넓이를 가져옵니다.
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # VideoWriter 객체를 생성하여 결과를 저장할 비디오 파일을 준비합니다.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정 (MP4 형식)
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break     
        # PIL 이미지로 변환
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # 감지 수행
        scores, boxes = detect(pil_img, model, transform, device)       
        # 결과 오버레이
        overlay_results(frame, scores, boxes)
        
        if save_output and output_video_path is not None:
            # 결과를 비디오 파일에 기록합니다.
            out.write(frame)
        
        # 결과 출력
        cv2.imshow('Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 비디오 객체와 VideoWriter 객체를 해제합니다.
    cap.release()
    if save_output and output_video_path is not None:
        out.release()
    cv2.destroyAllWindows()






# 모델과 훈련 설정 파라미터
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    
    
    
    # model eval 모드로
    model.eval()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 체크포인트 로드
    checkpoint_base = r'F:\YOON\research\detr_jaxa\detr\output\random_30000_30 epochs'
    checkpoint_name = 'checkpoint.pth'
    load_checkpoint(model, os.path.join(checkpoint_base, checkpoint_name))


  
    transform = T.Compose([
    # T.Resize(800),
    T.ToTensor(),
    # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    T.Normalize([0.3369, 0.3866, 0.4526], [0.1927, 0.2150, 0.2402])
])
    video_base = r'F:\YOON\JAXA\datasets\synthetic\new\1st_trajec\detr_dataset'
    video_path = os.path.join(video_base, 'images_30fps_trimmed.mp4')
    # video_path = r'F:\YOON\JAXA\datasets\robot_trajec\robot_traj_0205.mp4'
    
    output_video_path = os.path.join(checkpoint_base, 'output_150fps_trimmed.mp4')
    detect_from_video(video_path, model, transform, device, output_video_path, save_output=False)
    

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    # override manually
    args.dataset_file = 'JAXA'
    args.data_path = r'F:\YOON\JAXA\datasets\synthetic\new\random_30000\detr_dataset'
    args.output_dir = 'output'
    args.batch_size = 8
    args.epochs = 100
    args.world_size = 2

    main(args)
    print('done')


