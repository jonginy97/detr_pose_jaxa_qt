import time
import torch
from .misc import box_cxcywh_to_xyxy, rescale_bboxes, CLASSES
from torchvision.ops import nms

def output_to_dic(max_indices, max_scores, bboxes_scaled, pred_poses_int, class_names=None):
    results = {}

    # 클래스 이름 리스트를 딕셔너리로 변환
    if class_names:
        class_mapping = {idx: name for idx, name in enumerate(class_names)}
    else:
        class_mapping = {}

    for idx in range(len(max_indices)):
        cls_idx = max_indices[idx].item()
        score = round(max_scores[idx].item(), 3)
        
        # bbox와 pose에 값을 할당합니다.
        bbox = bboxes_scaled[idx].tolist()
        pose = pred_poses_int[idx].tolist()
        
        # 값을 반올림합니다.
        bbox = [round(val, 2) for val in bbox]
        pose = [round(val, 2) for val in pose]

        
        # 클래스 이름 매핑
        cls = class_mapping.get(cls_idx, f'class_{cls_idx}')
        
        results[cls] = {
            'score': score,
            'bbox': bbox,
            'pose': pose
        }
    return results


@torch.no_grad()
def detect(im, model, transform, device): 
    start_time = time.time()
    img = transform(im).unsqueeze(0).to(device)
    outputs = model(img)
    probas = outputs['pred_logits'].softmax(-1)[0, :, 1:]
    
    probas = (probas * 1000).round() / 1000
    
    # 각 객체에 대한 최대 점수와 해당 클래스 인덱스를 얻습니다.
    max_scores, max_indices = probas.max(dim=1)
    
    # 최대 점수가 0.7 이상인 객체만 선택합니다.
    keep = (max_scores > 0.7).nonzero(as_tuple=True)[0]
    
    # 선택된 객체들의 점수와 클래스 인덱스를 필터링합니다.
    max_indices = max_indices[keep]
    max_scores = max_scores[keep]
    
    # 바운딩 박스를 스케일링합니다.
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size, device)
    bboxes_scaled = torch.clamp(bboxes_scaled, min=0)
    
    # 포즈 예측을 처리합니다.
    pred_poses = outputs['pred_poses'][0, keep]
    scale_factors = torch.tensor([150, 100, 450, 180, 180, 180], dtype=torch.float32).to(device)
    pred_poses_scaled = pred_poses * scale_factors
    
 
    pred_poses_int = pred_poses_scaled
    # pred_poses_int = pred_poses_int.to(torch.int64)
    # round long 은 round 함수를 사용하여 소수점 이하를 반올림한 후 long 타입으로 변환합니다. long 타입은 정수형 데이터 타입입니다.
    
    # NMS를 적용하여 중복된 바운딩 박스를 제거합니다.
    if keep.numel() > 0:
        nms_keep = nms(bboxes_scaled, max_scores, iou_threshold=0.2)
        
        # NMS 후의 인덱스를 원래 인덱스로 매핑합니다.
        final_keep = keep[nms_keep]
        
        # NMS 결과를 사용하여 최종 결과를 업데이트합니다.
        max_indices = max_indices[nms_keep]
        max_scores = max_scores[nms_keep]
        bboxes_scaled = bboxes_scaled[nms_keep]
        pred_poses_int = pred_poses_int[nms_keep]
    else:
        # 탐지 결과가 없는 경우 빈 텐서를 생성합니다.
        final_keep = torch.tensor([], dtype=torch.int64, device=device)
        max_indices = torch.tensor([], dtype=torch.int64, device=device)
        max_scores = torch.tensor([], device=device)
        bboxes_scaled = torch.empty((0, 4), device=device)
        pred_poses_int = torch.empty((0, 6), dtype=torch.int64, device=device)
    
    # NMS 이후에 각 클래스별로 가장 높은 점수를 가진 탐지 결과만 남깁니다.
    if len(max_scores) > 0:
        unique_classes = max_indices.unique()
        selected_indices = []
        for cls in unique_classes:
            cls_indices = (max_indices == cls).nonzero(as_tuple=True)[0]
            cls_scores = max_scores[cls_indices]
            max_score_idx = cls_scores.argmax()
            selected_idx = cls_indices[max_score_idx]
            selected_indices.append(selected_idx)
        
        # 선택된 인덱스로 결과를 업데이트합니다.
        selected_indices = torch.tensor(selected_indices, device=device)
        max_indices = max_indices[selected_indices]
        max_scores = max_scores[selected_indices]
        bboxes_scaled = bboxes_scaled[selected_indices]
        pred_poses_int = pred_poses_int[selected_indices]
        
        # **final_keep도 업데이트합니다.**
        final_keep = final_keep[selected_indices]
    else:
        # 탐지 결과가 없는 경우 빈 텐서를 생성합니다.
        max_indices = torch.tensor([], dtype=torch.int64, device=device)
        max_scores = torch.tensor([], device=device)
        bboxes_scaled = torch.empty((0, 4), device=device)
        pred_poses_int = torch.empty((0, 6), dtype=torch.int64, device=device)
        final_keep = torch.tensor([], dtype=torch.int64, device=device)
    
    total_time = time.time() - start_time
    print(f'Inference time: {total_time:.4f} seconds')
    
    # 각 탐지 결과를 출력합니다.
    for idx in range(len(max_scores)):
        cls = max_indices[idx].item()
        score = max_scores[idx].item()
        bbox = bboxes_scaled[idx].tolist()
        pose = pred_poses_int[idx].tolist()
        # print(f'Detection {idx}: Class {cls}, Score {score:.2f}, BBox {bbox}, Pose {pose}')
        
    # probas[final_keep], bboxes_scaled, pred_poses_int 출력
    # print(probas[final_keep])
    # print(bboxes_scaled)
    # print(pred_poses_int)
    
    # 최종 확률, 바운딩 박스, 포즈를 반환합니다.
    return probas[final_keep], bboxes_scaled, pred_poses_int
