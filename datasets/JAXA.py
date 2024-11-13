# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
JAXA dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/Face_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as JAXA_mask

import datasets.transforms as T


class JAXADetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(JAXADetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertJAXAPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(JAXADetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_JAXA_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = JAXA_mask.frPyObjects(polygons, height, width)
        mask = JAXA_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertJAXAPolysToMask(object):
    def __init__(self, return_masks=False, normalization='max'):
        self.return_masks = return_masks
        self.normalization = normalization # 'max' or 'z-score' for position and orientation

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_JAXA_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)
                              
        # Extract position and orientation directly and convert to tensor
        positions = [obj["pose"]["position"] for obj in anno if "pose" in obj]
        orientations = [obj["pose"]["orientation"] for obj in anno if "pose" in obj]

        # Convert to tensors
        positions = torch.tensor(positions, dtype=torch.float32)
        orientations = torch.tensor(orientations, dtype=torch.float32)
       
        # Perform normalization only if positions and orientations are not empty
        if positions.numel() > 0:
            if self.normalization == 'max':
                # Max normalization
                position_max_factors = torch.tensor([150.0, 100.0, 450.0], dtype=torch.float32)
                positions /= position_max_factors
            elif self.normalization == 'z-score':
                raise NotImplementedError("Z-score normalization is not yet implemented.\n \
                                          Please define the necessary mean and std values.")

        if orientations.numel() > 0: # numel() returns the number of elements in the tensor
            if self.normalization == 'max':
                orientation_max_factors = torch.tensor([180.0, 180.0, 180.0], dtype=torch.float32)
                orientations /= orientation_max_factors
            elif self.normalization == 'z-score':
                raise NotImplementedError("Z-score normalization is not yet implemented.\n \
                                          Please define the necessary mean and std values.")

        # 6D pose creation
        if positions.numel() > 0 and orientations.numel() > 0:
            pose_6d = torch.cat((positions, orientations), dim=1)
        else:
            pose_6d = torch.tensor([])

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
        pose_6d = pose_6d[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["pose_6d"] = pose_6d
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to JAXA api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


# def make_JAXA_transforms():

#     normalize = T.Compose([
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]


#     return T.Compose([
#         T.RandomHorizontalFlip(),
#         T.RandomSelect(
#             T.RandomResize(scales, max_size=1333),
#             T.Compose([
#                 T.RandomResize([400, 500, 600]),
#                 T.RandomSizeCrop(384, 600),
#                 T.RandomResize(scales, max_size=1333),
#             ])
#         ),
#         normalize,
#     ])


def make_JAXA_transforms(color_transforms_enabled=False):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.3369, 0.3866, 0.4526], [0.1927, 0.2150, 0.2402])
    ])

    color_transforms = T.Compose([
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomGrayscale(p=0.1),
        T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0))], p=0.5)
    ])

    transforms_list = []
    
    # color_transforms_enabled가 True인 경우 색상 변환을 추가
    if color_transforms_enabled:
        transforms_list.append(color_transforms)

    transforms_list.extend([
        normalize
    ])

    return T.Compose(transforms_list)



def build(args):
    root = Path(args.data_path)
    assert root.exists(), f'provided Jaxa path {root} does not exist'
    img_folder = root / "images"
    ann_file = root / "annotations.json"

    dataset = JAXADetection(img_folder, ann_file, make_JAXA_transforms(), return_masks=args.masks)

    print(f"Total dataset size: {len(dataset)}")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")

    return train_dataset, test_dataset