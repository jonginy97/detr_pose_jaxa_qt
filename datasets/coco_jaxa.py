import os
import argparse
import re
import numpy as np
import cv2
import json
import yaml
from tqdm import tqdm
from PIL import Image
from skimage import measure
from scipy.spatial.transform import Rotation as R


def parse_args():
    parser = argparse.ArgumentParser(description="Generate COCO-style annotations from images and masks.")
    parser.add_argument('--base_dir', type=str, help="Base directories for images and masks")
    parser.add_argument('--output_json', type=str, help="Output JSON file for annotations")
    parser.add_argument('--bbox_obtain', action='store_true', help="Whether to obtain bounding boxes")
    parser.add_argument('--poly_obtain', action='store_true', help="Whether to obtain polygons")

    return parser.parse_args()

def bbox_obtainer(mask, color):
    if color == 'R':
        color1 = 2  # Red channel
        color2 = 0  # Blue channel
    elif color == 'G':
        color1 = 1  # Green channel
        color2 = 0  # Blue channel
    else:
        print("The color is not defined")
        return 0, 0, 0, 0

    if mask is None:
        print("Failed to load image")
        return 0, 0, 0, 0  # Default bbox values when image cannot be loaded

    y_min, y_max, x_min, x_max = 0, 0, 0, 0
    is_exist = False

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if 60 + mask[y, x, color2] < mask[y, x, color1]:
                y_min = y
                is_exist = True
                break
        if is_exist:
            break

    if not is_exist:
        return 0, 0, 0, 0  # Default bbox values if color not found

    is_exist = False
    for x in range(mask.shape[1]):
        for y in range(mask.shape[0]):
            if 60 + mask[y, x, color2] < mask[y, x, color1]:
                x_min = x
                is_exist = True
                break
        if is_exist:
            break

    if not is_exist:
        return 0, 0, 0, 0  # Default bbox values if color not found

    is_exist = False
    for y in range(mask.shape[0] - 1, -1, -1):
        for x in range(mask.shape[1]):
            if 60 + mask[y, x, color2] < mask[y, x, color1]:
                y_max = y
                is_exist = True
                break
        if is_exist:
            break

    if not is_exist:
        return 0, 0, 0, 0  # Default bbox values if color not found

    is_exist = False
    for x in range(mask.shape[1] - 1, -1, -1):
        for y in range(mask.shape[0]):
            if 60 + mask[y, x, color2] < mask[y, x, color1]:
                x_max = x
                is_exist = True
                break
        if is_exist:
            break

    if not is_exist:
        return 0, 0, 0, 0  # Default bbox values if color not found

    return y_min, x_min, y_max, x_max

def polygon_obtainer(mask, color):
    if color == 'R':
        color_idx = 2
    elif color == 'G':
        color_idx = 1
    elif color == 'B':
        color_idx = 0
    else:
        print("The color is not defined")
        return []

    mask = mask[:, :, color_idx]
    _, binary_mask = cv2.threshold(mask, 60, 255, cv2.THRESH_BINARY)
    contours = measure.find_contours(binary_mask, 0.5)

    polygons = []
    for contour in contours:
        polygon = [(int(point[1]), int(point[0])) for point in contour]
        polygons.append(polygon)

    return polygons

def load_pose_data(file_path):
    """
    input :  file_path - path of the file to load pose data
                forceps.txt or scissors.txt
                ex)               
                "1":
                position:
                x: -37.1498
                y: -30.9973
                z: 386.494
                orientation:
                x: -0.235884
                y: 0.371606
                z: -0.554596
                w: 0.706181
                grip: 1
             
    output:  
            positions - [num_data, 3] : [num_data, [x, y, z]]
            orientations - [num_data, 3] : [num_data, [roll, pitch, yaw]]
            grips - [num_data] : [grip values]
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다:", file_path)
        return None
    except yaml.YAMLError as exc:
        print("YAML 파일 읽는 중 오류 발생:", exc)
        return None

    num_data = len(data)
    positions = np.zeros((num_data, 3)) 
    orientations = np.zeros((num_data, 3))
    grips = np.zeros(num_data)

    for i in range(num_data):
        entry = data[str(i + 1)]
        positions[i] = [entry["position"][k] for k in ["x", "y", "z"]]
        rotation = [entry["orientation"][k] for k in ["w", "x", "y", "z"]]
        # permute wxyz to xyzw
        rotation = [rotation[1], rotation[2], rotation[3], rotation[0]]
        r = R.from_quat(rotation)
        orientations[i] = r.as_euler('zyx', degrees=True)
        # permute zyx to xyz
        orientations[i] = [orientations[i][2], orientations[i][1], orientations[i][0]]
        grips[i] = entry["grip"]

    return positions, orientations, grips

def serialize_polygon(polygon):
    return [int(coord) for point in polygon for coord in point]

def create_coco_json(base_dir, output_json, bbox_obtain, poly_obtain):
    categories = [
        {"id": 1, "name": "forceps", "supercategory": "tool"},
        {"id": 2, "name": "scissors", "supercategory": "tool"}
    ]

    images = []
    annotations = []
    annot_count = 0
    image_list = os.listdir(os.path.join(base_dir, 'images'))
    
    print("Original image list:", image_list)
    image_list = sorted(image_list, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf'))
    print("Sorted image list:", image_list)
    
    print("Total number of images:", len(image_list))
    
    print("Loading pose data...")
    positions_forceps, orientations_forceps, grips_forceps = load_pose_data(os.path.join(base_dir, 'forceps.txt'))
    positions_scissors, orientations_scissors, grips_scissors = load_pose_data(os.path.join(base_dir, 'scissors.txt'))
    
    print("Loaded pose data for forceps:", positions_forceps.shape, orientations_forceps.shape, grips_forceps.shape)
    
    print("Creating COCO JSON file...")
    
    #tqdm
    for index, image_name in enumerate(tqdm(image_list)):

        image_path = os.path.join(base_dir, 'images', image_name)       
        mask_path = os.path.join(base_dir, 'semseg', image_name)

        img = Image.open(image_path)
        img_w, img_h = img.size

        img_elem = {
            "file_name": image_name,
            "height": img_h,
            "width": img_w,
            "id": index
        }
        images.append(img_elem)

        mask = cv2.imread(mask_path)
        if mask is None:
            print("Failed to load mask:", mask_path)
            continue
        if bbox_obtain:
            bbox1 = bbox_obtainer(mask, 'R')
            bbox2 = bbox_obtainer(mask, 'G')
            
        if poly_obtain:
            polygons1 = polygon_obtainer(mask, 'R')
            polygons2 = polygon_obtainer(mask, 'G')

        if bbox1 != (0, 0, 0, 0):
            x1, y1, w1, h1 = bbox1[1], bbox1[0], bbox1[3] - bbox1[1], bbox1[2] - bbox1[0]
            pose_info = {
                "position": positions_forceps[index].tolist(),
                "orientation": orientations_forceps[index].tolist(),
                "grip": grips_forceps[index]
            }
            annotation1 = {
                "id": annot_count,
                "image_id": index,
                "category_id": 1,
                "bbox": [x1, y1, w1, h1],
                "pose": pose_info,
                "area": w1 * h1,
                "segmentation": [serialize_polygon(polygon) for polygon in polygons1],
                "iscrowd": 0
            }
            annotations.append(annotation1)
            annot_count += 1

        if bbox2 != (0, 0, 0, 0):
            x2, y2, w2, h2 = bbox2[1], bbox2[0], bbox2[3] - bbox2[1], bbox2[2] - bbox2[0]
            pose_info = {
                "position": positions_scissors[index].tolist(),
                "orientation": orientations_scissors[index].tolist(),
                "grip": grips_scissors[index]
            }
            annotation2 = {
                "id": annot_count,
                "image_id": index,
                "category_id": 2,
                "bbox": [x2, y2, w2, h2],
                "pose": pose_info,
                "area": w2 * h2,
                "segmentation": [serialize_polygon(polygon) for polygon in polygons2],
                "iscrowd": 0
            }
            annotations.append(annotation2)
            annot_count += 1

    coco_format_json = {
        "categories": categories,
        "images": images,
        "annotations": annotations
    }

    json_str = json.dumps(coco_format_json, indent=4)
    json_str = json_str.replace('\n                    ', '')

    with open(output_json, 'w') as json_file:
        json_file.write(json_str)

    print(f"COCO JSON file saved to {output_json}")


if __name__ == "__main__":
    args = parse_args()
    args.base_dir = r'F:\YOON\debugger\jaxa_100samples_720'
    args.output_json = os.path.join(args.base_dir, 'annotations.json')
    args.bbox_obtain = True
    args.poly_obtain = True
    print(args)
    create_coco_json(args.base_dir, args.output_json, args.bbox_obtain, args.poly_obtain)
