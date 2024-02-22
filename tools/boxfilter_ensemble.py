import os
import os.path as osp
import json
from ensemble_boxes import *
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from ensemble_boxes import *
from tqdm import tqdm

# 박스 필터링을 위한 함수 정의
def intersection_area(box1, box2):
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)

    return inter_width * inter_height / box1_area


test_img_dir = '/data/ephemeral/home/data/medical/img/test' # 테스트 이미지 폴터 경로
json_path = '../../data/medical/json_folder' # 앙상블을 수행한 json 파일들의 폴더 경로
save_dir = './ensem_dir' # 저장할 폴더 경로

os.makedirs(save_dir, exist_ok=True)

annotations = []
for js in os.listdir(json_path):
    with open(osp.join(json_path,js), 'r') as f:
        annotations.append(json.load(f))

# wbf 앙상블을 위한 정의
iou_thr = 0.5
skip_box_thr = 0.0001
sigma = 0.1

# 겹치는 박스 영역에 대한 임계값 정의
min_area_thr = 0.005
max_area_thr = 0.99

# 겹치는 박스 개수에 대한 임계값 정의
min_num_thr = 2
max_num_thr = 1

save_dict = {"images" : {}}
scores = [float(score.split('.csv')[0]) for score in os.listdir(json_path)]


for image in tqdm(os.listdir(test_img_dir), desc="Processing images"):
    img = Image.open(osp.join(test_img_dir, image))


    save_dict['images'][image] = {"words" : {}}

    width, height = img.size
    
    boxes_list = []

    for idx in range(len(annotations)):

        for id in annotations[idx]['images'][image]['words']:
            pts = annotations[idx]['images'][image]['words'][id]['points']
            
            np_points = np.array(pts)
            x_coords = np_points[:, 0]
            y_coords = np_points[:, 1]

            x_min, x_max = x_coords.min()/width, x_coords.max()/width
            y_min, y_max = y_coords.min()/height, y_coords.max()/height

            if not 0<= x_min <=1 or not 0<= x_max <=1 or not 0<= y_min <=1 or not 0<= y_max <=1:
                continue
            
            boxes_list.append([x_min, y_min, x_max, y_max, 0, 0, scores[idx]])
    
    for i in range(len(boxes_list)):
        for j in range(len(boxes_list)):

            if boxes_list[i][-1] == boxes_list[j][-1]: # 동일한 모델인 경우 pass
                continue
            
            area = intersection_area(boxes_list[i], boxes_list[j])

            if min_area_thr < area: # 겹치는 경우
                boxes_list[i][4] += 1
            
            if area > max_area_thr: # 다른 박스에 포함되는 경우
                boxes_list[i][5] += 1
            
    boxes_dict = dict()

    # box filtering
    for b in boxes_list:
        if b[-1] not in boxes_dict:
            boxes_dict[b[-1]] = []

        if b[4] >= min_num_thr and b[5] <= max_num_thr:
            boxes_dict[b[-1]].append(b[:4])
        
    
    boxes_list_ens = []
    scores_list = []
    labels_list = []

    for scr in boxes_dict:
        boxes_list_ens.append(boxes_dict[scr])
        scores_list.append([scr]*len(boxes_dict[scr]))
        labels_list.append([1]*len(boxes_dict[scr]))

    boxes, _, _ = weighted_boxes_fusion(boxes_list_ens, scores_list, labels_list, iou_thr=iou_thr)
    
    for id,box in enumerate(boxes):
        x1, y1, x2, y2 = box
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
        point = [
            [x1,y1],
            [x2,y1],
            [x2,y2],
            [x1,y2]
        ]
        save_dict['images'][image]['words'][str(id)] = {'points' : point}


output_fname = 'ensemble.csv'
with open(osp.join(save_dir,output_fname), 'w') as f:
    json.dump(save_dict, f, indent=4)
