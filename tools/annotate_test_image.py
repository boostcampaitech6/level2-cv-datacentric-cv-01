import os
import os.path as osp
import json
import cv2
import numpy as np

palette_rgb = [
    (255, 0, 0),      # Red
    (208, 255, 0),    # Yellow-green
    (0, 116, 255),    # Light blue
    (184, 0, 255),    # Purple
    (255, 147, 0),    # Orange
    (61, 255, 0),     # Green
    (0, 255, 245),    # Cyan
    (37, 0, 255),     # Blue
    (0, 255, 92),     # Green-cyan
    (255, 0, 171)     # Pink
]

# Convert RGB to BGR
palette = [(b, g, r) for (r, g, b) in palette_rgb]


"""
result_path : inference한 파일을 저장한 디렉토리를 지정합니다.
test_img_path : test 데이터셋 디렉토리의 경로를 지정합니다.
save_dir : 저장할 디렉토리 경로를 지정합니다.
"""
result_path = '../data/medical/inference'
test_img_path = '../data/medical/img/test'
save_dir = '../../annotate_dir'
os.makedirs(save_dir, exist_ok=True)

annotations = []
for js in os.listdir(result_path):
    with open(osp.join(result_path,js), 'r') as f:
        annotations.append(json.load(f))

for img in annotations[0]['images']:

    image = cv2.imread(osp.join(test_img_path,img))

    for idx in range(len(annotations)):
        for box_num in annotations[idx]['images'][img]['words']:
            
            points = annotations[idx]['images'][img]['words'][box_num]['points']
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))

            cv2.polylines(image, [pts], isClosed=True, color=palette[idx], thickness=2)
    
    cv2.imwrite(osp.join(save_dir,img),image)
