from typing import Dict
import json
import datetime

def coco_to_ufo(file: Dict, output_path: str) -> None:
    for annotation in file['annotations']:
        file_info = file['images'][int(annotation['image_id'])-1]
        image_name = file_info['file_name']
        if image_name not in ufo['images']:
            ufo['images'][image_name] = {
                "paragraphs": {},
                "words": {},
                "chars": {},
                "img_w": file_info["width"],
                "img_h": file_info["height"],
                "tags": ["re-annotated"],
                "relations": {},
                "annotation_log": {
                    "worker": "",
                    "timestamp": now,
                    "tool_version": "LabelMe or CVAT",
                    "source": None
                    },
                "license_tag": {
                    "usability": True,
                    "public": False,
                    "commercial": True,
                    "type": None,
                    "holder": "Upstage"
                    }
                }
            anno_id = 1
        ufo['images'][image_name]['words'][str(anno_id).zfill(4)] = {
            "transcription": "",
            "points": [annotation['segmentation'][0][i:i+2] for i in range(0, len(annotation['segmentation'][0]), 2)],
            "orientation": "Horizontal",
            "language": None,
            "tags": None,
            "confidence": None,
            "illegibility": False
        }
        anno_id += 1

    with open(output_path, "w") as f:
        json.dump(ufo, f, indent=4)


now = datetime.datetime.now()
now = now.strftime('%Y-%m-%d %H:%M:%S')

input_path = '../data/medical/ufo/train_coco.json'
output_path = '../data/medical/ufo/train_a.json'

ufo = {
    'images': {}
}

with open(input_path, 'r') as f:
    file = json.load(f)
coco_to_ufo(file, output_path)