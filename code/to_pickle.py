import pickle
from tqdm import tqdm
import os.path as osp

from east_dataset import EASTDataset
from dataset import SceneTextDataset

import albumentations as A

def main():
    data_dir = '/data/ephemeral/home/level2-cv-datacentric-cv-01/data/medical'
    image_size = 2048
    crop_size = 1024
    ignore_tags = ['masked', 'excluded-region', 'maintable', 'stamp']
    custom_augmentation = [
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.Blur(blur_limit=7, p=0.5),
        A.IAAAdditiveGaussianNoise(scale=(0.01*255, 0.05*255), p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=(0.7760271717131425, 0.7722186515548635, 0.7670997062399484), 
                    std=(0.17171108542242774, 0.17888224507630185, 0.18678791254805846), p=1.0)
    ]

    train_dataset = SceneTextDataset(
            root_dir=data_dir,
            split='train',
            json_name='train_split.json',
            image_size=image_size,
            crop_size=crop_size,
            ignore_tags=ignore_tags,
            custom_transform=custom_augmentation,
            color_jitter=False,
            normalize=False
        )
    train_dataset = EASTDataset(train_dataset)

    for i in tqdm(range(len(train_dataset))):
            g = train_dataset.__getitem__(i)
            with open(file=osp.join(data_dir, f"{i}.pkl"), mode="wb") as f:
                pickle.dump(g, f)
            
if __name__ == '__main__':
    main()