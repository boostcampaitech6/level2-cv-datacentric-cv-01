import pickle
from tqdm import tqdm
import os
import os.path as osp

from east_dataset import EASTDataset
from dataset import SceneTextDataset

import albumentations as A

def main():
    data_dir = '/data/ephemeral/home/level2-cv-datacentric-cv-01/data/medical'
    ignore_tags = ['masked', 'excluded-region', 'maintable', 'stamp']
    custom_augmentation_dict = {
        'CJ': A.ColorJitter(0.2, 0.2, 0.2, 0.2, p=0.5),
        'Default_CJ': A.ColorJitter(0.5, 0.5, 0.5, 0.25, p=0.5),
        'GB': A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        'B': A.Blur(blur_limit=7, p=0.5),
        'GN': A.GaussNoise(p=0.5),
        'HSV': A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        'RBC': A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        'C': A.CLAHE(p=0.5),
        'S&P': A.OneOf([
                A.MultiplicativeNoise(),
                A.MultiplicativeNoise(multiplier=(0.5, 1.5), per_channel=True),
                ], p=0.5),
        'N': A.Normalize(mean=(0.7760271717131425, 0.7722186515548635, 0.7670997062399484), 
                        std=(0.17171108542242774, 0.17888224507630185, 0.18678791254805846), p=1.0),
        'Default_N': A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), p=1.0)
    }

    # image_size = [1024]
    image_size = [1024, 1536, 2048, 4096, 8192]
    crop_size = [1024]
    # aug_select = ['CJ','N']
    # aug_select = ['CJ','C','N']
    aug_select = ['CJ','GB', 'HSV','N']
    # aug_select = ['Default_CJ','GB','Default_N']
    # aug_select = ['Default_CJ','GN','Default_N']
    # aug_select = ['Default_CJ','HSV','Default_N']

    custom_augmentation = []
    for s in aug_select:
        custom_augmentation.append(custom_augmentation_dict[s])

    pkl_dir = f'pickle/{image_size}_cs{crop_size}_aug{aug_select}/train/'
    # pkl_dir = f'pickle_is{image_size}_cs{crop_size}_aug{aug_select}/train/'
    
    # 경로 폴더 생성
    os.makedirs(osp.join(data_dir, pkl_dir), exist_ok=True)
    
    for i, i_size in enumerate(image_size):
        for j, c_size in enumerate(crop_size):
            if c_size > i_size:
                continue
            train_dataset = SceneTextDataset(
                    root_dir=data_dir,
                    split='train',
                    json_name='train_split.json',
                    image_size=i_size,
                    crop_size=c_size,
                    ignore_tags=ignore_tags,
                    custom_transform=A.Compose(custom_augmentation),
                    color_jitter=False,
                    normalize=False
                )
            train_dataset = EASTDataset(train_dataset)

            ds = len(train_dataset)
            for k in tqdm(range(ds)):
                data = train_dataset.__getitem__(k)
                if data is not None:
                    with open(file=osp.join(data_dir, pkl_dir, f"{ds*i+ds*j+k}.pkl"), mode="wb") as f:
                        pickle.dump(data, f)
            
if __name__ == '__main__':
    main()