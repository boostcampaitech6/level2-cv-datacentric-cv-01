import pickle
from tqdm import tqdm
import os.path as osp

from east_dataset import EASTDataset
from dataset import SceneTextDataset

def main():
    data_dir = '/data/ephemeral/home/level2-cv-datacentric-cv-01/data/medical'
    image_size = 2048
    crop_size = 1024
    ignore_tags = ['masked', 'excluded-region', 'maintable', 'stamp']

    train_dataset = SceneTextDataset(
            root_dir=data_dir,
            split='train',
            train_val='train_split.json',
            image_size=image_size,
            crop_size=crop_size,
            ignore_tags=ignore_tags
        )
    train_dataset = EASTDataset(train_dataset)

    for i in tqdm(range(len(train_dataset))):
            g = train_dataset.__getitem__(i)
            with open(file=osp.join(data_dir, f"{i}.pkl"), mode="wb") as f:
                pickle.dump(g, f)
            
if __name__ == '__main__':
    main()