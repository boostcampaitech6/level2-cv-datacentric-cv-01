import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
import wandb

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../data/medical'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--ignore_tags', type=list, default=['masked', 'excluded-region', 'maintable', 'stamp'])

    parser.add_argument('-m', '--mode', type=str, default='on', help='wandb logging mode(on: online, off: disabled)')
    parser.add_argument('-p', '--project', type=str, default='datacentric', help='wandb project name')
    parser.add_argument('-d', '--data', default='original', type=str, help='description about dataset')

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(args, data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, ignore_tags, mode, project, data):
    dataset = SceneTextDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()

    # WandB
    if mode == 'on':
        wandb.init(
            project=project,
            entity='nae-don-nae-san',
            group=data,
            name=f'{max_epoch}e_{learning_rate}'
        )
        wandb.config.update(args)
        wandb.watch(model)

    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))
                with torch.cuda.amp.autocast(enabled=False):
                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss = loss.item()
                epoch_loss += train_loss
                pbar.update(1)
                train_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(train_dict)

                if mode == 'on':    
                    wandb.log({'train_loss': train_loss, 'cls_loss': extra_info['cls_loss'], 
                            'angle_loss': extra_info['angle_loss'], 'iou_loss': extra_info['iou_loss']}, step=epoch)

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)

    if mode == 'on':
        # wandb.run.summary['best_f1'] = best_f1
        wandb.alert('Training Task Finished', f"TRAIN_LOSS: {train_loss:.4f}")
        wandb.finish()

def main(args):
    do_training(args, **args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
