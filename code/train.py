import os
import os.path as osp
import time
import math
import random
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
from deteval import calc_deteval_metrics
from utils import get_gt_bboxes, get_pred_bboxes, seed_everything, AverageMeter

import albumentations as A
import numpy as np

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '../data/medical'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'trained_models'))

    parser.add_argument('--seed', type=int, default=137)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)
    
    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--ignore_tags', type=list, default=['masked', 'excluded-region', 'maintable', 'stamp'])

    parser.add_argument('-m', '--mode', type=str, default='on', help='wandb logging mode(on: online, off: disabled)')
    parser.add_argument('-p', '--project', type=str, default='datacentric', help='wandb project name')
    parser.add_argument('-d', '--data', default='original', type=str, help='description about dataset')

    parser.add_argument("--optimizer", type=str, default='adam', choices=['adam', 'adamW'])
    parser.add_argument("--scheduler", type=str, default='cosine', choices=['multistep', 'cosine'])
    parser.add_argument("--resume", type=str, default=None, choices=[None, 'resume', 'finetune'])
    
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def do_training(args):
    
    ### Train Loader ###
    train_dataset = SceneTextDataset(
        args.data_dir,
        split='train_split',
        json_name='train_split.json',
        image_size=args.image_size,
        crop_size=args.input_size,
        ignore_tags=args.ignore_tags
    )
    train_dataset = EASTDataset(train_dataset)
    train_num_batches = math.ceil(len(train_dataset) / args.batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    ### Val Loader ###
    ''' 아래 코드는 val loss를 위한 코드입니다.'''
    '''
    val_dataset = SceneTextDataset(
        args.data_dir,
        split='valid_split',
        train_val='valid_split.json',
        image_size=args.image_size,
        crop_size=args.image_size,
        ignore_tags=args.ignore_tags,
        color_jitter=False,
    )
    val_dataset = EASTDataset(val_dataset)
    val_num_batches = math.ceil(len(val_dataset) / args.batch_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    '''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    
    ### Resume or finetune ###
    if args.resume == "resume":
        checkpoint = torch.load(osp.join(args.model_dir, "latest.pth"))
        model.load_state_dict(checkpoint)
    elif args.resume == "finetune":
        checkpoint = torch.load(osp.join(args.model_dir, "best.pth"))
        model.load_state_dict(checkpoint)
    
    ### Optimizer ###
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    ### Scheduler ###
    if args.scheduler == "multistep":
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.max_epoch // 2, args.max_epoch // 2 * 2], gamma=0.1)
    elif args.scheduler == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=0)
        
    ### WandB ###
    if args.mode == 'on':
        wandb.init(
            project=args.project,
            entity='nae-don-nae-san',
            group=args.data,
            name=f'{args.max_epoch}e_{args.optimizer}_{args.scheduler}_{args.learning_rate}'
        )
        wandb.config.update(args)
        wandb.watch(model)
        
    ### Train ###
    best_val_loss = np.inf
    best_f1_score = 0

    train_loss = AverageMeter()
    val_loss = AverageMeter()
    
    model.train()
    for epoch in range(args.max_epoch):
        epoch_start = time.time()
        train_loss.reset()
        with tqdm(total=train_num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description(f'[Epoch {epoch + 1}]')

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss.update(loss.item())

                pbar.update(1)
                train_dict = {
                    'train total loss': train_loss.avg,
                    'train cls loss': extra_info['cls_loss'],
                    'train angle loss': extra_info['angle_loss'],
                    'train iou loss': extra_info['iou_loss']
                }
                pbar.set_postfix(train_dict)
                if args.mode == 'on':    
                    wandb.log(train_dict, step=epoch)

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            train_loss.avg, timedelta(seconds=time.time() - epoch_start)))

        ### Val ###
        # 매 val_interval 에폭마다, 마지막 5에폭 이후 validation 수행
        if (epoch + 1) % args.val_interval == 0 or epoch >= args.max_epoch - 5:
            with torch.no_grad():
                
                ''' 아래 코드는 val loss를 위한 코드입니다.'''
                '''
                model.eval()
                epoch_start = time.time()
                with tqdm(total=val_num_batches) as pbar:
                    for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                        pbar.set_description('Evaluate')
                        loss, extra_info = model.train_step(
                            img, gt_score_map, gt_geo_map, roi_mask
                        )
                        val_loss.update(loss.item())
                        
                        pbar.update(1)
                        val_dict = {
                            "val total loss": val_loss.avg,
                            "val cls loss": extra_info["cls_loss"],
                            "val angle loss": extra_info["angle_loss"],
                            "val iou loss": extra_info["iou_loss"],
                        }
                        pbar.set_postfix(val_dict)
                        if args.mode == 'on':
                            wandb.log(val_dict, step=epoch)
                            
                if val_loss.avg < best_val_loss:
                    print(f"New best model for val loss : {val_loss.avg}! saving the best model..")
                    bestpt_fpath = osp.join(args.model_dir, "best.pth")
                    torch.save(model.state_dict(), bestpt_fpath)
                    best_val_loss = val_loss.avg
                '''
                
                ''' 아래 코드는 val f1 score를 위한 코드입니다. '''
                print("Calculating validation results...")
                valid_images = [f for f in os.listdir(osp.join(args.data_dir, 'img/valid_split/')) if f.endswith('.jpg')]

                pred_bboxes_dict = get_pred_bboxes(model, args.data_dir, valid_images, args.input_size, args.batch_size, split='valid_split')            
                gt_bboxes_dict = get_gt_bboxes(args.data_dir, json_file='ufo/valid_split.json', valid_images=valid_images)

                result = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict)
                total_result = result['total']
                precision, recall = total_result['precision'], total_result['recall']
                f1_score = 2*precision*recall/(precision+recall)
                print(f'Precision: {precision} Recall: {recall} F1 Score: {f1_score}')
                
                val_dict = {
                            'val precision': precision,
                            'val recall': recall,
                            'val f1_score': f1_score
                        }
                if args.mode == 'on':
                    wandb.log(val_dict, step=epoch)
                
                ### Save Best Model ###
                if best_f1_score < f1_score:
                    print(f"New best model for f1 score : {f1_score}! saving the best model..")
                    bestpt_fpath = osp.join(args.model_dir, 'best.pth')
                    torch.save(model.state_dict(), bestpt_fpath)
                    best_f1_score = f1_score

        # Save the Lastest Model
        if (epoch + 1) % args.save_interval == 0:
            ckpt_fpath = osp.join(args.model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)

    if args.mode == 'on':
        wandb.alert('Training Task Finished', f"TRAIN_LOSS: {train_loss:.4f}")
        wandb.finish()


def main(args):
    do_training(args)

if __name__ == '__main__':
    args = parse_args()
    
    seed_everything(args.seed)
    
    if not osp.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    main(args)
