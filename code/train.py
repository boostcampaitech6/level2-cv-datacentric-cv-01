import os
import os.path as osp
import time
import math
import json
import random
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from optimizer import optim
from scheduler import sched
from tqdm import tqdm
import wandb

from east_dataset import EASTDataset
from dataset import SceneTextDataset, PickleDataset
from model import EAST
from deteval import calc_deteval_metrics
from utils import get_gt_bboxes, get_pred_bboxes, seed_everything, AverageMeter

import albumentations as A
import numpy as np

def parse_args():
    parser = ArgumentParser()

     # pkl 데이터셋 경로
    parser.add_argument('--train_dataset_dir', type=str, default="/data/ephemeral/home/level2-cv-datacentric-cv-01/data/medical/pickle/[1024, 1536, 2048]_cs[256, 512, 1024, 2048]_aug['CJ', 'GB', 'N']/train/")
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '../data/medical'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'trained_models'))
    parser.add_argument('--seed', type=int, default=137)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--device', default='cuda:0' if cuda.is_available() else 'cpu')
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
    parser.add_argument('-d', '--data', default='pickle', type=str, help='description about dataset', choices=['original', 'pickle'])
    parser.add_argument("--optimizer", type=str, default='Adam', choices=['adam', 'adamW'])
    parser.add_argument("--scheduler", type=str, default='cosine', choices=['multistep', 'cosine'])
    parser.add_argument("--resume", type=str, default=None, choices=[None, 'resume', 'finetune'])
    
    args = parser.parse_args()

    if args.data == 'original':
        args.data_name = 'original'
        args.save_dir = os.path.join(args.model_dir, f'{args.max_epoch}e_{args.optimizer}_{args.scheduler}_{args.learning_rate}')
    elif args.data == 'pickle':
        args.data_name = args.train_dataset_dir.split('/')[-3]
        args.save_dir = os.path.join(args.model_dir, f'{args.max_epoch}e_{args.optimizer}_{args.scheduler}_{args.learning_rate}_{args.data_name}')
    os.makedirs(args.save_dir, exist_ok=True)

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def do_training(args):
    
    ### Train Loader ###
    if args.data == 'original':
        train_dataset = SceneTextDataset(
            args.data_dir,
            split='train',
            json_name=f'train{args.fold}.json',
            image_size=args.image_size,
            crop_size=args.input_size,
            ignore_tags=args.ignore_tags,
            pin_memory=True,
        )
        train_dataset = EASTDataset(train_dataset)
        
    elif args.data == 'pickle':
        train_dataset = PickleDataset(args.train_dataset_dir)
        
    
    train_num_batches = math.ceil(len(train_dataset) / args.batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    ### Val Loader ###
    ''' 아래 코드는 val loss를 위한 코드입니다.'''
    '''
    val_dataset = SceneTextDataset(
        args.data_dir,
        split='valid',
        train_val=f'valid{fold}.json',
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
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    
    ### Resume or finetune ###
    if args.resume == "resume":
        checkpoint = torch.load(osp.join(args.save_dir, "latest.pth"))
        model.load_state_dict(checkpoint)
    elif args.resume == "finetune":
        checkpoint = torch.load(osp.join(args.save_dir, "best.pth"))
        model.load_state_dict(checkpoint)
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim(args, trainable_params)
    scheduler = sched(args, optimizer)
      
    ### WandB ###
    if args.mode == 'on':
        wandb.init(
            project=args.project,
            entity='nae-don-nae-san',
            group=args.data,
            name=f'{args.max_epoch}e_{args.optimizer}_{args.scheduler}_{args.learning_rate}_{args.data_name}'
        )
        wandb.config.update(args)
        wandb.watch(model)
        
    ### Train ###
    best_val_loss = np.inf
    best_f1_score = 0

    train_loss = AverageMeter()
    val_loss = AverageMeter()
    
    model.train()
    total_start_time = time.time()
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
        epoch_duration = time.time() - epoch_start
        
        print('Mean loss: {:.4f} | Elapsed time: {} |'.format(
            train_loss.avg, timedelta(seconds=epoch_duration)))

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
                valid_json_file = f'ufo/valid{args.fold}.json'
                with open(osp.join(args.data_dir, valid_json_file), 'r', encoding='utf-8') as file:
                    data = json.load(file)
                valid_images = list(data['images'].keys())

                pred_bboxes_dict = get_pred_bboxes(model, args.data_dir, valid_images, args.input_size, args.batch_size, split='train')            
                gt_bboxes_dict = get_gt_bboxes(args.data_dir, json_file=valid_json_file, valid_images=valid_images)

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
                    # bestpt_fpath = osp.join(args.model_dir, 'best.pth')
                    bestpt_fpath = osp.join(args.save_dir, 'best.pth')
                    torch.save(model.state_dict(), bestpt_fpath)
                    best_f1_score = f1_score

            elapsed_time = time.time() - total_start_time
            estimated_time_left = elapsed_time / (epoch + 1) * (args.max_epoch - epoch - 1)
            # 예상 종료 시간을 현재 시간 기준으로 변환
            
            eta = str(timedelta(seconds=estimated_time_left))
            print(f'Epoch {epoch + 1} Validation Finised | Left ETA: {eta}')
        
        # Save the Lastest Model
        if (epoch + 1) % args.save_interval == 0:
            ckpt_fpath = osp.join(args.save_dir, 'latest.pth')
            # ckpt_fpath = osp.join(args.model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)
        
        total_duration = time.time() - total_start_time
        # print('Mean loss: {:.4f} | Elapsed time: {} |'.format(
        #     train_loss.avg, timedelta(seconds=epoch_duration)))

    if args.mode == 'on':
        wandb.alert('Training Task Finished', f"TRAIN_LOSS: {train_loss.avg:.4f}")
        wandb.finish()

def main(args):
    do_training(args)

if __name__ == '__main__':
    args = parse_args()
    print(args)
    seed_everything(args.seed)

    main(args)
