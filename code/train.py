import os 
import glob
import random
import datetime
import argparse 
from pathlib import Path
from importlib import import_module 

import numpy as np 
from tqdm.auto import tqdm
import albumentations as A 

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import DataLoader 

import wandb 
from dataset import XRayDataset 

from utils.loss import create_criterion 
from utils.augmentation import create_transforms 
from dataset import XRayDataset 

CLASSES = [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ] 

# seed 고정이 확실히 되는지 확인할 필요 있음 (동일한 조건에서 학습한 결과가 조금씩 다르게 나옴...) 
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)  


def save_model(model, epoch):
    '''
        mean_dice 기준 상위 5개 weight만 저장하는 함수 
    '''
    file_name = f'epoch{epoch+1:03d}.pth'
    output_folder_path = os.path.join(args.saved_model_dir, args.name)
    output_file_path = os.path.join(output_folder_path, file_name)
    
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)
    
    sorted_file_list = sorted(os.listdir(output_folder_path))
    if len(sorted_file_list) >= 5:
        os.remove(os.path.join(output_folder_path, sorted_file_list[0]))
        
    torch.save(model, output_file_path) 
    

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)


def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = torch.from_numpy(images).cuda(), torch.from_numpy(masks).cuda()       
            model = model.cuda()
            
            outputs = model(images)['out']  # models.segmentation.fcn_resnet50() 으로 학습시 사용 (baseline)
            # outputs = model(images)  # 미션 코드로 학습시 사용 
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # restore original size
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(outputs, masks)
            dices.append(dice)
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    
    return avg_dice


# shared memory 문제를 해결하기 위한 함수 
def customcollatefn(sample):

    img, label = list(zip(*sample))

    img = np.array(img, dtype=np.float32)
    label = np.array(label, dtype=np.float32)

    return img, label


def train(IMAGE_ROOT, LABEL_ROOT, SAVED_MODEL, args): 
    # Set device 
    set_seed(args.seed)  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = os.path.join(SAVED_MODEL, args.name)
    
    # Get agmentation 
    tf = create_transforms(args.augmentation)
    val_tf = A.Compose([A.Resize(512, 512)])
    
    # Make dataset 
    train_dataset = XRayDataset(IMAGE_ROOT, LABEL_ROOT, args.fold_num, is_train=True, transforms=tf)
    valid_dataset = XRayDataset(IMAGE_ROOT, LABEL_ROOT, args.fold_num, is_train=False, transforms=val_tf)
    
    # Load Data 
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size,
        collate_fn = customcollatefn,
        shuffle=True,
        num_workers=8,
        drop_last=True,  
    )

    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=2,  # 1 
        collate_fn = customcollatefn,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )
    
    # Define model  
    model_module = getattr(import_module(f"models.{args.model}"), args.model_class)  # default: BaseModel
    model = model_module().to(device)
    
    # Define Loss & optimizer 
    criterion = create_criterion(args.criterion)
    
    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = opt_module(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-6
    )
    # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5) # 구현 필요 
    
    # Train model
    print(f'Start training..')
    
    # n_class = len(CLASSES)
    best_dice = 0.
    
    for epoch in range(args.epochs):
        model.train()

        loss_value = 0
        for step, (images, masks) in enumerate(train_loader):            
            # gpu 연산을 위해 device 할당
            images, masks = torch.from_numpy(images).cuda(), torch.from_numpy(masks).cuda()
            model = model.cuda()
            
            # inference
            outputs = model(images)['out'] 
            # outputs = model(images)
            
            # loss 계산
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_value += loss.item()
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                train_loss = loss_value / 25
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{args.epochs}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(train_loss,4)}'
                )
                wandb.log({
                    "Train/loss": train_loss,
                })
                loss_value = 0
             
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % args.log_interval == 0:
            dice = validation(epoch + 1, model, valid_loader, criterion)
            wandb.log({
                "Val/mean_dice": dice,
            })
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {args.saved_model_dir}/{args.name}")
                best_dice = dice
                save_model(model, epoch) 

if __name__ == '__main__': 
    wandb.init(project="segmentation", reinit=True)  # team project 추가 필요 
    parser = argparse.ArgumentParser() 
    
    parser.add_argument('--image_root', type=str, default="/opt/ml/input/data/train/DCM")
    parser.add_argument('--label_root', type=str, default="/opt/ml/input/data/train/outputs_json")
    parser.add_argument('--saved_model_dir', type=str, default="/opt/ml/input/code/saved_model") 

    parser.add_argument('--seed', type=int, default=21, help='random seed (default: 21)') 
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)') 
    parser.add_argument('--fold_num', type=int, default=0, help='fold number (default: 0)')
    # parser.add_argument('--dataset', type=str, default='XRayDataset', help='dataset augmentation type (default: XRayDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 64)') 
    parser.add_argument('--log_interval', type=int, default=1, help='how many batches to wait before logging training status')
    parser.add_argument('--model', type=str, default='fcn', help='model type (default: fcn)') 
    parser.add_argument('--model_class', type=str, default='FCN_Resnet50', help='model class type (default: FCN_Resnet50)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 1e-4)')
    parser.add_argument('--criterion', type=str, default='BCEWithLogitsLoss', help='criterion type (default: cross_entropy)')
    parser.add_argument('--name', default='exp', help='model save at ./saved_model/{name}')


    args = parser.parse_args()
      
    wandb.run.name = args.name
    image_root = args.image_root 
    label_root = args.label_root 
    saved_model_dir = args.saved_model_dir 
    train(image_root, label_root, saved_model_dir, args)