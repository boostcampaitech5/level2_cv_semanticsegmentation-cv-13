import logging
import wandb
import time
import os
import json
import torch
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
from utils.util import plot_confusion_matrix,toConfusionMatrix, calculateScore,dice_coef, ElapsedTime
from tqdm import tqdm

_logger = logging.getLogger('train')


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()


    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class cmMetter:
    #1epoch까지의 결과 저장
    def __init__(self):
        self.reset()


    def reset(self):
        self.pred = None
        self.label = None


    def update(self,pred,label):
        if type(self.pred) != np.ndarray:
            self.pred = pred.cpu().detach().numpy().reshape(-1)
            self.label = label.cpu().detach().numpy()
        else:
            self.pred = np.concatenate((self.pred,pred.cpu().detach().numpy().reshape(-1)))
            self.label = np.concatenate((self.label, label.cpu().detach().numpy()))

@ElapsedTime
def train(model,accelerator, dataloader, criterion, optimizer,log_interval, args) -> dict:   
    losses_m = AverageMeter()
    interval_time = 0
    
    dices = []
    dice_per_batch = 0
    thr = 0.5
    
    model.train()
    optimizer.zero_grad()

    pbar_train = tqdm(dataloader)   
    for idx, (images, masks) in enumerate(pbar_train):
        with accelerator.accumulate(model):
            images, masks = images, masks

            # predict
            outputs = model(images)['out']
            
            # get loss & loss backward
            loss = criterion(outputs, masks)
            accelerator.backward(loss)
            
            # loss update
            optimizer.step()
            optimizer.zero_grad()
            losses_m.update(loss.item())
            
            # accuracy
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(outputs, masks)
            dice_per_batch = torch.mean(dice, dim=0)
            
            pbar_train.set_postfix({'Dice': torch.mean(dice_per_batch).item(), 'Loss': losses_m.val})

    pbar_train.close()
    return OrderedDict([('dice',torch.mean(dice_per_batch).item()), ('loss',losses_m.avg)])
    
@ElapsedTime
def val(model, dataloader, accelerator, criterion, log_interval, args) -> dict:

    total_loss = 0
    thr = 0.5
    best_dice = 0.
    dices = []

    model.eval()
    with torch.no_grad():
        pbar_val = tqdm(dataloader)
        for idx, (images, masks) in enumerate(pbar_val):
            with accelerator.accumulate(model):
                images, masks = images, masks
                
                # predict
                outputs = model(images)['out']
                
                # get loss 
                loss = criterion(outputs, masks)
                
                # total loss and acc
                total_loss += loss.item()
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > thr).detach().cpu()
                masks = masks.detach().cpu()
                
                dice = dice_coef(outputs, masks)
                dice_per_batch = torch.mean(dice, dim=0)
                dices.append(dice)
				
            pbar_val.set_postfix({'Dice': torch.mean(dice_per_batch).item(), 'Loss': total_loss/(idx + 1)})

        dices = torch.cat(dices, 0)
        dices_per_class = torch.mean(dices, 0)
        pbar_val.close()
    return OrderedDict([('dice', torch.mean(dices_per_class).item()), ('loss',total_loss/len(dataloader))])


def fit(model, trainloader, valloader,  criterion, optimizer, lr_scheduler, accelerator, savedir: str, args) -> None:

    best_dice = 0
    step = 0
    log_interval = 5
    
    for epoch in range(args.epochs):
        _logger.info(f'\nEpoch: {epoch+1}/{args.epochs}')
        
        train_metrics = train(model, accelerator, trainloader, criterion, optimizer, log_interval, args)
        val_metrics = val(model, valloader, accelerator, criterion, log_interval,args)
        
        # wandb

        metrics = OrderedDict(lr=optimizer.param_groups[0]['lr'])
        metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
        metrics.update([('val_' + k, v) for k, v in val_metrics.items()])
        
        print(metrics)
        if args.use_wandb:
            print("wandb logging")
            wandb.log(metrics, step=epoch)

        step += 1

        # step scheduler
        # if lr_scheduler:
        #     lr_scheduler.step()

        # checkpoint
        if best_dice < val_metrics['dice']:
            # save results
            state = {'best_epoch':epoch, 'best_dice':val_metrics['dice']}
            json.dump(state, open(os.path.join(savedir, f'best_results.json'),'w'), indent=4)

            # save model
            torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
            
            _logger.info('Best dice {0:.3%} to {1:.3%}'.format(best_dice, val_metrics['dice']))

            best_dice = val_metrics['dice']
    
    
    _logger.info('Best Metric: {0:.3%} (epoch {1:})'.format(state['best_dice'], state['best_epoch']))
    
