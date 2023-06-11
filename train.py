import logging
import wandb
import time
import os
import json
import torch
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
from utils.util import plot_confusion_matrix,toConfusionMatrix, calculateScore,dice_coef
import tqdm

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


def train(model,accelerator, dataloader, criterion, optimizer,log_interval, args) -> dict:   
    losses_m = AverageMeter()
    interval_time = 0
    
    dices = []
    dice_per_batch = 0
    thr = 0.5
    
    model.train()
    optimizer.zero_grad()
    print(len(dataloader))
    for idx, (images, masks) in enumerate(dataloader):
        with accelerator.accumulate(model):
            tic = time.time()
            images, masks = images, masks

            # predict
            outputs = model(images)['out']
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
                
            # restore original size
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
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
            
            toc = time.time()
            interval_time += toc - tic
            if idx % log_interval == 0 and idx != 0: 
                _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                            'Dice: {dice:.3f} '
                            'LR: {lr:.3e} '
                            'Time: {batch_time:.3f}s'.format(idx+1, len(dataloader), 
                                                                                    loss       = losses_m, 
                                                                                    dice       = torch.mean(dice_per_batch).item(), 
                                                                                    lr         = optimizer.param_groups[0]['lr'],
                                                                                    batch_time = interval_time))
                interval_time = 0
    
    return OrderedDict([('train_dices',torch.mean(dice_per_batch).item()), ('loss',losses_m.avg)])
    

def val(model, dataloader, accelerator, criterion,log_interval, args) -> dict:

    total_loss = 0
    thr = 0.5
    best_dice = 0.
    dices = []
    interval_time = 5
    
    model.eval()
    with torch.no_grad():
        for idx, (images, masks) in enumerate(dataloader):
            with accelerator.accumulate(model):
                tic = time.time()
                images, masks = images, masks
                
                # predict
                outputs = model(images)['out']
                
                output_h, output_w = outputs.size(-2), outputs.size(-1)
                mask_h, mask_w = masks.size(-2), masks.size(-1)
                
                # restore original size
                if output_h != mask_h or output_w != mask_w:
                    outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
                
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
                toc = time.time()
                interval_time += toc - tic
                if idx % log_interval == 0 and idx != 0: 
                    _logger.info('VAL [%d/%d]: Loss: %.3f | Dice: %.3f%%' 
                                 'Time: {batch_time:.3f}s'.format(idx+1, len(dataloader), total_loss/(idx+1), torch.mean(dice_per_batch).item(),interval_time))

            dices = torch.cat(dices, 0)
            dices_per_class = torch.mean(dices, 0)
    return OrderedDict([('dice', torch.mean(dices_per_class).item()), ('loss',total_loss/len(dataloader))])


def fit(model, trainloader, valloader,  criterion, optimizer, lr_scheduler, accelerator, savedir: str, args) -> None:

    best_dice = 0
    step = 0
    log_interval = 1

    
    for epoch in range(args.epochs):
        _logger.info(f'\nEpoch: {epoch+1}/{args.epochs}')
        # train_metrics = train(model,accelerator, trainloader, criterion, optimizer, log_interval, args) 
        val_metrics = val(model, valloader, accelerator, criterion, log_interval,args)

        # wandb

        metrics = OrderedDict(lr=optimizer.param_groups[0]['lr'])
        # metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
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
            #save confusion_matrix
            # if args.use_cm:
            #     fig = plot_confusion_matrix(val_metrics['cm'],args.num_classes)
            #     if args.use_wandb:
            #         wandb.log({'Confusion Matrix': wandb.Image(fig, caption=f"Epoch-{epoch}")},step=epoch)
    
    
    _logger.info('Best Metric: {0:.3%} (epoch {1:})'.format(state['best_dice'], state['best_epoch']))
    
