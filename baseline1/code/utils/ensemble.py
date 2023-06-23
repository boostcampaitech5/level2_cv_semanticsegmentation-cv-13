import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tqdm 
import torch 


def encode_mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs) 


def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)


def get_ensemble_mask(output_paths, start_idx, end_idx, thr):  
    
    total_masks = None 
    for output_path in output_paths:  
        rle_df = pd.read_csv(output_path) 
        rles = list(rle_df['rle'][start_idx:end_idx]) 
        print(len(rles))
        
        masks = [] 
        for rle in rles: 
            if rle != '' and type(rle) != float:
                mask = decode_rle_to_mask(rle, height=2048, width=2048)
            else: 
                mask = np.zeros((2048, 2048), dtype=np.uint8)
            masks.append(mask) 
        
        masks = np.stack(masks, 0)  
        
        if total_masks is None: 
            total_masks = masks
        else: 
            total_masks += masks 

    print('make binary mask...')
    total_masks[total_masks < thr] = 0  
    total_masks[total_masks >= thr] = 1 
    
    print(total_masks.shape, total_masks.sum(), total_masks.max())
    
    return total_masks  


def get_ensemble_rles(output_paths, split_num, thr):  
    step = 8700 // split_num 
    
    rles = [] 
    for count in range(split_num): 
        start_idx = count * step 
        end_idx = start_idx + step 
        
        print()
        print('load mask...')
        sub_masks = get_ensemble_mask(output_paths, start_idx, end_idx, thr) 
        
        
        print('sub_masks[sub_masks > 1] = ', sub_masks[sub_masks > 1].any())
        print('sub_mask.sum() = ', sub_masks.sum())
        print(f'encoding {count + 1}th masks...')
        for mask in sub_masks: 
            rle = encode_mask_to_rle(mask)
            rles.append(rle)
            
    return rles 

def get_ensemble_result(output_paths, save_path, split_num, thr):  
    '''
        ensemble 결과를 .csv 파일로 만드는 함수 
        
        Args: 
            output_paths: list, ensemble 할 .csv 파일들의 경로 
            save_path: str, ensemble 결과를 저장할 경로 
            split_num: int, 몇 번에 나눠서 ensemble 할 것인지 (메모리 이슈) 
            thr: int, binary mask로 만들기 위한 threshold 값 (몇 개의 모델이 해당 픽셀을 positive로 예측했는지)
    '''
    rles_df = pd.read_csv(output_paths[0])   

    rles = get_ensemble_rles(output_paths, split_num, thr)

    submission = rles_df.copy() 
    submission['rle'] = rles 
    submission.to_csv(save_path, index=False)  


def dice_coef(y_true, y_pred):  
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)


def get_best_model_idx(model_paths, data_loader=None, append_dice:list=None):
    dices_per_model = [] 
    for model_path in model_paths:
        model = torch.load(model_path)
        model = model.cuda() 
        model.eval()
        
        dices = [] 
        if data_loader is not None:
            with torch.no_grad(): 
            
                for i, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
                    images, masks = torch.from_numpy(images).cuda(), torch.from_numpy(masks).cuda()

                    pred = model(images)
                    
                    pred = torch.sigmoid(pred)
                    pred = (pred > 0.5).detach().cpu() 
                    masks = masks.detach().cpu() 
                    
                    dice = dice_coef(masks, pred)
                    dices.append(dice)
                    
            dices = torch.cat(dices, 0)
            dices_per_class = torch.mean(dices, 0)
            dices_per_model.append(dices_per_class.numpy())  
    
    if append_dice is not None: 
        for dices in append_dice: 
            dices_per_model.append(np.array(dices))
        
    for i, dices in enumerate(dices_per_model): 
        print(f'model {i}: ', list(dices))
        
    best_model_idx_per_class = np.argmax(np.array(dices_per_model), 0)
            
    return best_model_idx_per_class  


def get_weighted_ensemble_mask(output_paths, start_idx, end_idx, best_model_idx_per_class, weight, thr):  
    
    base = len(output_paths) - 1 + weight
    total_masks = None 
    for idx, output_path in enumerate(output_paths):   
        rle_df = pd.read_csv(output_path) 
        rles = list(rle_df['rle'][start_idx:end_idx]) 
        
        masks = [] 
        for best_idx, rle in zip(best_model_idx_per_class, rles): 
            if rle != '' and type(rle) != float:
                mask = decode_rle_to_mask(rle, height=2048, width=2048)
            else: 
                mask = np.zeros((2048, 2048), dtype=np.uint8)
                
            if idx == best_idx: 
                mask = mask * weight 
            
            masks.append(mask) 
        
        masks = np.stack(masks, 0)  
        
        if total_masks is None: 
            total_masks = masks
        else: 
            total_masks += masks 
            
    total_masks = total_masks / base 

    total_masks[total_masks < thr] = 0  
    total_masks[total_masks >= thr] = 1 
    
    return total_masks 


def get_weighed_ensemble_rles(output_paths, best_model_idx_per_class, weight, thr):  
    step = 8700 // 300 
    
    rles = [] 
    for count in tqdm(range(300)): 
        start_idx = count * step 
        end_idx = start_idx + step 
        
        sub_masks = get_weighted_ensemble_mask(output_paths, start_idx, end_idx, best_model_idx_per_class, weight, thr) 
        
        for mask in sub_masks: 
            rle = encode_mask_to_rle(mask)
            rles.append(rle)
            
    return rles 


def get_weighed_ensemble_result(output_paths, save_path, best_model_idx_per_class, weight, thr): 
    rles_df = pd.read_csv(output_paths[0])   
    
    rles = get_weighed_ensemble_rles(output_paths, best_model_idx_per_class, weight, thr)
    
    submission = rles_df.copy() 
    submission['rle'] = rles 
    submission.to_csv(save_path, index=False) 
    
    print()
    print('Done!')