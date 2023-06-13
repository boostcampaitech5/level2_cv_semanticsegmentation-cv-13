import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


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