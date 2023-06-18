import os 
import cv2 
import numpy as np 
import pandas as pd 
import json 

PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

def decode_rle_to_mask(rle, height, width):
    if isinstance(rle, float):
        return np.zeros((height, width), dtype=np.uint8)
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)


def make_mask(image, points):
    zero_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask = cv2.fillPoly(zero_mask, [points], 1)
    return mask  


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        # np.where(조건, 참일 때 값, 거짓일 때 값)
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c])   
    return image 


def get_masked_image_form_json(image_path, annotation_path, alpha=0.4, class_num=None):
    image = cv2.imread(image_path) 
    
    with open(annotation_path, "r") as f:
        annotations = json.load(f) 
    annotations = annotations['annotations']
    
    if class_num is not None:
        points = np.array(annotations[class_num]['points'])
        mask = make_mask(image, points)
        masked_image = apply_mask(image, mask, PALETTE[class_num], alpha=alpha) 
    else: 
        for class_ann in annotations:
            points = np.array(class_ann['points'])
            mask = make_mask(image, points)
            masked_image = apply_mask(image, mask, PALETTE[CLASS2IND[class_ann['label']]], alpha=alpha)
            
    return masked_image 


def get_masked_image_from_csv(image_path, csv_path, class_num=None, alpha=0.4, kind='mask'): 
    
    image_name = image_path.split('/')[-1]
    image = cv2.imread(image_path) 
    
    rles_df = pd.read_csv(csv_path)  
    
    rles = rles_df[rles_df['image_name'] == image_name]
    
    for i, rle in enumerate(rles['rle']):
        if class_num == None: 
            mask = decode_rle_to_mask(rle, 2048, 2048)
            if kind == 'mask':
                masked_image = apply_mask(image, mask, PALETTE[i], alpha=alpha)  
            elif kind == 'polyline': 
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
                masked_image = cv2.drawContours(image, contours, -1, PALETTE[i], 2, cv2.LINE_8, hierarchy, 100)
        else: 
            if class_num == i: 
                if kind == 'mask':
                    mask = decode_rle_to_mask(rle, 2048, 2048)
                    masked_image = apply_mask(image, mask, PALETTE[i], alpha=alpha)
                elif kind == 'polyline': 
                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
                    masked_image = cv2.drawContours(image, contours, -1, PALETTE[i], 2, cv2.LINE_8, hierarchy, 100)
                
    return masked_image