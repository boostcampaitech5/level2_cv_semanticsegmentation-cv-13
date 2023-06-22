import argparse
import multiprocessing
import os
from tqdm.auto import tqdm
from importlib import import_module

import pandas as pd
import numpy as np 
import cv2 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import albumentations as A 

from dataset import XRayInferenceDataset 


def load_model(saved_model_path, args):
    model_module = getattr(import_module(f"models.{args.model}"), args.model_class)
    model = model_module()

    # model.load_state_dict(torch.load(model_path, map_location=device))
    model = torch.load(saved_model_path)
    return model

def encode_mask_to_rle(mask):
    '''
        mask: numpy array binary mask 
        1 - mask 
        0 - background
        Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def test(model, data_loader, thr=0.5):
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
    
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            # outputs = model(images)['out']
            outputs = model(images)
            
            
            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class


@torch.no_grad()
def inference(image_root, saved_model_dir, exp_name, file_name, args):
    
    saved_model_path = os.path.join(saved_model_dir, exp_name, file_name+'.pth') 
    output_path = os.path.join(saved_model_dir, exp_name, file_name+'_ouput.csv')
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(saved_model_path, args).to(device)
    model.eval()

    tf = A.Resize(1024, 1024)
    test_dataset = XRayInferenceDataset(image_root, tf)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False
    ) 
    
    rles, filename_and_class = test(model, test_loader)
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename] 
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    }) 
    
    df.to_csv(output_path, index=False)
    
    print(f"Inference Done! Inference result saved at {output_path}, lines: {len(df)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    # parser.add_argument('--resize', type=tuple, default=(128, 96), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='fcn', help='model type (default: fcn)') 
    parser.add_argument('--model_class', type=str, default='FCN_Resnet50', help='model class type (default: FCN_Resnet50)')

    # Container environment
    parser.add_argument('--image_root', type=str, default="/opt/ml/input/data/test/DCM")
    parser.add_argument('--saved_model_dir', type=str, default="/opt/ml/checkpoint/") 
    parser.add_argument('--exp_name', type=str, default='exp1_hrnet/')
    parser.add_argument('--file_name', type=str, default='epoch024')

    args = parser.parse_args()

    image_root = args.image_root
    saved_model_dir = args.saved_model_dir
    exp_name = args.exp_name
    file_name = args.file_name

    inference(image_root, saved_model_dir, exp_name, file_name, args)