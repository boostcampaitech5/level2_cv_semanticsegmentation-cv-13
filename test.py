import logging
import time
import os
import json
import torch
import numpy as np
import pandas as pd
from utils.util import plot_confusion_matrix, toConfusionMatrix, decode_rle_to_mask, encode_mask_to_rle
from datasets.dataset import  XRayInferenceDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import argparse

_logger = logging.getLogger('test')
SAVE_DIR = ""


def test(model, data_loader, args, thr=0.5):
    CLASS2IND = {v: i for i, v in enumerate(args.classes)}
    
    IND2CLASS = {v: k for k, v in CLASS2IND.items()}

    
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = args.num_classes

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            outputs = model(images)['out']
            
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

def run(args):
    thr = 0.5
    
    save_dir = args.savedir + "/exp4/"
    
    model = __import__('models.model', fromlist='model').__dict__[args.model_name](args.num_classes, **args.model_param)
    model_path = save_dir + "best_model.pt"
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    test_dataset = XRayInferenceDataset(args)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    print("model testing...")
    print(model_path)
    rles, filename_and_class = test(model, test_loader, args, thr)
    
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    
    df = pd.DataFrame({
    "image_name": image_name,
    "class": classes,
    "rle": rles,
    })
    
    print("result saving...")
    df.to_csv(save_dir + f"output_{args.exp_name}.csv", index=False)
    print("done!")
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Classification for Computer Vision")

    with open('config.json') as f:
        config = json.load(f)

    for key in config:
        parser_key = key.replace('_', '-')
        parser.add_argument(f'--{parser_key}', default=config[key], type=type(config[key]))

    args = parser.parse_args()
    run(args)
    