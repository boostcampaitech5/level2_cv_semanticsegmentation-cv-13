import os
import torch
from .transform import get_transform
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import multiprocessing
from PIL import Image
import numpy as np
from sklearn.model_selection import GroupKFold
import json
import cv2
import time


class CustomDataset(Dataset):
    """Data loader를 만들기 위한 base dataset class"""

    def __init__(self, args:dict, train: bool = True):
        self.datadir = args.datadir
        self.train = train
        
        self.train_data = pd.read_csv(os.path.join(self.datadir, args.train_file))
        self.valid_data = pd.read_csv(os.path.join(self.datadir, args.valid_file))
        
        self.transform = args.transform


    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.valid_data)


    def __getitem__(self, idx):
        # Train Mode
        if self.train:
            img = Image.open(self.train_data.iloc[idx].ImageID)
            label = self.train_data.iloc[idx].ans
            if self.transform:
                transform, cfg = get_transform(self.transform)
                img = transform(img)
        
        # Validation Mode
        else:
            img = Image.open(self.valid_data.iloc[idx].ImageID)
            label = self.valid_data.iloc[idx].ans
            if self.transform:
                #이미지 사이즈 관련 Transform 만 진행
                val_trans = ["resize","centercrop","totensor","normalize"]
                trans_list = []
                for t in self.transform:
                    if t in val_trans:
                        trans_list.append(t)
                transform, cfg = get_transform(trans_list)
                img = transform(img)

        return img, torch.LongTensor([label]).squeeze()
    
    
class XRayDataset(Dataset):
    def __init__(self, args:dict, is_train=True):
        self.image_root = os.path.join(args.datadir,args.train_path)
        self.label_root = os.path.join(args.datadir,args.label_path)

        self.args = args
        self.class2ind = {v: i for i, v in enumerate(self.args.classes)}

        pngs = {
            os.path.relpath(os.path.join(root, fname), start=self.image_root)
            for root, _dirs, files in os.walk(self.image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        jsons = {
            os.path.relpath(os.path.join(root, fname), start=self.label_root)
            for root, _dirs, files in os.walk(self.label_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }
        pngs = sorted(pngs)
        jsons = sorted(jsons)
        
        pngs.remove('ID058/image1661392064531.png')
        pngs.remove('ID058/image1661392103627.png')
        jsons.remove('ID058/image1661392064531.json')
        jsons.remove('ID058/image1661392103627.json')
        
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        
        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        # dummy label
        ys = [0 for fname in _filenames]
        
        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == 0:
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                
                # skip i > 0
                break
        
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.translist = args.transform
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)
        
        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (self.args.num_classes, )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = self.class2ind[c]
            
            points = np.array(ann["points"])
            
            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.translist is not None:
            inputs = {"image": image, "mask": label} #if self.is_train else {"image": image}
            if self.is_train:
                transform, cfg = get_transform(self.translist)
            else:
                val_trans = ["resize","centercrop","totensor","normalize"]
                trans_list = []
                for t in self.translist:
                    if t in val_trans:
                        trans_list.append(t)
                transform, cfg = get_transform(trans_list)
                
            result = transform(**inputs)
            
            image = result["image"]
            label = result["mask"] #if self.is_train else label

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        label = label.transpose(2, 0, 1)

        return image, label

class XRayInferenceDataset(Dataset):
    def __init__(self, args:dict):
        self.image_root = os.path.join(args.datadir,args.test_path)
        
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=self.image_root)
            for root, _dirs, files in os.walk(self.image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }

        pngs = sorted(pngs)
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))
        
        self.filenames = _filenames
        self.translist = args.transform
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.translist is not None:
            inputs = {"image": image}
            val_trans = ["resize","centercrop","totensor","normalize"]
            trans_list = []
            for t in self.translist:
                if t in val_trans:
                    trans_list.append(t)
                    
            transform, cfg = get_transform(trans_list)
            result = transform(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        
        image = torch.from_numpy(image).float()
            
        return image, image_name

def create_dataloader(dataset, batch_size: int = 4, shuffle: bool = False,num_workers: int = multiprocessing.cpu_count() // 2):

    return DataLoader(
        dataset     = dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers
    )
    
