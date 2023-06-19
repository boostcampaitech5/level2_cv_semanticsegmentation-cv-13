#임시로 여기 작성
import torch
import matplotlib as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import time

def plot_confusion_matrix(cm, num_classes, normalize=False, save_path=None):
    plt.clf()
    if normalize:
        n_total = torch.sum(cm, 1).view(num_classes, 1)
        np_cm = cm / n_total
        np_cm = np_cm.numpy()
        ax = sns.heatmap(np_cm, annot=True, cmap='Blues', linewidth=.5,
                        fmt=".2f", annot_kws = {'size' : 6})
    else:
        np_cm = cm.numpy()
        ax = sns.heatmap(np_cm, annot=True, cmap='Blues', linewidth=.5,
                        fmt="d", annot_kws = {'size' : 6})

    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')
    ax.xaxis.set_ticklabels([i for i in range(num_classes)])
    ax.xaxis.tick_top()
    ax.yaxis.set_ticklabels([i for i in range(num_classes)])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    
    return ax


def toConfusionMatrix(y_pred, y_label, num_classes:int) -> np.ndarray:
    #cm[y_pred][y_gt]
    cm = confusion_matrix(y_label, y_pred, labels = np.arange(num_classes).tolist())

    return cm


def calculateScore(y_pred, y_label, num_classes:int) -> float:

    return f1_score(y_label, y_pred, labels=np.arange(num_classes), average='micro')

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)


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

class ElapsedTime():
    def __init__(self, func):
        self.func = func
    
    def __call__(self, *args, **kwargs):
        tic = time.time()
        result = self.func(*args, **kwargs)
        toc = time.time()
        print(f"elapsed time running function '{self.func.__name__}': {toc - tic}s")
        return result
		
def customcollatefn(sample):

    img,label = list(zip(*sample))

    img = np.array(img,dtype=np.float32)

    label = np.array(label,dtype=np.float32)
    
    return img,label