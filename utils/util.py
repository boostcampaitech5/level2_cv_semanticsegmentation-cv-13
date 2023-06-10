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

class ElapsedTime():
    def __init__(self, func):
        self.func = func
    
    def __call__(self):
        tic = time.time()
        self.func()
        toc = time.time()
        print(f'elapsed time: {toc - tic}s')