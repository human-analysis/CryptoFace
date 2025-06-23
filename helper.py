import os
import copy
import torch
import numpy as np
from utils import evaluate
from tqdm import tqdm
from multipledispatch import dispatch

test_data = ["lfw", "cfp_fp", "cplfw", "agedb_30", "calfw"]

@torch.no_grad()
def validate_emore(backbone, dataloaders, device=torch.device("cpu"), fuse=False):
    backbone.eval()
    acc_dict = {}
    threshold_dict = {}
    num_sample_dict = {}
    for dataname, loader in dataloaders.items():
        all_emb1, all_emb2, all_label = [], [], []
        for img1, img2, lab in loader:
            img1, img2, lab = img1.to(device), img2.to(device), lab.to(device)
            if fuse:
                emb1 = backbone.forward_fuse(img1)
                emb2 = backbone.forward_fuse(img2)    
            else:
                emb1 = backbone(img1)
                emb2 = backbone(img2)
            norm1 = torch.norm(emb1, 2, 1, True)
            emb1 = torch.divide(emb1, norm1)
            norm2 = torch.norm(emb2, 2, 1, True)
            emb2 = torch.divide(emb2, norm2)
            emb1, emb2, lab = emb1.cpu().numpy(), emb2.cpu().numpy(), lab.cpu().numpy()
            if emb1.ndim > 2:
                emb1 = np.reshape(emb1, (-1, emb1.shape[-1]))
                emb2 = np.reshape(emb2, (-1, emb2.shape[-1]))
                lab = lab.flatten()
            all_emb1.append(emb1)
            all_emb2.append(emb2)
            all_label.append(lab)
        
        all_emb1 = np.vstack(all_emb1)
        all_emb2 = np.vstack(all_emb2)
        all_label = np.concatenate(all_label)
        num_samples = len(all_emb1) + len(all_emb2)
        tpr, fpr, accuracy, best_thresholds = evaluate(all_emb1, all_emb2, all_label, nrof_folds=10)
        acc, best_threshold = accuracy.mean(), best_thresholds.mean()
        acc_dict[dataname+"_acc"] = acc
        threshold_dict[dataname+"_threshold"] = best_threshold
        num_sample_dict[dataname+"_sample"] = float(num_samples)
        acc_mean = sum([v for v in acc_dict.values()]) / len(acc_dict)
        res = {}
        res.update(acc_dict)
        res.update(threshold_dict)
        res.update(num_sample_dict)
        res.update({"acc": acc_mean})

    return acc_mean, res


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@dispatch(torch.nn.Module, str)
def model2txt(net, folder):
    for name, paramx in tqdm(net.state_dict().items(), desc="Extract weights", position=0, leave=True,  bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        param = copy.deepcopy(paramx)
        name = name.replace('.', '_') + '.txt'
        f = open(os.path.join(folder, name), 'w')
        if param.dim() == 0:
            f.write('{:.6e}\n'.format(0.))
        else:
            param = param.view(param.size(0), -1)
            for single in list(param):
                for num in list(single):
                    f.write('{:.6e}\n'.format(num.item()))
        f.close()

@dispatch(torch.nn.Parameter, str, str)
def model2txt(paramx, folder, name):
    param = copy.deepcopy(paramx.detach())
    f = open(os.path.join(folder, name), 'w')
    if param.ndim == 2:
        assert param.shape[0] == param.shape[1] == 256
        for i in range(256):
            param[i] = param[i].view(-1, 4).T.flatten()
    param = param.view(param.size(0), -1)
    for single in list(param):
        for num in list(single):
            f.write('{:.6e}\n'.format(num.item()))
    f.close()

def l2norm(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    y = np.sum(x ** 2, axis=1, keepdims=True)
    norm_inv = a * y ** 2 + b * y + c
    x_norm = x * norm_inv
    return x_norm