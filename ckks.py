import os
import copy
import torch
import random
import numpy as np
from tqdm import tqdm
from argument import CLI
import torch.backends.cudnn as cudnn
from helper import validate_emore
from models import build_model
from datasets import FaceEmore
from helper import model2txt
from utils import calculate_roc_train_test
from sklearn.model_selection import KFold
from helper import l2norm, count_parameters

def main(args):

    if int(torch.__version__[0]) > 1:
        torch.set_float32_matmul_precision('medium')
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    device = torch.device(f"cuda:{args.gpu}") if args.gpu is not None else torch.device("cpu")

    # output
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, f"cryptoface_{args.arch}{args.input_size}")
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    log = open(os.path.join(args.output_dir, "log.txt"), "w")

    # model
    backbone = build_model(args)
    backbone.to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt["backbone"])
    backbone.eval()
    print("#Params: {:d}".format(count_parameters(backbone)))
    print("#Params: {:d}".format(count_parameters(backbone)), file=log)

    # dataset
    lfw_loader, cfpfp_loader, cplfw_loader, agedb30_loader, calfw_loader = FaceEmore(args)
    test_loaders = {"lfw": lfw_loader, "cfp_fp": cfpfp_loader, "cplfw": cplfw_loader, "agedb_30": agedb30_loader, "calfw": calfw_loader}

    # evalute original model
    test_acc, test_metric = validate_emore(backbone, test_loaders, device)
    test_metric = sorted(test_metric.items())
    print("-"*10+"Evalaute PyTorch Model"+"-"*10)
    print("-"*10+"Evalaute PyTorch Model"+"-"*10, file=log)
    for k, v in test_metric:
        print("{:}: {:.4f}".format(k, v))
        print("{:}: {:.4f}".format(k, v), file=log)
    print("Acc: {:.4f}".format(test_acc))
    print("Acc: {:.4f}".format(test_acc), file=log)

    # evaluate fused model
    backbone.fuse()
    test_acc, test_metric = validate_emore(backbone, test_loaders, device, fuse=True)
    test_metric = sorted(test_metric.items())
    print("-"*10+"Evalaute Fuse PyTorch Model"+"-"*10)
    print("-"*10+"Evalaute Fuse PyTorch Model"+"-"*10, file=log)
    for k, v in test_metric:
        print("{:}: {:.4f}".format(k, v))
        print("{:}: {:.4f}".format(k, v), file=log)
    print("Acc: {:.4f}".format(test_acc))
    print("Acc: {:.4f}".format(test_acc), file=log)
    
    # write txt model
    for i in range(len(backbone.nets)):
        os.mkdir(args.output_dir+f"/net{i}")
        model2txt(backbone.nets[i], args.output_dir+f"/net{i}")
        model2txt(backbone.weights[i], args.output_dir+f"/net{i}", "weight.txt")
    model2txt(backbone.bias, args.output_dir, "bias.txt")
    
    # estimate threshold and polynomial approximation of 1/sqrt(x)
    poly_metric = {}
    poly_accuracy = []
    with torch.no_grad():
        for dataname, loader in test_loaders.items():
            poly_thresholds = []
            all_emb1, all_emb2, all_label = [], [], []
            for img1, img2, lab in tqdm(loader, desc=f"Estimate Polynomial {dataname}", position=0, leave=True,  bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
                img1, img2, lab = img1.to(device), img2.to(device), lab.to(device)
                emb1 = backbone.forward_fuse(img1)
                emb2 = backbone.forward_fuse(img2)
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
            all_emb1 = all_emb1.astype(np.float64)
            all_emb2 = all_emb2.astype(np.float64)
            all_emb1_sum = np.sum(all_emb1 ** 2, axis=1)
            all_emb2_sum = np.sum(all_emb2 ** 2, axis=1)
            
            k_fold = KFold(n_splits=10, shuffle=False)
            assert all_emb1.shape[0] == all_emb2.shape[0] == all_label.shape[0]
            nrof_pairs = all_emb1.shape[0]
            indices = np.arange(nrof_pairs)
            thresholds = np.arange(0, 4, 0.01)
            all_accuracy, all_best_threshold = [], []
            for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
                # train_set to estimate polynomial approximation of 1/sqrt(*)
                val = np.concatenate((all_emb1_sum[train_set], all_emb2_sum[train_set]))
                val = val[np.isfinite(val)]
                val = np.sort(val)[0:int(len(val)*0.99)]
                val_mean, val_std, val_min, val_max = np.mean(val), np.std(val), np.min(val), np.max(val)
                x1 = val_mean
                x2 = min(val_mean + val_std, val_max)
                x3 = max(val_mean - val_std, val_min)
                y1, y2, y3 = 1/np.sqrt(x1), 1/np.sqrt(x2), 1/np.sqrt(x3)
                a, b, c = np.linalg.solve(np.array([[x1**2, x1, 1], 
                                                    [x2**2, x2, 1], 
                                                    [x3**2, x3, 1]]), 
                                          np.array([y1, y2, y3]))
                emb1_norm, emb2_norm = l2norm(all_emb1, a, b, c), l2norm(all_emb2, a, b, c)
                tpr, fpr, accuracy, best_threshold = calculate_roc_train_test(thresholds, emb1_norm, emb2_norm, all_label, train_set, test_set, pca=0)
                poly_thresholds.append([x3, x1, x2, a, b, c, best_threshold])
                all_accuracy.append(accuracy)
                all_best_threshold.append(best_threshold)
            poly_metric.update({f"{dataname}_acc": np.mean(all_accuracy), f"{dataname}_threshold": np.mean(all_best_threshold)})
            poly_accuracy.append(np.mean(all_accuracy))
            poly_thresholds = np.asarray(poly_thresholds)
            np.savetxt(args.output_dir+f"/threshold_{dataname}.txt", poly_thresholds, fmt="%.6e", delimiter=",")
    poly_metric.update({f"acc": np.mean(poly_accuracy)})
    print("-"*10+"Evalaute Polynomial Model"+"-"*10)
    print("-"*10+"Evalaute Polynomial Model"+"-"*10, file=log)
    poly_metric = sorted(poly_metric.items())
    for k, v in poly_metric:
        print("{:}: {:.4f}".format(k, v))
        print("{:}: {:.4f}".format(k, v), file=log)
    log.close()


if __name__ == '__main__':
    args = CLI()
    main(args)
