import os
import pickle
import argparse
import mxnet as mx
import numpy as np
import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
from sklearn.model_selection import KFold
from tqdm import tqdm
from einops import rearrange

FACE_DATASETS = OrderedDict([(0, "lfw"), (1, "cfp_fp"), (2, "cplfw"), (3, "agedb_30"), (4, "calfw")])

def WebFace4M(args):
    input_size = args.input_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    data_dir = args.train_data_dir
    trans = [transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    if input_size != 112:
        trans += [transforms.Resize(input_size, antialias=True)]
    transform = transforms.Compose(trans)
    dataset = ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)
    return loader


def FaceEmore(args):
    input_size = args.input_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    data_dir = args.test_data_dir
    dataset_list = [FaceBinDataset(data_dir, v_, input_size) for (k_, v_) in FACE_DATASETS.items()]
    loader_list = [DataLoader(dataset, batch_size, False, num_workers=num_workers, pin_memory=pin_memory) for dataset in dataset_list]
    return loader_list


class FaceBinDataset(torch.utils.data.Dataset):
    def __init__(self, bin_dir, data_name, input_size: int = 112):
        """
        :param bin_dir: directory of bin files
        :param data_name: one of [lfw, cfp_fp, cplfw, agedb_30, calfw]
        :param res: resolution
        """
        bin_path = os.path.join(bin_dir, f"{data_name}.bin")
        trans = [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        if input_size != 112:
            trans += [transforms.Resize(input_size, antialias=True)]
        self.transform = transforms.Compose(trans)
        self.bins, self.issame = pickle.load(open(bin_path, 'rb'), encoding='bytes')

    def __len__(self):
        """
        :return: num of pairs
        """
        return len(self.issame)

    def __getitem__(self, idx):

        img1 = mx.image.imdecode(self.bins[2 * idx]).asnumpy()
        img1 = self.transform(img1)
        img2 = mx.image.imdecode(self.bins[2 * idx + 1]).asnumpy()
        img2 = self.transform(img2)
        return img1, img2, self.issame[idx]
    

def image2txt(args):
    args.input_size  = 64
    args.batch_size = 1
    args.num_workers = 1
    args.pin_memory = False
    args.test_data_dir = args.data
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    lfw_loader, cfpfp_loader, cplfw_loader, agedb30_loader, calfw_loader = FaceEmore(args)
    test_loaders = {"lfw": lfw_loader, "cfp_fp": cfpfp_loader, "cplfw": cplfw_loader, "agedb_30": agedb30_loader, "calfw": calfw_loader}
    for dataname, loader in test_loaders.items():
        images1, images2, labels = [], [], []
        for img1, img2, lab in tqdm(loader, desc=f"{dataname}", position=0, leave=True,  bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            img1 = img1.view(img1.size(0), -1).numpy()
            img2 = img2.view(img2.size(0), -1).numpy()
            lab = lab.flatten().numpy()
            images1.append(img1)
            images2.append(img2)
            labels.append(lab)
        images1 = np.vstack(images1)
        images2 = np.vstack(images2)
        labels = np.vstack(labels)
        k_fold = KFold(n_splits=10, shuffle=False)
        assert images1.shape[0] == images2.shape[0] == labels.shape[0]
        nrof_pairs = images1.shape[0]
        indices = np.arange(nrof_pairs)
        out_dir = os.path.join(args.output, dataname)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        for fold_idx, (_, test_set) in enumerate(k_fold.split(indices)):
            np.savetxt(os.path.join(out_dir, f"fold{fold_idx}_images1.txt"), images1[test_set], fmt="%.6f", delimiter=",")
            np.savetxt(os.path.join(out_dir, f"fold{fold_idx}_images2.txt"), images2[test_set], fmt="%.6f", delimiter=",")
            np.savetxt(os.path.join(out_dir, f"fold{fold_idx}_labels.txt"), labels[test_set], fmt="%d", delimiter=",")


def image2txt_patch(args):
    args.batch_size = 1
    args.num_workers = 1
    args.pin_memory = False
    args.test_data_dir = args.data
    H, W = args.input_size // args.patch_size, args.input_size // args.patch_size
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    lfw_loader, cfpfp_loader, cplfw_loader, agedb30_loader, calfw_loader = FaceEmore(args)
    test_loaders = {"lfw": lfw_loader, "cfp_fp": cfpfp_loader, "cplfw": cplfw_loader, "agedb_30": agedb30_loader, "calfw": calfw_loader}
    for dataname, loader in test_loaders.items():
        images1, images2, labels = [], [], []
        for img1, img2, lab in tqdm(loader, desc=f"{dataname}", position=0, leave=True,  bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            img1 = rearrange(img1, 'b c (h p1) (w p2) -> (h w) b c p1 p2', h=H, w=W)
            img2 = rearrange(img2, 'b c (h p1) (w p2) -> (h w) b c p1 p2', h=H, w=W)
            img1 = torch.concatenate([img1[i].view(1, -1) for i in range(H*W)], dim=1)
            img2 = torch.concatenate([img2[i].view(1, -1) for i in range(H*W)], dim=1)
            img1 = img1.numpy()
            img2 = img2.numpy()
            lab = lab.flatten().numpy()
            images1.append(img1)
            images2.append(img2)
            labels.append(lab)
        images1 = np.vstack(images1)
        images2 = np.vstack(images2)
        labels = np.vstack(labels)
        k_fold = KFold(n_splits=10, shuffle=False)
        assert images1.shape[0] == images2.shape[0] == labels.shape[0]
        nrof_pairs = images1.shape[0]
        indices = np.arange(nrof_pairs)
        out_dir = os.path.join(args.output, f"{args.input_size}")
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        out_dir = os.path.join(out_dir, dataname)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        for fold_idx, (_, test_set) in enumerate(k_fold.split(indices)):
            np.savetxt(os.path.join(out_dir, f"fold{fold_idx}_images1.txt"), images1[test_set], fmt="%.6f", delimiter=",")
            np.savetxt(os.path.join(out_dir, f"fold{fold_idx}_images2.txt"), images2[test_set], fmt="%.6f", delimiter=",")
            np.savetxt(os.path.join(out_dir, f"fold{fold_idx}_labels.txt"), labels[test_set], fmt="%d", delimiter=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Write Face Emore to Txt Files')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('-o', '--output', metavar='DIR', help='path to txt files')
    parser.add_argument('--input-size', metavar='N', type=int, default=64, help='input image size (default:64)')
    parser.add_argument('--patch-size', metavar='N', default=32, type=int, help='patch size (default: 32)')
    parser.add_argument('--use-patch', action='store_true')
    parser.add_argument('--not-use-patch', dest='use_patch', action='store_false')
    parser.set_defaults(use_patch=False)
    args = parser.parse_args()
    if args.use_patch:
        image2txt_patch(args)
    else:
        image2txt(args)