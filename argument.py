import os
import yaml
import argparse
from pathlib import Path
from argparse import Namespace

def CLI():
    parser = argparse.ArgumentParser(description='Train Face Recognition Model')
    parser.add_argument('config', metavar='DIR', type=str, default=None, help='path to config yaml file')
    parser.add_argument('-a', '--arch', metavar='ARCH', type=str, default=None)
    parser.add_argument('--train-data-dir', metavar='PATH', default=None, type=str, help='path to training data')
    parser.add_argument('--val-data-dir', metavar='PATH', default=None, type=str, help='path to validation data')
    parser.add_argument('--test-data-dir', metavar='PATH', default=None, type=str, help='path to test data')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--output-dir', metavar='PATH', default=None, type=str, help='path to output')
    parser.add_argument('--input-size', metavar='N', type=int, default=None, help='input image size')
    parser.add_argument('--seed', metavar='N', type=int, default=0, help='random seed (default:0)')
    parser.add_argument('--batch-size', metavar='N', default=None, type=int, help='batch size')
    parser.add_argument('--patch-size', metavar='N', default=None, type=int, help='patch size')
    parser.add_argument('--epochs', default=None, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--num-workers', default=None, type=int, metavar='N', help='number of workers')
    parser.add_argument('--lr', help='learning rate', default=None, type=float)
    parser.add_argument('--jigsaw-weight', help='weight of jigsaw', default=None, type=float)
    parser.add_argument('--lr-milestones', default=None, nargs='+', help='epochs for reducing LR')
    parser.add_argument('--lr-gamma', default=None, type=float, help='multiply when reducing LR')
    parser.add_argument('--momentum', default=None, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=None, type=float, help='weight decay')
    parser.add_argument('--ckpt-path', type=str, metavar='PATH', default=None, help='path to ckpt')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--not-resume', dest='resume', action='store_false')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--not-evaluate', dest='evaluate', action='store_false')
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument('--not-pin-memory', dest='pin_memory', action='store_false')
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--not-use-wandb', dest='use_wandb', action='store_false')
    parser.add_argument('--wandb-proj', type=str, metavar='NAME', default=None, help='wandb project name')
    parser.add_argument('--note', metavar='NOTE', type=str, default=None)
    parser.add_argument('--grad-clip', metavar='F', type=float, default=None, help='gradient clip value (default: None)')
    parser.set_defaults(resume=False, evaluate=False, pin_memory=True, use_wandb=False)
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Not found {args.config}")
    config = yaml.safe_load(Path(args.config).read_text())
    args = vars(args)
    for k in args.keys():
        if args[k] is not None:
            config[k] = args[k]
    args = Namespace(**config)

    if args.resume and not os.path.isfile(args.ckpt_path):
        raise FileNotFoundError(f"Not found {args.ckpt_path}")
  
    args.num_classes, args.num_images = 205990, 4235242
    print(f'Check training dataset (WebFace4M): {args.num_images} images with {args.num_classes} classes')
    # WebFace4M: 205990 IDs and 4235242 Imgs 
    for data_name in ["lfw", "cplfw", "agedb_30", "calfw", "cfp_fp"]:
        if not os.path.isfile(os.path.join(args.test_data_dir, f'{data_name}.bin')):
            raise FileNotFoundError(f'Not found: {data_name} in {args.test_data_dir}')

    return args