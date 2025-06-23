import os
import wandb
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
from models import build_model, ArcFace
from datasets import WebFace4M, FaceEmore
from argument import CLI
from time import gmtime, strftime
from helper import validate_emore, count_parameters
from tqdm import tqdm
import torch.nn.functional as F

class FaceModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

def main(args):    
    if int(torch.__version__[0]) > 1:
        torch.set_float32_matmul_precision('medium')
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    device = torch.device(f"cuda:{args.gpu}") if args.gpu is not None else torch.device("cpu")

    # evaluate
    if args.evaluate:
        lfw_loader, cfpfp_loader, cplfw_loader, agedb30_loader, calfw_loader = FaceEmore(args)
        test_loaders = {"lfw": lfw_loader, "cfp_fp": cfpfp_loader, "cplfw": cplfw_loader, "agedb_30": agedb30_loader, "calfw": calfw_loader}
        backbone = build_model(args)
        backbone.to(device)
        ckpt = torch.load(args.ckpt_path, map_location=device)
        backbone.load_state_dict(ckpt["backbone"])
        test_acc, test_metric = validate_emore(backbone, test_loaders, device)
        test_metric = sorted(test_metric.items())
        for k, v in test_metric:
            print("{:}: {:.4f}".format(k, v))
        print("Acc: {:.4f}".format(test_acc))
        return

    
    # model
    backbone = build_model(args)
    head = ArcFace(256, args.num_classes)
    model = FaceModel(backbone, head)
    model.to(device)
    backbone_params = count_parameters(backbone.nets) + count_parameters(backbone.linear)
    head_params = count_parameters(head)
    print(f"Backbone: {backbone_params/10**6}M params")
    print(f"Head: {head_params/10**6}M params")

    # dataset
    train_loader = WebFace4M(args)
    lfw_loader, cfpfp_loader, cplfw_loader, agedb30_loader, calfw_loader = FaceEmore(args)
    test_loaders = {"lfw": lfw_loader, "cfp_fp": cfpfp_loader, "cplfw": cplfw_loader, "agedb_30": agedb30_loader, "calfw": calfw_loader}

    # optimizer
    optimizer = optim.SGD([{'params': [model.head.kernel], 'weight_decay': args.weight_decay},
                            {'params': model.backbone.parameters()}], lr=args.lr, momentum=args.momentum)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)
    criterion = nn.CrossEntropyLoss()

    # log
    prefix = f"{args.arch}-{args.input_size}"
    if args.resume:
        args.output_dir = os.path.split(args.ckpt_path)[0]
    else:
        current_time = strftime("%m-%d_0", gmtime())
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        args.output_dir = os.path.join(args.output_dir, f"{prefix}_{current_time}")
        if os.path.isdir(args.output_dir):
            while True:
                cur_exp_number = int(args.output_dir[-2:].replace('_', ""))
                args.output_dir = args.output_dir[:-2] + "_{}".format(cur_exp_number + 1)
                if not os.path.isdir(args.output_dir):
                    break
        os.mkdir(args.output_dir)    
    if args.use_wandb:
        wandb.init(project=args.wandb_proj, config=vars(args), name=f"{prefix}", dir=args.output_dir, resume=args.resume)

    # resume
    test_acc = None
    start_epoch = 1
    jigsaw_weight = args.jigsaw_weight
    if args.resume:
        ckpt = torch.load(args.ckpt_path, map_location=device)
        backbone.load_state_dict(ckpt["backbone"])
        head.load_state_dict(ckpt["head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        test_acc = ckpt["test_acc"]
        jigsaw_weight = ckpt["jigsaw_weight"]
        print(f"Resume from: {args.ckpt_path}")
    
    # main loop
    for epoch in range(start_epoch, args.epochs + 1):
        # training 
        model.train()
        with tqdm(train_loader, unit="batch", position=0, leave=True,  bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as tepoch:
            for image, label in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                image, label = image.to(device), label.to(device)
                optimizer.zero_grad()

                embedding, pred, truth, = backbone.forward(image)
                cos_thetas = head(embedding, label)
                loss = criterion(cos_thetas, label)
                loss_jigsaw = F.cross_entropy(pred, truth)
                loss += jigsaw_weight * loss_jigsaw
                
                loss.backward()
                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
                optimizer.step()
    
                current_lr = optimizer.param_groups[0]['lr']

                tepoch.set_postfix(loss=loss.item(), acc=test_acc, lr = current_lr)
                if args.use_wandb:
                    wandb.log({"loss": loss.item(), "lr": current_lr})
                
            scheduler.step()

        # testing
        test_acc, test_metric = validate_emore(backbone, test_loaders, device)
        if args.use_wandb:
                test_metric.update({"epoch": epoch})
                wandb.log(test_metric)

        # checkpoint
        torch.save({"backbone": backbone.state_dict(), 
                    "head": head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "test_acc": test_acc,
                    "jigsaw_weight": jigsaw_weight},
                    args.output_dir+"/model.ckpt")
    
    # log
    print("Acc: {:.4f}".format(test_acc))
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = CLI()
    main(args)
