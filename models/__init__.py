from .pcnn import PatchCNN
from .head import ArcFace

def build_model(args):
    return PatchCNN(input_size=args.input_size, patch_size=args.patch_size)