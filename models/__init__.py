from .BayeSeg import build as build_BayeSeg
from .vqBayeSeg import build as build_vqBayeSeg
from .vqUnet import build as build_vqUNet
from .Unet import build as build_UNet
from .VqVae import build as build_vqvae
from .udaBayeSeg import build as build_udaBayeSeg

def build_model(args):
    if args.model == "BayeSeg":
        return build_BayeSeg(args)
    if args.model == "vqBayeSeg":
        return build_vqBayeSeg(args)
    elif args.model == "vqUNet":
        return build_vqUNet(args)
    elif args.model == "UNet":
        return build_UNet(args)
    elif args.model == "vqvae":
        return build_vqvae(args)
    elif args.model == "udaBayeSeg":
        return build_udaBayeSeg(args)
    else:
        raise ValueError("invalid model:{}".format(args.model))
