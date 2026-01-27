from .BayeSeg import build as build_BayeSeg
from .Unet import build as build_UNet
from .udaBayeSeg import build as build_udaBayeSeg

def build_model(args):
    if args.model == "BayeSeg":
        return build_BayeSeg(args)
    elif args.model == "UNet":
        return build_UNet(args)
    elif args.model == "udaBayeSeg":
        return build_udaBayeSeg(args)
    else:
        raise ValueError("invalid model:{}".format(args.model))
