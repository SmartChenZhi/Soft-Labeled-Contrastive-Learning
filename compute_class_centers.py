import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from model.segmentation_models import segmentation_models
from utils.utils_ import get_device, update_class_center_iter, cal_centroid

def build_args(ns):
    class A: pass
    a = A()
    a.data_dir = ns.data_dir
    a.scratch = False
    a.crop = ns.crop
    a.aug_s = False
    a.aug_t = False
    a.aug_mode = 'simple'
    a.normalization = 'minmax'
    a.pin_memory = False
    a.num_workers = ns.num_workers
    a.clahe = False
    a.rev = False
    a.split = ns.split
    a.fold = ns.fold
    a.val_num = 0
    a.percent = 100
    a.noM3AS = True
    a.raw_data_dir = None
    a.bs = ns.batch_size
    return a

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-data_dir', type=str, required=True)
    p.add_argument('-raw', action='store_true')
    p.add_argument('-backbone', type=str, default='resnet50')
    p.add_argument('-fold', type=int, default=0)
    p.add_argument('-split', type=int, default=0)
    p.add_argument('-batch_size', type=int, default=8)
    p.add_argument('-num_workers', type=int, default=0)
    p.add_argument('-crop', type=int, default=224)
    p.add_argument('-domain', type=str, choices=['s','t'], required=True)
    p.add_argument('-method', type=str, choices=['hard_s','soft_t'], required=True)
    p.add_argument('-momentum', type=float, default=0.95)
    p.add_argument('-threshold', type=float, default=-2.0)
    p.add_argument('-low_thd', type=float, default=0.6)
    p.add_argument('-high_thd', type=float, default=0.99)
    p.add_argument('-weighted_ave', action='store_true')
    p.add_argument('-max_batches', type=int, default=-1)
    args = p.parse_args()

    device = get_device()
    if args.raw:
        from dataset.data_generator_mmwhs_raw import prepare_dataset as prepare_dataset_mmwhs_raw
        dl_args = build_args(args)
        _, _, content_loader, style_loader = prepare_dataset_mmwhs_raw(dl_args, aug_counter=False, vert=False)
    else:
        from dataset.data_generator_mmwhs import prepare_dataset as prepare_dataset_mmwhs
        dl_args = build_args(args)
        _, _, content_loader, style_loader = prepare_dataset_mmwhs(dl_args, aug_counter=False, vert=False)

    seg = segmentation_models(name=args.backbone, pretrained=False,
                              decoder_channels=(512,256,128,64,32), in_channel=3,
                              classes=4, multilvl=True).to(device)
    seg.eval()

    feat_ch = seg.classifier.in_channels
    centers = torch.zeros((4, feat_ch), device=device)

    if args.domain == 's':
        loader = content_loader
    else:
        loader = style_loader

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            img, lbl, _ = batch
            img = img.to(device)
            if args.method == 'hard_s':
                pred, pred_aux, dcdr_ft = seg(img)
                lbl = lbl.to(device)
                centers = update_class_center_iter(dcdr_ft, lbl, centers, m=args.momentum, num_class=4)
            else:
                pred, pred_aux, dcdr_ft = seg(img)
                prob = F.softmax(pred, dim=1)
                c, _, _ = cal_centroid(decoder_ft=dcdr_ft, label=prob, previous_centroid=None,
                                       momentum=args.momentum, pseudo_label=True, n_class=4, partition=1,
                                       threshold=args.threshold, thd_w=0.0, weighted_ave=args.weighted_ave,
                                       epoch=bi, max_epoch=max(1, args.max_batches if args.max_batches>0 else len(loader)),
                                       low_thd=args.low_thd, high_thd=args.high_thd, stdmin=False)
                if isinstance(c, list):
                    c = c[0]
                centers = args.momentum * centers + (1 - args.momentum) * c
            if args.max_batches > 0 and bi+1 >= args.max_batches:
                break

    out_name = f"class_center_{'ct' if args.domain=='s' else 'mr'}_f{args.fold}.npy"
    np.save(out_name, centers.detach().cpu().numpy())
    print(out_name)

if __name__ == '__main__':
    main()