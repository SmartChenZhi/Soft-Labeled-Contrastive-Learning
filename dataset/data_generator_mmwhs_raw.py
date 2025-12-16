import os.path
from pathlib import Path
import math

import numpy as np
from torch.utils import data
import re
import SimpleITK as sitk
import elasticdeform
import pandas as pd

import config
from utils.utils_ import tranfer_data_2_scratch, load_raw_data_mmwhs, load_mnmx_csv, assert_match
from dataset.data_generator_mscmrseg import ImageProcessor


class DataGenerator(data.Dataset):
    def __init__(self, phase="train", modality="ct", crop_size=224, n_samples=-1, augmentation=False,
                 data_dir='../data/mscmrseg/mmwhs/CT_MR_2D_Dataset_DA-master', bs=16, domain='s',
                 aug_mode='simple', aug_counter=False, normalization='minmax', fold=0, vert=False, split=0,
                 val_num=0, M3ASdata=True, zoom=1, percent=100):
        assert modality == "ct" or modality == "mr"
        self._modality = modality
        self._crop_size = crop_size
        self._phase = phase
        self._index = 0  # start from the 0th sample
        self._totalcount = 0
        self._augmentation = augmentation
        self._aug_mode = aug_mode
        self._aug_counter = aug_counter
        self._normalization = normalization
        self._image_files, self._mask_files, self._vert_files = [], [], []
        self._vert = vert
        self._zoom = zoom
        self._percent = percent
        # self._mnmx = load_mnmx_csv(modality, percent)

        num_dict = {'t': {'CT': np.setdiff1d(config.MMWHS_CT_S_TRAIN_SET, config.MMWHS_CT_T_VALID_SET),
                          'MR': np.setdiff1d(config.MMWHS_MR_S_TRAIN_SET, config.MMWHS_MR_T_VALID_SET if val_num == 0
                          else config.MMWHS_MR_T_VALID_SET1)},
                    's': {'CT': config.MMWHS_CT_S_TRAIN_SET, 'MR': config.MMWHS_MR_S_TRAIN_SET}}
        if M3ASdata:
            parent_fold = os.path.join(data_dir, f'{modality.upper()}_woGT')
            for num in num_dict[domain][modality.upper()]:
                self._image_files += [os.path.join(parent_fold, f'img{num}_slice{slc_num}.nii.gz') for slc_num in range(1, 17)]
                self._mask_files += [os.path.join(parent_fold, f'lab{num}_slice{slc_num}.nii.gz') for slc_num in range(1, 17)]
                if vert:
                    self._vert_files += [os.path.join(data_dir, f'vert{modality.upper()}/lab{num}_slice{slc_num}.npy') for slc_num in range(1, 17)]
        if domain == 't':
             train_extra = config.train_extra_list[split][fold]
        elif domain == 's':
            train_extra = config.train_extra_list[split][0] + config.train_extra_list[split][1]
        else:
            raise NotImplementedError
        print(f'domain {domain}, extra training samples: {np.sort(train_extra)}')
        train_extra = np.array(train_extra, dtype=int)
        if modality == 'ct':
            train_extra += 32
        parent_fold = os.path.join(data_dir, f'{modality.upper()}_withGT')
        for num in train_extra:
            self._image_files += [os.path.join(parent_fold, f'img{num}_slice{slc_num}.nii.gz') for slc_num in range(1, 17)]
            self._mask_files += [os.path.join(parent_fold, f'lab{num}_slice{slc_num}.nii.gz') for slc_num in range(1, 17)]
            if vert:
                self._vert_files += [os.path.join(data_dir, f'vert{modality.upper()}/lab{num}_slice{slc_num}.npy') for slc_num in range(1, 17)]
        assert len(self._image_files) == len(self._mask_files) and \
               len(self._image_files) > 0, f'data dir: {data_dir}, img file len: {len(self._image_files)}, ' \
                                           f'mask file len: {len(self._mask_files)}'
        self._len = len(self._image_files)
        print("{}: {}".format(modality, self._len))
        # self._shuffle_indices = np.arange(self._len)
        # self._shuffle_indices = np.random.permutation(self._shuffle_indices)
        if n_samples == -1:
            self._n_samples = self._len + self._len % bs
        else:
            self._n_samples = n_samples
        self._names = [Path(file).stem.split('.')[0] for file in self._image_files]

    def __len__(self):
        return self._n_samples

    @property
    def n_samples(self):
        return self._n_samples

    @n_samples.setter
    def n_samples(self, value):
        self._n_samples = value

    def __getitem__(self, index):
        i = index % self._len
        assert_match(self._image_files[i], self._mask_files[i])
        img, mask = load_raw_data_mmwhs(self._image_files[i], self._mask_files[i])
        # m = re.search('img\d+_', self._image_files[i])
        # img_name = m.group()[:-1]
        # vmin, vmax = self._mnmx.loc[img_name].min99, self._mnmx.loc[img_name].max99
        # img = np.clip((np.array(img, np.float32) - vmin) / (vmax - vmin), 0, 1)
        
        # Calculate min/max from the image itself
        img = np.array(img, np.float32)
        vmin = np.percentile(img, 0.5)
        vmax = np.percentile(img, 99.5)
        img = np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)

        aug_img, aug_mask = img, mask
        if self._augmentation:
            aug_mask = np.expand_dims(aug_mask, axis=-1)
            if self._aug_mode == 'simple':
                aug_img, aug_mask = ImageProcessor.simple_aug(image=aug_img, mask=aug_mask)
            else:
                aug_img, aug_mask = ImageProcessor.heavy_aug(image=aug_img, mask=aug_mask, vmax=1, aug_mode=self._aug_mode)
                if np.random.uniform(0, 1) < .5:
                    [aug_img, aug_mask] = elasticdeform.deform_random_grid([aug_img, aug_mask], axis=[(0, 1), (0, 1)],
                                                                           sigma=np.random.uniform(1, 7), order=0,
                                                                           mode='constant')
            aug_mask = aug_mask[..., 0]
        aug_img = np.stack([aug_img, aug_img, aug_img], axis=0)
        if self._normalization == 'zscore':
            # idx = np.where(aug_img != 0)
            # mean, std = aug_img[idx].mean(), aug_img[idx].std()
            mean, std = aug_img.mean(), aug_img.std()
            aug_img = (np.array(aug_img, np.float32) - mean) / std
        if self._vert:
            vertices = np.load(self._vert_files[i])
            return aug_img, aug_mask, vertices
        if self._aug_counter:
            if self._augmentation:
                mask = np.expand_dims(mask, axis=-1)
                if self._aug_mode == 'simple':
                    img, _ = ImageProcessor.simple_aug(image=img, mask=mask)
                else:
                    img, _ = ImageProcessor.heavy_aug(image=img, mask=mask, vmax=1, aug_mode=self._aug_mode)
            img = np.stack([img, img, img], axis=0)
            if self._normalization == 'zscore':
                mean, std = img.mean(), img.std()
                img = (np.array(img, np.float32) - mean) / std
            return aug_img, img, self._names[i]  # (3, 256, 256) (4, 256, 256)
        else:
            return aug_img, aug_mask, self._names[i]


def prepare_dataset(args, aug_counter=False, vert=False):
    scratch = tranfer_data_2_scratch(args.data_dir, args.scratch)
    content_dataset = DataGenerator(modality='mr' if args.rev else 'ct', crop_size=args.crop,
                                    augmentation=args.aug_s, data_dir=scratch, bs=args.bs,
                                    aug_mode=args.aug_mode, normalization=args.normalization,
                                    aug_counter=aug_counter if args.rev else False, fold=args.fold, domain='s',
                                    vert=vert, split=args.split, val_num=args.val_num, percent=args.percent)
    style_dataset = DataGenerator(modality='ct' if args.rev else 'mr', crop_size=args.crop,
                                  augmentation=args.aug_t, data_dir=scratch, bs=args.bs,
                                  aug_mode=args.aug_mode, normalization=args.normalization,
                                  aug_counter=False if args.rev else aug_counter, fold=args.fold, domain='t',
                                  vert=vert, split=args.split, val_num=args.val_num, M3ASdata=args.noM3AS, percent=args.percent)
    n_samples = int(
        math.ceil(max(content_dataset.n_samples, style_dataset.n_samples) / args.bs) * args.bs)
    content_dataset.n_samples = n_samples
    style_dataset.n_samples = n_samples
    content_loader = data.DataLoader(content_dataset, batch_size=args.bs, shuffle=True,
                                          num_workers=args.num_workers,
                                          pin_memory=args.pin_memory)
    print('content dataloader created.')
    style_loader = data.DataLoader(style_dataset, batch_size=args.bs, shuffle=True,
                                        num_workers=args.num_workers,
                                        pin_memory=args.pin_memory)
    print('style dataloader created.')
    return scratch, None, content_loader, style_loader


if __name__ == "__main__":

    def getcolormap():
        from matplotlib.colors import ListedColormap
        colorlist = np.round(
            np.array([[0, 0, 0], [186, 137, 120], [124, 121, 174], [240, 216, 152], [148, 184, 216]]) / 256, decimals=2)
        mycolormap = ListedColormap(colors=colorlist, name='mycolor', N=5)
        return mycolormap


    import matplotlib.pyplot as plt
    bssfp_generator = DataGenerator(phase='train', modality='bssfp', crop_size=224, n_samples=1000, augmentation=True,
                                    data_dir='F:/data/mscmrseg/origin')
    for img, msk in bssfp_generator:
        print(img.shape, msk.shape)
        print(img.min(), img.max())
        print(np.argmax(msk, axis=-3).min(), np.argmax(msk, axis=-3).max())
        f, plots = plt.subplots(1, 2)
        plots[0].axis('off')
        plots[1].axis('off')
        plots[0].imshow(img[1], cmap='gray')
        plots[1].imshow(np.argmax(msk, axis=0), cmap=getcolormap(), vmin=0, vmax=3)
        plt.show()
        pass
