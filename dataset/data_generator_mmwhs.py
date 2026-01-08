import os.path
import re
from pathlib import Path
import math

import numpy as np
import pandas as pd
from torch.utils import data
import cv2
from glob import glob
import elasticdeform

import config
from utils.utils_ import tranfer_data_2_scratch, assert_match
from dataset.data_generator_mscmrseg import ImageProcessor


class DataGenerator(data.Dataset):
    def __init__(self, phase="train", modality="ct", crop_size=224, n_samples=-1, augmentation=False,
                 data_dir='../data/mmwhs/CT_MR_2D_Dataset_DA_png', bs=16, domain='s',
                 aug_mode='simple', aug_counter=False, clahe=False, normalization='minmax', fold=0, vert=False, split=0,
                 val_num=0, M3ASdata=True, zoom=1):
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
        self._ifclahe = clahe
        self._vert = vert
        self._zoom = zoom
        if clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        num_dict = {'t': {'CT': np.setdiff1d(config.MMWHS_CT_S_TRAIN_SET, config.MMWHS_CT_T_VALID_SET),
                          'MR': np.setdiff1d(config.MMWHS_MR_S_TRAIN_SET, config.MMWHS_MR_T_VALID_SET if val_num == 0
                          else config.MMWHS_MR_T_VALID_SET1)},
                    's': {'CT': config.MMWHS_CT_S_TRAIN_SET, 'MR': config.MMWHS_MR_S_TRAIN_SET}}
        if M3ASdata:
            parent_fold = os.path.join(data_dir, f'{modality.upper()}_train')
            for num in num_dict[domain][modality.upper()]:
                self._image_files += [os.path.join(parent_fold, f'img{num}_slice{slc_num}.png') for slc_num in range(1, 17)]
                self._mask_files += [os.path.join(parent_fold, f'lab{num}_slice{slc_num}.png') for slc_num in range(1, 17)]
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
        parent_fold = os.path.join(data_dir, f'{modality.upper()}_test')
        for num in train_extra:
            self._image_files += [os.path.join(parent_fold, f'img{num}_slice{slc_num}.png') for slc_num in range(1, 17)]
            self._mask_files += [os.path.join(parent_fold, f'lab{num}_slice{slc_num}.png') for slc_num in range(1, 17)]
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

    def get_images_masks(self, img_path, mask_path):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {os.path.abspath(img_path)}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {os.path.abspath(mask_path)}")

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Failed to read image (corrupted or format not supported): {img_path}")
        if mask is None:
            raise ValueError(f"Failed to read mask (corrupted or format not supported): {mask_path}")

        mask = (mask == 87) * 1 + (mask == 212) * 2 + (mask == 255) * 3
        mask = np.array(mask, dtype=np.uint8)
        return img, mask

    def __getitem__(self, index):
        i = index % self._len
        assert_match(self._image_files[i], self._mask_files[i])
        img, mask = self.get_images_masks(img_path=self._image_files[i], mask_path=self._mask_files[i])
        if self._ifclahe:
            img = self.clahe.apply(img[..., 0])
            img = np.stack([img, img, img], axis=-1)
        aug_img, aug_mask = img, mask
        if self._augmentation:
            aug_mask = np.expand_dims(aug_mask, axis=-1)
            if self._aug_mode == 'simple':
                aug_img, aug_mask = ImageProcessor.simple_aug(image=aug_img, mask=aug_mask)
            else:
                aug_img, aug_mask = ImageProcessor.heavy_aug(image=aug_img, mask=aug_mask, aug_mode=self._aug_mode)
                if np.random.uniform(0, 1) < .5:
                    [aug_img, aug_mask] = elasticdeform.deform_random_grid([aug_img, aug_mask], axis=[(0, 1), (0, 1)],
                                                                           sigma=np.random.uniform(1, 7), order=0,
                                                                           mode='constant')
            aug_mask = aug_mask[..., 0]
        if self._crop_size and ((aug_img.shape[1] != self._crop_size) or (aug_img.shape[0] != self._crop_size)):
            aug_img = ImageProcessor.crop_volume(aug_img, crop_size=self._crop_size // 2)
            aug_mask = ImageProcessor.crop_volume(np.array(aug_mask), crop_size=self._crop_size // 2)
        aug_img = np.moveaxis(aug_img, -1, -3)
        if self._normalization == 'minmax':
            aug_img = np.array(aug_img, np.float32) / 255.
        elif self._normalization == 'zscore':
            # idx = np.where(aug_img != 0)
            # mean, std = aug_img[idx].mean(), aug_img[idx].std()
            mean, std = aug_img.mean(), aug_img.std()
            aug_img = (np.array(aug_img, np.float32) - mean) / std
        else:
            raise NotImplementedError(f'self._normalization = {self._normalization} is not defined for the data loader.')
        if self._vert:
            vertices = np.load(self._vert_files[i])
            return aug_img, aug_mask, vertices
        if self._aug_counter:
            if self._augmentation:
                mask = np.expand_dims(mask, axis=-1)
                if self._aug_mode == 'simple':
                    img, _ = ImageProcessor.simple_aug(image=img, mask=mask)
                else:
                    img, _ = ImageProcessor.heavy_aug(image=img, mask=mask, aug_mode=self._aug_mode)
                # mask = mask[..., 0]
            if self._crop_size and img.shape[1] != self._crop_size:
                img = ImageProcessor.crop_volume(img, crop_size=self._crop_size // 2)
            img = np.moveaxis(img, -1, -3)
            # mask = to_categorical(np.array(mask), num_classes=4)
            if self._normalization == 'minmax':
                img = np.array(img, np.float32) / 255.
            elif self._normalization == 'zscore':
                mean, std = img.mean(), img.std()
                img = (np.array(img, np.float32) - mean) / std
            else:
                raise NotImplementedError
            return aug_img, img, self._names[i]  # (3, 256, 256) (4, 256, 256)
        else:
            return aug_img, aug_mask, self._names[i]


def prepare_dataset(args, aug_counter=False, vert=False):
    scratch = tranfer_data_2_scratch(args.data_dir, args.scratch)
    scratch_raw = tranfer_data_2_scratch(args.raw_data_dir, args.scratch)
    content_dataset = DataGenerator(modality='mr' if args.rev else 'ct', crop_size=args.crop,
                                    augmentation=args.aug_s, data_dir=scratch, bs=args.bs,
                                    aug_mode=args.aug_mode, normalization=args.normalization, clahe=args.clahe,
                                    aug_counter=False if args.rev else aug_counter, fold=args.fold, domain='s',
                                    vert=vert, split=args.split, val_num=args.val_num)
    style_dataset = DataGenerator(modality='ct' if args.rev else 'mr', crop_size=args.crop,
                                  augmentation=args.aug_t, data_dir=scratch, bs=args.bs,
                                  aug_mode=args.aug_mode, normalization=args.normalization, clahe=args.clahe,
                                  aug_counter=aug_counter if args.rev else False, fold=args.fold, domain='t',
                                  vert=vert, split=args.split, val_num=args.val_num, M3ASdata=args.noM3AS)
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
    return scratch, scratch_raw, content_loader, style_loader


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