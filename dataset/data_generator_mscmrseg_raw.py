import os.path
from pathlib import Path
import math

import numpy as np
from torch.utils import data
import SimpleITK as sitk
import elasticdeform

import config
from utils.utils_ import tranfer_data_2_scratch, assert_match
from dataset.data_generator_mscmrseg import ImageProcessor


def load_raw_data_mscmrseg(img_path, mask_path=None):
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))  # (H, W) or (H, W, C)
    if img.ndim == 3:
        img = img[..., 0]
    img = np.array(img, np.float32)
    if mask_path is not None:
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        # MSCMRseg label encoding: map original label values to 0-3
        mask = (mask == 85) * 1 + (mask == 212) * 2 + (mask == 255) * 3
        mask = np.array(mask, dtype=np.uint8)
    else:
        mask = None
    return img, mask


class DataGenerator(data.Dataset):
    def __init__(self, phase="train", modality="bssfp", crop_size=224, n_samples=-1, augmentation=False, clahe=False,
                 data_dir='../data/mscmrseg/origin', pat_id=-1, slc_id=-1, bs=16, aug_mode='simple', aug_counter=False,
                 normalization='minmax', fold=0, domain='s', vert=False):
        assert modality == "bssfp" or modality == "t2" or modality == 'lge'
        self._modality = modality
        self._crop_size = crop_size
        self._phase = phase
        self._index = 0
        self._totalcount = 0
        self._augmentation = augmentation
        self._aug_mode = aug_mode
        self._aug_counter = aug_counter
        self._normalization = normalization
        self._vert = vert
        self._ifclahe = clahe
        if modality == 'bssfp':
            pat_name = 'bSSFP'
        else:
            pat_name = modality
        st = 'A' if (modality == 'bssfp' or modality == 't2') else 'B'
        self._image_files, self._mask_files, self._vert_files = [], [], []
        if domain == 't':
            pat_ids = config.MSCMRSEG_TEST_FOLD1 if fold == 0 else config.MSCMRSEG_TEST_FOLD2
        elif domain == 's':
            pat_ids = config.MSCMRSEG_TEST_FOLD1 + config.MSCMRSEG_TEST_FOLD2
        else:
            raise NotImplementedError
        for pat_id in pat_ids:
            self._image_files += sorted(Path(os.path.join(data_dir, f'train{st}')).glob(
                f'pat_{pat_id}_{pat_name}_*.nii.gz'))
            self._mask_files += sorted(Path(os.path.join(data_dir, f'train{st}mask')).glob(
                f'pat_{pat_id}_{pat_name}_*.nii.gz'))
            if vert:
                self._vert_files += sorted(Path(os.path.join(data_dir, f'vert{st}')).glob(
                    f'pat_{pat_id}_{pat_name}_*.npy'))
        self._image_files = [str(f) for f in self._image_files]
        self._mask_files = [str(f) for f in self._mask_files]
        self._vert_files = [str(f) for f in self._vert_files]
        assert len(self._image_files) == len(self._mask_files) and \
               len(self._image_files) > 0, f'data dir: {data_dir}, img file len: {len(self._image_files)}, ' \
                                           f'mask file len: {len(self._mask_files)}'
        self._len = len(self._image_files)
        print("{}: {}".format(modality, self._len))
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
        img, mask = load_raw_data_mscmrseg(self._image_files[i], self._mask_files[i])

        # Normalize using percentile clipping (raw nii values, not 0-255)
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
        if self._crop_size and aug_img.shape[1] != self._crop_size:
            aug_img = ImageProcessor.crop_volume(aug_img, crop_size=self._crop_size // 2)
            aug_mask = ImageProcessor.crop_volume(np.array(aug_mask), crop_size=self._crop_size // 2)
        if self._normalization == 'zscore':
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
            if self._crop_size and img.shape[1] != self._crop_size:
                img = ImageProcessor.crop_volume(img, crop_size=self._crop_size // 2)
            if self._normalization == 'zscore':
                mean, std = img.mean(), img.std()
                img = (np.array(img, np.float32) - mean) / std
            return aug_img, img, self._names[i]
        else:
            return aug_img, aug_mask, self._names[i]


def prepare_dataset(args, aug_counter=False, vert=False):
    scratch = tranfer_data_2_scratch(args.data_dir, args.scratch)
    content_dataset = DataGenerator(modality='lge' if args.rev else 'bssfp', crop_size=args.crop,
                                    augmentation=args.aug_s, data_dir=scratch, bs=args.bs, clahe=args.clahe,
                                    aug_mode=args.aug_mode, normalization=args.normalization, fold=args.fold,
                                    aug_counter=aug_counter if args.rev else False, domain='s', vert=vert)
    style_dataset = DataGenerator(modality='bssfp' if args.rev else 'lge', crop_size=args.crop,
                                  augmentation=args.aug_t, data_dir=scratch, bs=args.bs, clahe=args.clahe,
                                  aug_mode=args.aug_mode, normalization=args.normalization, fold=args.fold,
                                  aug_counter=False if args.rev else aug_counter, domain='t', vert=vert)
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


def init_test_dataset(args, scratch=None):
    style_dataset = DataGenerator(modality='bssfp' if args.rev else 'lge', crop_size=args.crop,
                                  augmentation=False, data_dir=scratch, bs=args.bs, clahe=args.clahe,
                                  aug_mode=args.aug_mode, normalization=args.normalization, fold=args.fold,
                                  aug_counter=False, domain='t', vert=False)
    style_loader = data.DataLoader(style_dataset, batch_size=args.bs, shuffle=False,
                                   num_workers=args.num_workers,
                                   pin_memory=args.pin_memory)
    return style_loader
