import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
import SimpleITK as sitk
import cv2
from pathlib import Path
import pandas as pd
import os
from glob import glob

from utils.utils_ import crop_volume


def preprocess_volume(img_volume):
    """
    :param img_volume: A patient volume
    :return: applying CLAHE and Bilateral filter for contrast enhacnmeent and denoising
    """
    prepross_imgs = []
    for i in range(len(img_volume)):
        img = img_volume[i]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        cl1 = clahe.apply(img)
        prepross_imgs.append(cl1)
    return np.array(prepross_imgs)


def nii_to_png_mscmrseg(crop_size=224, preprocess=True, parent_folder='preprocess2.0', target_res=1., save_centroid=False):
    """
    preprocess the mscmrseg dataset. bssfp and t2 images are saved to '(train/test)A', the labels are saved to
    '(train/test)Amask'. The lge images and labels are saved to the corresponding folder replacing 'A' with 'B'.
    The images will be resized to have a pixel spacing of (1, 1) at the (x, y) dimension, while keep that at the z dim
    the same. The images will be histogram equalized with cv2.
    :param parent_folder:
    :param preprocess:
    :param crop_size: The size of the cropped image
    :param target_res:
    :return:
    """
    modalities = ['bSSFP', 't2', 'lge']
    datas = ['labels', 'dataset']  # ['dataset', 'labels']
    pat_id_start, pat_id_end = 1, 46
    modal_dict = {'bSSFP': 'C0', 't2': 'T2', 'lge': 'LGE'}
    # train_test = 'train'
    # st = 'A'
    # label = 'mask'
    csv_filename = f'label_centroid_mscmrseg_res{target_res}.csv'
    # if os.path.exists(csv_filename):
    #     df = pd.read_csv(csv_filename)
    # else:
    label_centroid = {'modality': [], 'pat_id': [], 'centroid_x': [], 'centroid_y': []}
    df = pd.DataFrame(label_centroid)
    for modality in modalities:
        st = 'A' if ((modality == 'bSSFP') or (modality == 't2')) else 'B'
        modal_fn = modal_dict[modality]
        for pat_id in range(pat_id_start, pat_id_end):
            centroid = None
            for data in datas:
                if data == 'labels':
                    centroid = None
                    label = 'mask'
                    manual_fn = '_manual'
                    order = 0
                else:
                    label = ''
                    manual_fn = ''
                    order = 3
                train_test = 'test' if pat_id < 6 else 'train'
                print(f"saving the {pat_id}st {modality} {data}")
                path = f"F:/data/mscmrseg/raw_data/{data}/patient{pat_id}_{modal_fn}{manual_fn}.nii.gz"
                print(f'read from {path}')
                vol = sitk.ReadImage(path)
                vol = sitk.Cast(sitk.RescaleIntensity(vol), sitk.sitkUInt8)
                spacing = vol.GetSpacing()
                vol = sitk.GetArrayFromImage(vol)
                vol = zoom(vol, (1, spacing[0] / target_res, spacing[1] / target_res), order=order, mode='nearest')
                if centroid is None:
                    coord = np.where(vol > 0)
                    centroid = np.array(coord[1:]).mean(axis=1).round().astype(np.int16)
                    if save_centroid:
                        df2 = pd.DataFrame([[modality, pat_id, centroid[0], centroid[1]]],
                                           columns=['modality', 'pat_id', 'centroid_x', 'centroid_y'])
                        df = df.append(df2, ignore_index=True)
                vol = crop_volume(vol, crop_size//2, centroid=centroid)
                if preprocess and data == 'dataset':
                    vol = preprocess_volume(vol)
                l = 0
                for m in vol:
                    save_path = f'F:/data/mscmrseg/{parent_folder}/{train_test}{st}{label}/pat_{pat_id}_{modality}_{l}.png'
                    if not Path(save_path).parent.exists():
                        Path(save_path).parent.mkdir(parents=True)
                        print(str(Path(save_path).parent) + ' created.')
                    cv2.imwrite(filename=save_path, img=m)
                    l += 1
    csv_path = f'F:/data/mscmrseg/{csv_filename}'
    if save_centroid and (not os.path.exists(csv_path)):
        df.to_csv(csv_path, index=False)
    print("finish")


def nii_to_png_mmwhs(percent=100):
    read = '/root/data/mmwhs/CT_MR_2D_Dataset_DA-master'
    save = '/root/data/mmwhs/CT_MR_2D_Dataset_DA_png'
    if percent != 100:
        save += f'/max{percent}'
    dir_dic = {'CT_withGT': {'start': 33, 'end': 52, 'save': 'CT_test'},
               'CT_woGT': {'start': 1, 'end': 32, 'save': 'CT_train'},
               'MR_withGT': {'start': 1, 'end': 20, 'save': 'MR_test'},
               'MR_woGT': {'start': 21, 'end': 46, 'save': 'MR_train'},
               }
    for key in dir_dic.keys():
        for i in range(dir_dic[key]['start'], dir_dic[key]['end'] + 1):
            paths = glob(f'{read}/{key}/img{i}_slice*nii.gz')
            img = []
            for path in paths:
                arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
                img.append(arr)
            img = np.array(img)
            if percent != 100:
                vmin, vmax = np.percentile(img, 1), np.percentile(img, 99)
            else:
                vmin, vmax = img.min(), img.max()
            for path in paths:
                arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
                arr = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
                arr = (arr * 255).astype(np.uint8)
                save_path = Path(f'{save}/{dir_dic[key]["save"]}/').joinpath(
                    Path(Path(path).stem).stem + '.png')
                if not save_path.parent.exists():
                    save_path.parent.mkdir(parents=True)
                cv2.imwrite(str(save_path), arr[..., 0])
                lab_path = Path(path).parent.joinpath(Path(path).name.replace('img', 'lab'))
                if lab_path.exists():
                    lab = sitk.GetArrayFromImage(sitk.ReadImage(str(lab_path)))
                    lab = (lab == 205) * 87 + (lab == 420) * 178 + (lab == 500) * 212 + (lab == 550) * 233 + (lab == 600) * 255
                    lab = lab.astype(np.uint8)
                    save_path = Path(f'{save}/{dir_dic[key]["save"]}/').joinpath(
                        Path(Path(lab_path).stem).stem + '.png')
                    cv2.imwrite(str(save_path), lab)
                print(f'{save_path} saved')


def crop_pad_mmwhs():
    for fold in ['CT_train', 'CT_test', 'MR_train', 'MR_test']:
        path = os.path.join('F:\data\mmwhs\CT_MR_2D_Dataset_DA_png/', fold)
        img_paths = glob(os.path.join(path, 'img*.png'))
        lab_paths = glob(os.path.join(path, 'lab*.png'))
        for lab_path in lab_paths:
            lab = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)
            lab = np.pad(lab, ((2, 2), (0, 0)))
            lab = lab[:, 8: -8]
            cv2.imwrite(lab_path, lab)
        for img_path in img_paths:
            img = cv2.imread(img_path)
            img = np.pad(img, ((2, 2), (0, 0), (0, 0)))
            img = img[:, 8: -8]
            cv2.imwrite(img_path, img)


def zoomin_for_DDFSeg(trgt_res=.5, crop_size=256):
    half_crop = crop_size // 2
    directory = 'D:\Work\ERC_project\Projects\data\mmwhs\CT_MR_2D_Dataset_DA_png'
    new_directory = 'D:\Work\ERC_project\Projects\data\mmwhs\DDFSeg'
    folds = ['CT_test', 'CT_train', 'MR_test', 'MR_train']
    sample_nums = [np.arange(33, 53), np.arange(1, 21), np.arange(1, 21), np.arange(21, 47)]
    for fold, sample_num in zip(folds, sample_nums):
        for sample in sample_num:
            img_paths = Path(directory).joinpath(fold).joinpath(f'img{sample}_slice*.png')
            for slice_num in range(1, len(glob(str(img_paths))) + 1):
                img_path = Path(directory).joinpath(fold).joinpath(f'img{sample}_slice{slice_num}.png')
                lab_path = Path(directory).joinpath(fold).joinpath(f'lab{sample}_slice{slice_num}.png')
                img, lab = cv2.imread(str(img_path)), cv2.imread(str(lab_path), cv2.IMREAD_GRAYSCALE)
                lab = (lab == 87) * 87 + (lab == 212) * 212 + (lab == 255) * 255
                img = zoom(img, (1 / trgt_res, 1 / trgt_res, 1), order=3, mode='nearest')
                lab = zoom(lab, (1 / trgt_res, 1 / trgt_res), order=0, mode='nearest')
                shape = lab.shape
                centerxy = np.array(np.where(lab > 0)).mean(axis=1).round().astype(int)
                top = max(centerxy[0] - half_crop, 0)
                left = max(centerxy[1] - half_crop, 0)
                if top >= shape[0] - crop_size:
                    top = shape[0] - crop_size
                if left >= shape[1] - crop_size:
                    left = shape[1] - crop_size
                img = img[top: top + crop_size, left: left + crop_size]
                lab = lab[top: top + crop_size, left: left + crop_size]

                new_img_path = Path(new_directory).joinpath(fold).joinpath(f'img{sample}_slice{slice_num}.png')
                new_lab_path = Path(new_directory).joinpath(fold).joinpath(f'lab{sample}_slice{slice_num}.png')
                cv2.imwrite(str(new_img_path), img[..., 0])
                cv2.imwrite(str(new_lab_path), lab)



if __name__ == '__main__':
    # for i in range(0, 8401):
    #     img = np.load(f'F:\data\mmwhs\mr_val\mask/mr_val_slice{i}.tfrecords.npy')
    #     # img = 255 * (img[..., 1] - img[..., 1].min()) / (img[..., 1].max() - img[..., 1].min())
    #     plt.imshow(img, cmap='gray')
    #     plt.show()
    """scale the image to have resolution 1 * 1, crop to 224 * 224. no CLAHE"""
    # nii_to_png_mscmrseg(crop_size=224, preprocess=False, parent_folder='origin', target_res=1)
    """preprocess procedure in Disentangle domain features for cross-modality cardiac image segmentation"""
    # nii_to_png_mscmrseg(crop_size=256, preprocess=False, parent_folder='origin.5', target_res=.6, save_centroid=True)
    # nii_to_png_mmwhs()
    # crop_pad_mmwhs()
    # zoomin_for_DDFSeg()
    nii_to_png_mmwhs(99)
