from datetime import datetime
import os

import cv2
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import zoom
import SimpleITK as sitk
import torch
from pathlib import Path

from colorama import Fore, Style

import config
from utils.utils_ import (read_img, keep_largest_connected_components, crop_volume, get_device, load_raw_data_mmwhs,
                          load_mnmx_csv, easy_dic, check_mkdir_parent_dir, name_the_model, check_del)
from utils.timer import timeit
from metric import metrics
from dataset.data_generator_mscmrseg import ImageProcessor


def save_results_to_lists(res, endo_dc, rv_dc, myo_dc, endo_hd, rv_hd, myo_hd, endo_asd, rv_asd, myo_asd):
    endo_dc.append(res['lv'][0])
    rv_dc.append(res['rv'][0])
    myo_dc.append(res['myo'][0])
    if res['lv'][1] != -1:
        endo_hd.append(res['lv'][1])
    if res['rv'][1] != -1:
        rv_hd.append(res['rv'][1])
    if res['myo'][1] != -1:
        myo_hd.append(res['myo'][1])
    if res['lv'][2] != -1:
        endo_asd.append(res['lv'][2])
    if res['rv'][2] != -1:
        rv_asd.append(res['rv'][2])
    if res['myo'][2] != -1:
        myo_asd.append(res['myo'][2])
    return endo_dc, rv_dc, myo_dc, endo_hd, rv_hd, myo_hd, endo_asd, rv_asd, myo_asd


class Evaluator:
    """
        Evaluate the performance of a segmentation model with the raw data of bSSFP and LGE
    """

    def __init__(self, data_dir='../data/mscmrseg/origin', raw_data_dir='../data/mscmrseg/raw_data',
                 normalization='minmax', clahe=False, raw=False,
                 class_name=('myo', 'lv', 'rv'), colored=False, dataset='mscmrseg'):
        """
        :param data_dir:
        :param class_name:
        :param colored: Whether to color some texts to make the messages more readable
        """
        self.class_name = class_name
        self._data_dir = data_dir
        self._raw_data_dir = raw_data_dir
        print(f'data_dir: {data_dir}, raw_data_dir: {raw_data_dir}')
        self._color_preffixR, self._color_preffixG, self._color_preffixB = '', '', ''
        self._color_suffix = ''
        self._dataset = dataset
        if colored:
            self._color_preffixR, self._color_preffixG, self._color_preffixB = Fore.RED, Fore.GREEN, Fore.BLUE
            self._color_suffix = Style.RESET_ALL
        self._normalization = normalization
        # if normalization == 'zscore':
        #     self._df_stat = pd.read_csv(Path(self._data_dir).joinpath('mscmrseg_uint8_mean_std.csv'))
        self._ifclahe=clahe
        if self._ifclahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self._raw = raw

    def evaluate_single_dataset(self, seg_model, model_name='best_model', modality='lge', phase='test', ifhd=True,
                                ifasd=True, save_csv=False, save_hd=False, weight_dir=None, bs=32, toprint=True,
                                lge_train_test_split=None, cal_unctnty=False, watch_pat=None, klc=True,
                                ifhd95=True, pred_index=0, fold_num=0, split=0, val_num=0, crop_size=224,
                                spacing=1, percent=100, save_pred=False, volume=False, save_norm=False, verbose=False):
        if self._dataset == 'mscmrseg':
            measures = self.evaluate_single_dataset_mscmrseg(seg_model, model_name=model_name, modality=modality,
                                                             phase=phase, ifhd=ifhd, ifasd=ifasd,
                                                             weight_dir=weight_dir, bs=bs, toprint=toprint,
                                                             lge_train_test_split=lge_train_test_split,
                                                             cal_unctnty=cal_unctnty, watch_pat=watch_pat, klc=klc,
                                                             ifhd95=ifhd95, crop_size=crop_size, pred_index=pred_index,
                                                             fold_num=fold_num, save_pred=save_pred, save_norm=save_norm,
                                                             verbose=verbose)
        elif self._dataset == 'mmwhs':
            # modality == 'mr' for CTMR dataset
            measures = self.evaluate_single_dataset_mmwhs(seg_model, model_name=model_name, modality=modality,
                                                          phase=phase, ifhd=ifhd, spacing=spacing,
                                                          ifasd=ifasd, save_csv=save_csv, save_hd=save_hd,
                                                          weight_dir=weight_dir, bs=bs, toprint=toprint,
                                                          cal_unctnty=cal_unctnty, watch_pat=watch_pat, klc=klc,
                                                          ifhd95=ifhd95, crop_size=crop_size, pred_index=pred_index,
                                                          fold_num=fold_num, split=split, val_num=val_num, percent=percent,
                                                          save_pred=save_pred, volume=volume, verbose=verbose)
        else:
            print(self._dataset)
            raise NotImplementedError
        return measures

    def get_things_ready(self, seg_model, weight_dir):
        colorlist = np.round(
            np.array([[0, 0, 0], [186, 137, 120], [240, 216, 152], [148, 184, 216]]) / 256,
            decimals=2)
        mycolormap = ListedColormap(colors=colorlist, name='mycolor', N=4)
        device = get_device()
        seg_model.eval()
        if weight_dir is not None:
            print(f'model weights dir: {Path(weight_dir).absolute()}')
            try:
                seg_model.load_state_dict(torch.load(weight_dir)['model_state_dict'])
            except:
                seg_model.load_state_dict(torch.load(weight_dir))
            print("model loaded from {}{}{}".format(self._color_preffixB, weight_dir, self._color_suffix))
        seg_model = seg_model.to(device)
        return device, mycolormap, seg_model

    def predict_single_pat_mscmrseg(self, seg_model, device, modality='lge', bs=32, klc=True, crop_size=224,
                                    pred_index=0, pat_id=7, pred_ft=False):
        with torch.no_grad():
            if modality == 'lge':
                folder = 'LGE'
            elif modality == 'bssfp':
                folder = 'C0'
            else:
                raise ValueError('modality can only be \'bssfp\' or \'lge\'')

            mask_path = os.path.join(self._raw_data_dir, 'labels/patient{}_{}_manual.nii.gz'.format(pat_id, folder))
            nimg = sitk.ReadImage(mask_path)
            spacing = nimg.GetSpacing()
            nimg = sitk.GetArrayFromImage(nimg)
            vol = read_img(pat_id, nimg.shape[0], modality=modality, file_path=self._data_dir)
            if self._ifclahe:
                vol = [self.clahe.apply(tmp[..., 0]) for tmp in vol]
                vol = np.stack([vol, vol, vol], axis=-1)
            if vol.shape[1] != crop_size or vol.shape[2] != crop_size:
                vol = crop_volume(vol, crop_size=crop_size // 2)
            if self._normalization == 'minmax':
                x_batch = np.array(vol, np.float32) / 255.
            elif self._normalization == 'zscore':
                x_batch = (np.array(vol, np.float32) - vol.mean(axis=(1, 2, 3), keepdims=True)) / vol.std(
                    axis=(1, 2, 3), keepdims=True)
                x_batch = x_batch.astype(np.float32)
            else:
                raise NotImplementedError
            x_batch = np.moveaxis(x_batch, -1, 1)
            pred = []
            ft = []  # store the decoder features. Will be used to calculate the feature norm.
            for i in range(0, len(x_batch), bs):
                index = np.arange(i, min(i + bs, len(x_batch)))
                imgs = x_batch[index]
                pred_temp = seg_model(torch.tensor(imgs).to(device))
                pred1 = pred_temp[pred_index] if type(pred_temp) == tuple else pred_temp
                if pred_ft and type(pred_temp) == tuple:
                    ft.append(pred_temp[-1].detach().cpu().numpy())
                pred.append(torch.softmax(pred1, dim=1).cpu().detach().numpy())
            pred = np.concatenate(pred, axis=0)
            pred = np.argmax(pred, axis=1)
            if klc:
                pred = keep_largest_connected_components(pred)
            if pred_ft:
                return pred, nimg, spacing, ft
            else:
                return pred, nimg, spacing

    def evaluate_single_dataset_mscmrseg(self, seg_model, model_name='best_model', modality='lge', phase='test',
                                         ifhd=True, ifasd=True, weight_dir=None, bs=32, toprint=True,
                                         lge_train_test_split=None, cal_unctnty=False, watch_pat=None, klc=True,
                                         ifhd95=False, crop_size=224, pred_index=0, fold_num=0, save_pred=False,
                                         save_dir='prediction/MSCMRSeg', save_norm=False, verbose=False):
        """
        Function to compute the metrics for a single modality of a single dataset.
        Parameters
        ----------
        seg_model: t.nn.Module
        the segmentation module.
        model_name: str
        the model name to be saved.
        modality: str
        choose from "bssfp" and "lge".
        phase: str
        choose from "train", "valid" and "test".
        ifhd: bool
        whether to calculate HD.
        ifasd: bool
        whether to calculate ASD.
        save_csv: bool
        whether to save the resuls as csv file.
        weight_dir: str
        specify the directory to the weight if load weight.
        bs: int
        the batch size for prediction (only for memory saving).
        toprint: bool
        whether to print out the results.
        (following are not used for FUDA)
        lge_train_test_split: int
        specify from where the training data should be splitted into training and testing data.
        cal_unctnty: bool
        whether to calculate and print out the highest uncertainty (entropy) of the prediction.
        watch_pat: int
        specify the pat_id that should be printed out its uncertainty.

        Returns a dictionary of metrics {dc: [], hd: [], asd: []}.
        -------

        """
        print('Eval on MSCMRSeg')
        device, mycolormap, seg_model = self.get_things_ready(seg_model=seg_model, weight_dir=weight_dir)
        print("Evaluate the mode with {}{}{}".format(self._color_preffixG, device, self._color_suffix))
        uncertainty_list, uncertainty_slice_list = [], []
        if modality == 'lge':
            folder = 'LGE'
        elif modality == 'bssfp':
            folder = 'C0'
        else:
            raise ValueError('modality can only be \'bssfp\' or \'lge\'')
        with torch.no_grad():
            endo_dc, myo_dc, rv_dc = [], [], []
            endo_hd, myo_hd, rv_hd = [], [], []
            endo_asd, myo_asd, rv_asd, = [], [], []
            bg_winscore, endo_winscore, myo_winscore, rv_winscore = [], [], [], []
            ft_norm_list = []
            if phase == 'valid':
                pat_ids = np.arange(1, 6)
            elif phase == 'test':
                pat_ids = config.MSCMRSEG_TEST_FOLD2 if fold_num == 0 else config.MSCMRSEG_TEST_FOLD1

            for pat_id in pat_ids:
                pred, nimg, spacing, ft = self.predict_single_pat_mscmrseg(seg_model=seg_model, device=device,
                                                                           modality=modality, bs=bs, klc=klc, crop_size=crop_size,
                                                                           pred_index=pred_index, pat_id=pat_id, pred_ft=True)
                # mask_path = os.path.join(self._raw_data_dir, 'labels/patient{}_{}_manual.nii.gz'.format(pat_id, folder))
                #
                # nimg = sitk.ReadImage(mask_path)
                # spacing = nimg.GetSpacing()
                # nimg = sitk.GetArrayFromImage(nimg)
                # vol = read_img(pat_id, nimg.shape[0], modality=modality, file_path=self._data_dir)
                # if self._ifclahe:
                #     vol = [self.clahe.apply(tmp[..., 0]) for tmp in vol]
                #     vol = np.stack([vol, vol, vol], axis=-1)
                # if vol.shape[1] != crop_size or vol.shape[2] != crop_size:
                #     vol = crop_volume(vol, crop_size=crop_size // 2)
                # if self._normalization == 'minmax':
                #     x_batch = np.array(vol, np.float32) / 255.
                # elif self._normalization == 'zscore':
                #     # entry = self._df_stat[(self._df_stat['modality'] == modality) & (self._df_stat['pat_id'] == pat_id)]
                #     # mean = entry['mean'].values[0]
                #     # std = entry['std'].values[0]
                #     # x_batch = (np.array(vol, np.float32) - mean) / std
                #     x_batch = (np.array(vol, np.float32) - vol.mean(axis=(1, 2, 3), keepdims=True)) / vol.std(
                #         axis=(1, 2, 3), keepdims=True)
                #     x_batch = x_batch.astype(np.float32)
                # else:
                #     raise NotImplementedError
                # x_batch = np.moveaxis(x_batch, -1, 1)
                # pred = []
                # ft = []  # store the decoder features. Will be used to calculate the feature norm.
                # for i in range(0, len(x_batch), bs):
                #     index = np.arange(i, min(i + bs, len(x_batch)))
                #     imgs = x_batch[index]
                #     pred_temp = seg_model(torch.tensor(imgs).to(device))
                #     pred1 = pred_temp[pred_index] if type(pred_temp) == tuple else pred_temp
                #     if save_norm and type(pred_temp) == tuple:
                #         ft.append(pred_temp[-1].detach().cpu().numpy())
                #     pred.append(torch.softmax(pred1, dim=1).cpu().detach().numpy())
                # pred = np.concatenate(pred, axis=0)
                # pred_soft = pred
                # pred = np.argmax(pred, axis=1)  # (bg, myo, lv, rv)
                # bg_win_mean = pred_soft[:, 0][np.where(pred == 0)].mean()
                # myo_win_mean = pred_soft[:, 1][np.where(pred == 1)].mean()
                # lv_win_mean = pred_soft[:, 2][np.where(pred == 2)].mean()
                # rv_win_mean = pred_soft[:, 3][np.where(pred == 3)].mean()
                # bg_winscore.append(bg_win_mean)
                # myo_winscore.append(myo_win_mean)
                # endo_winscore.append(lv_win_mean)
                # rv_winscore.append(rv_win_mean)
                # del pred_soft
                # if klc:
                #     pred = keep_largest_connected_components(pred)
                if save_pred and (weight_dir is not None):
                    save_dir_pat = Path(save_dir).joinpath(f'pat_{pat_id}') .joinpath(name_the_model(None, model_dir=weight_dir))
                    for enu, pd in enumerate(pred):
                        pred_path = str(save_dir_pat.joinpath(f'pred{pat_id}_{enu}.png'))
                        check_mkdir_parent_dir(pred_path)
                        check_del(str(save_dir_pat.joinpath(f'pred{pat_id}_slice{enu}.png')))
                        plt.axis('off')
                        plt.imshow(pd, cmap=mycolormap, vmax=3, vmin=0)
                        plt.tight_layout()
                        plt.savefig(pred_path, dpi=300, bbox_inches='tight', pad_inches=0)
                        plt.clf()
                if save_norm:
                    ft = np.concatenate(ft, axis=0)  # (bs, 32, H, W)
                    ft = np.moveaxis(ft, 1, -1)  # (bs, H, W, 32）
                    for cls in range(1, 4):
                        mask = np.where(pred == cls)
                        tmp = ft[mask[0], mask[1], mask[2]]  # (N, 32)
                        norm = np.linalg.norm(tmp, axis=-1).astype(np.float16)  # (N, 1)
                        norm = norm.flatten()
                        ft_norm_list += [[model_name, pat_id, cls, list(norm)]]
                topad = int((np.around(nimg.shape[1] * spacing[0]) - pred.shape[1]) // 2)
                pred = np.pad(pred, ((0, 0), (topad, topad), (topad, topad)))
                pred = zoom(pred, (1, nimg.shape[1] / pred.shape[1], nimg.shape[1] / pred.shape[1]), order=0)
                assert pred.shape[1] == nimg.shape[1], 'The shape of the reconstructed prediction and the raw label ' \
                                                       'should be the same'
                nimg = (nimg == 200) * 1 + (nimg == 500) * 2 + (nimg == 600) * 3
                nimg = np.array(nimg, dtype=np.uint8)
                pred = np.array(pred).astype(np.uint8)
                res = metrics(nimg, pred, apply_hd=ifhd, apply_asd=ifasd, pat_id=pat_id, modality=modality,
                              class_name=self.class_name, ifhd95=ifhd95, spacing=(spacing[-1], *spacing[:-1]))
                endo_dc.append(res['lv'][0])
                rv_dc.append(res['rv'][0])
                myo_dc.append(res['myo'][0])
                if res['lv'][1] != -1:
                    endo_hd.append(res['lv'][1])
                if res['rv'][1] != -1:
                    rv_hd.append(res['rv'][1])
                if res['myo'][1] != -1:
                    myo_hd.append(res['myo'][1])
                if res['lv'][2] != -1:
                    endo_asd.append(res['lv'][2])
                if res['rv'][2] != -1:
                    rv_asd.append(res['rv'][2])
                if res['myo'][2] != -1:
                    myo_asd.append(res['myo'][2])
            results = {'endo_dc': endo_dc, 'rv_dc': rv_dc, 'myo_dc': myo_dc,
                       'endo_hd': endo_hd, 'rv_hd': rv_hd, 'myo_hd': myo_hd,
                       'endo_asd': endo_asd, 'rv_asd': rv_asd, 'myo_asd': myo_asd,
                       'bg_winscore': bg_winscore, 'myo_winscore': myo_winscore, 'lv_winscore': endo_winscore,
                       'rv_winscore': rv_winscore}
            results = easy_dic(results)
            if cal_unctnty:
                pat_highest_ucty = np.argmax(uncertainty_list) + pat_ids[0]
                print("The pat id with the highest uncertainty: {}".format(pat_highest_ucty))
                print("The slice with the highest uncertainty in the pat {}: {}".format(pat_highest_ucty, np.argmax(
                    uncertainty_slice_list[np.argmax(uncertainty_list)])))
                print("The pat id with the lowest uncertainty: {}".format(np.argmin(uncertainty_list) + pat_ids[0]))
                if watch_pat:
                    print("The slice with the highest uncertainty in the pat {}: {}".format(watch_pat, np.argmax(
                        uncertainty_slice_list[watch_pat - pat_ids[0]])))
                    print("Uncertainty of the slices of pat {}: {}".format(watch_pat, uncertainty_slice_list[
                        watch_pat - pat_ids[0]]))
                print("Uncertainty list: {}".format(np.round(uncertainty_list, 5)))
                print("The patient with the highest DC: {}".format(np.argmax(endo_dc) + pat_ids[0]))
                print("The patient with the lowest DC: {}".format(np.argmin(endo_dc) + pat_ids[0]))
                print("DC list: {}".format(np.round(endo_dc, 3)))
            measures = self.calculate_messages(endo_dc, rv_dc, myo_dc, endo_hd, rv_hd, myo_hd, endo_asd, rv_asd,
                                               myo_asd, toprint, modality, phase, ifhd, ifasd)
        if verbose:
            return measures, results, pat_ids, ft_norm_list
        else:
            return measures


    def evaluate_single_dataset_mmwhs(self, seg_model, model_name='best_model', modality='mr', phase='test', ifhd=True,
                                      ifasd=True, save_csv=False, save_hd=False, weight_dir=None, bs=32, toprint=True,
                                      lge_train_test_split=None, cal_unctnty=False, watch_pat=None, klc=True, spacing=1,
                                      ifhd95=False, crop_size=224, pred_index=0, fold_num=0, split=0, val_num=0, percent=100,
                                      save_pred=False, volume=False, verbose=False, save_dir='prediction/MMWHS'):
        """
        Function to compute the metrics for a single modality of a single dataset.
        Parameters
        ----------
        seg_model: t.nn.Module
        the segmentation module.
        model_name: str
        the model name to be saved.
        modality: str
        choose from "bssfp" and "lge".
        phase: str
        choose from "train", "valid" and "test".
        ifhd: bool
        whether to calculate HD.
        ifasd: bool
        whether to calculate ASD.
        save_csv: bool
        whether to save the resuls as csv file.
        weight_dir: str
        specify the directory to the weight if load weight.
        bs: int
        the batch size for prediction (only for memory saving).
        toprint: bool
        whether to print out the results.
        (following are not used for FUDA)
        lge_train_test_split: int
        specify from where the training data should be splitted into training and testing data.
        cal_unctnty: bool
        whether to calculate and print out the highest uncertainty (entropy) of the prediction.
        watch_pat: int
        specify the pat_id that should be printed out its uncertainty.

        Returns a dictionary of metrics {dc: [], hd: [], asd: []}.
        -------

        """
        assert fold_num == 0 or fold_num == 1
        print('Eval on MMWHS')
        device = get_device()
        print("Evaluate the mode with {}{}{}".format(self._color_preffixG, device, self._color_suffix))
        colorlist = np.round(
            np.array([[0, 0, 0], [186, 137, 120], [240, 216, 152], [148, 184, 216]]) / 256,
            decimals=2)
        mycolormap = ListedColormap(colors=colorlist, name='mycolor', N=4)
        test_fold1 = np.array(config.train_extra_list[split][1])
        test_fold2 = np.array(config.train_extra_list[split][0])
        num_dict = {'CT': {'valid': config.MMWHS_CT_T_VALID_SET,
                           'test': [test_fold1 + 32, test_fold2 + 32]},
                    'MR': {'valid': config.MMWHS_MR_T_VALID_SET if val_num == 0 else config.MMWHS_MR_T_VALID_SET1,
                           'test': [test_fold1, test_fold2]}}
        uncertainty_list, uncertainty_slice_list = [], []
        try:
            mnmx = load_mnmx_csv(modality, percent)
        except FileNotFoundError:
            mnmx = None
        with torch.no_grad():
            seg_model.eval()
            if weight_dir is not None:
                try:
                    seg_model.load_state_dict(torch.load(weight_dir)['model_state_dict'])
                except:
                    seg_model.load_state_dict(torch.load(weight_dir))
                print("model loaded from {}{}{}".format(self._color_preffixB, weight_dir, self._color_suffix))
            seg_model = seg_model.to(device)
            endo_dc, myo_dc, rv_dc = [], [], []
            endo_hd, myo_hd, rv_hd = [], [], []
            endo_asd, myo_asd, rv_asd, = [], [], []
            if phase == 'valid':
                sample_range = num_dict[modality.upper()]['valid']
            elif phase == 'test':
                sample_range = num_dict[modality.upper()]['test'][fold_num]
            else:
                raise NotImplementedError
            if save_csv:
                import pandas as pd
                csv_path = f'evaluation_mmwhs_f{fold_num}.csv'
                if Path(csv_path).exists():
                    df = pd.read_csv(csv_path)
                    columns = df.columns
                else:
                    columns = ['name']
                    for im in range(sample_range):
                        for slc in range(1, 17):
                            columns += [f'img{im}_slc{slc}']
                    df = pd.DataFrame(columns=columns)
            if save_hd:
                csv_hd = f'evaluate_mmwhs_hd_f{fold_num}.csv'
                if Path(csv_hd).exists():
                    df_hd = pd.read_csv(csv_hd)
                    columns_hd = df_hd.columns
                else:
                    columns_hd = ['name'] + [f'img{im}_slc{slc}' for slc in range(1, 17) for im in range(sample_range)]
                    df_hd = pd.DataFrame(columns=columns_hd)
            if self._raw:
                parent_fold = os.path.join(self._data_dir, f'{modality.upper()}_{"woGT" if phase == "valid" else "withGT"}')
                print(f'parent folder: {parent_fold}')
                suffix = 'nii.gz'
            else:
                parent_fold = os.path.join(self._data_dir, f'{modality.upper()}_{"train" if phase == "valid" else phase}')
                suffix = 'png'
            masks_fold = str(Path(self._data_dir).parent.joinpath(f'CT_MR_2D_Dataset_DA-master/{modality.upper()}_{"woGT" if phase == "valid" else "withGT"}'))
            img_paths_list, mask_paths_list = [], []
            preds = []
            masks = []
            for sample_num in sample_range:
                x_batch = []
                # construct the lists of image paths and mask paths
                img_paths = [os.path.join(parent_fold, f'img{sample_num}_slice{slc_num}.{suffix}') for slc_num in range(1, 17)]
                mask_paths = [os.path.join(masks_fold, f'lab{sample_num}_slice{slc_num}.nii.gz') for slc_num in range(1, 17)]
                assert len(img_paths) == 16 and len(mask_paths) == 16
                img_paths_list += img_paths
                mask_paths_list += mask_paths
                for img_path, mask_path in zip(img_paths, mask_paths):
                    if self._raw:
                        vol, _ = load_raw_data_mmwhs(img_path)
                        if mnmx is None:
                            img = np.array(vol, np.float32)
                            vmin = np.percentile(img, 0.5)
                            vmax = np.percentile(img, 99.5)
                        else:
                            img_name = f'img{sample_num}'
                            vmin, vmax = mnmx.loc[img_name].min99, mnmx.loc[img_name].max99
                        vol = np.clip((np.array(vol, np.float32) - vmin) / (vmax - vmin), 0, 1)
                        vol = (vol * 255).astype(np.uint8)
                        vol = np.stack([vol, vol, vol], axis=-1)
                    else:
                        vol = cv2.imread(img_path)
                        if self._ifclahe:
                            vol = self.clahe.apply(vol[..., 0])
                            vol = np.stack([vol, vol, vol], axis=-1)
                        if vol.shape[1] != crop_size or vol.shape[0] != crop_size:
                            vol = ImageProcessor.crop_volume(vol, crop_size // 2)
                    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))[..., 0]
                    # 205: myo, 500: left ventricle blood cavity; 600 right ventricle blood cavity
                    mask = (mask == 205) * 1 + (mask == 500) * 2 + (mask == 600) * 3
                    if self._normalization == 'zscore':
                        # idx = np.where(vol != 0)
                        # mean, std = vol[idx].mean(), vol[idx].std()
                        mean, std = vol.mean(), vol.std()
                        vol = (np.array(vol, np.float32) - mean) / std
                    x_batch.append(vol)
                    masks.append(mask)
                x_batch = np.array(x_batch)
                if self._normalization == 'minmax':
                    x_batch = x_batch / 255.
                x_batch = np.moveaxis(x_batch, -1, 1)
                pred = []
                # temp = []
                for i in range(0, len(x_batch), bs):
                    index = np.arange(i, min(i + bs, len(x_batch)))
                    imgs = x_batch[index]
                    pred_temp = seg_model(torch.tensor(imgs, dtype=torch.float32).to(device))
                    pred1 = pred_temp[pred_index] if type(pred_temp) == tuple else pred_temp
                    # uncertainty = F.softmax(pred1, dim=1).cpu().detach().numpy()
                    # temp.append(uncertainty)
                    pred.append(pred1.cpu().detach().numpy())
                pred = np.concatenate(pred, axis=0)
                pred = np.argmax(pred, axis=1)
                if klc:
                    pred_klc = []
                    for idx in range(len(pred)):
                        pred_tmp = keep_largest_connected_components(pred[idx: idx + 1])
                        pred_klc.append(pred_tmp)
                    pred = np.concatenate(pred_klc, axis=0)
                pred = np.array(pred).astype(np.uint8)
                preds.append(pred)
                if save_pred:
                    save_dir_pat = Path(save_dir).joinpath(f'pat_{sample_num}').joinpath(
                        name_the_model(None, model_dir=weight_dir))
                    for enu, pd in enumerate(pred):
                        pred_path = str(save_dir_pat.joinpath(f'pred{sample_num}_{enu}.png'))
                        check_mkdir_parent_dir(pred_path)
                        check_del(str(save_dir_pat.joinpath(f'pred{sample_num}_slice{enu}.png')))
                        plt.axis('off')
                        plt.imshow(pd, cmap=mycolormap, vmax=3, vmin=0)
                        plt.tight_layout()
                        plt.savefig(pred_path, dpi=300, bbox_inches='tight', pad_inches=0)
                        plt.clf()
            # x_batch = np.array(x_batch)
            # masks = np.array(masks, dtype=np.uint8)
            # if self._normalization == 'minmax':
            #     x_batch = x_batch / 255.
            # x_batch = np.moveaxis(x_batch, -1, 1)
            # pred = []
            # # temp = []
            # for i in range(0, len(x_batch), bs):
            #     index = np.arange(i, min(i + bs, len(x_batch)))
            #     imgs = x_batch[index]
            #     pred_temp = seg_model(torch.tensor(imgs, dtype=torch.float32).to(device))
            #     pred1 = pred_temp[pred_index] if type(pred_temp) == tuple else pred_temp
            #     # uncertainty = F.softmax(pred1, dim=1).cpu().detach().numpy()
            #     # temp.append(uncertainty)
            #     pred.append(pred1.cpu().detach().numpy())
            # pred = np.concatenate(pred, axis=0)
            # pred = np.argmax(pred, axis=1)
            # if klc:
            #     pred_klc = []
            #     for idx in range(len(pred)):
            #         pred_tmp = keep_largest_connected_components(pred[idx: idx + 1])
            #         pred_klc.append(pred_tmp)
            #     pred = np.concatenate(pred_klc, axis=0)
            # pred = np.array(pred).astype(np.uint8)
            masks = np.array(masks, dtype=np.uint8)
            preds = np.concatenate(preds, axis=0)
            pred = preds[:, 2:-2]
            pred = np.pad(pred, ((0, 0), (0, 0), (8, 8)))
            # masks = masks[:, 2:-2]
            # masks = np.pad(masks, ((0, 0), (0, 0), (8, 8)))
            dc_list, hd_list = [], []
            if volume:
                for i in range(0, len(pred), 16):
                    pd = pred[i: i + 16]
                    mask = masks[i: i + 16]
                    res = metrics(mask, pd, apply_hd=ifhd, apply_asd=ifasd, pat_id=Path(img_path).stem,
                                  modality=modality,
                                  class_name=self.class_name, ifhd95=ifhd95, spacing=(spacing, spacing, spacing))
                    # endo, rv, myo
                    endo_dc, rv_dc, myo_dc, endo_hd, rv_hd, myo_hd, endo_asd, rv_asd, myo_asd = save_results_to_lists(
                        res, endo_dc, rv_dc, myo_dc, endo_hd, rv_hd, myo_hd, endo_asd, rv_asd, myo_asd)
            else:
                for mask, pd, img_path in zip(masks, pred, img_paths_list):
                    res = metrics(mask, pd, apply_hd=ifhd, apply_asd=ifasd, pat_id=Path(img_path).stem,
                                  modality=modality,
                                  class_name=self.class_name, ifhd95=ifhd95, spacing=(spacing, spacing))
                    if save_csv:
                        dc_list.append(np.round((res['lv'][0] + res['rv'][0] + res['myo'][0]) / 3, 3))
                    if save_hd:
                        hd_list.append(np.round((res['lv'][1] + res['rv'][1] + res['myo'][1]) / 3, 3))
                    # endo, rv, myo
                    endo_dc, rv_dc, myo_dc, endo_hd, rv_hd, myo_hd, endo_asd, rv_asd, myo_asd = save_results_to_lists(
                        res, endo_dc, rv_dc, myo_dc, endo_hd, rv_hd, myo_hd, endo_asd, rv_asd, myo_asd)
            results = {'endo_dc': endo_dc, 'rv_dc': rv_dc, 'myo_dc': myo_dc,
                       'endo_hd': endo_hd, 'rv_hd': rv_hd, 'myo_hd': myo_hd,
                       'endo_asd': endo_asd, 'rv_asd': rv_asd, 'myo_asd': myo_asd}
            results = easy_dic(results)
            if cal_unctnty:
                pat_highest_ucty = np.argmax(uncertainty_list) + 0
                print("The pat id with the highest uncertainty: {}".format(pat_highest_ucty))
                print("The slice with the highest uncertainty in the pat {}: {}".format(pat_highest_ucty, np.argmax(
                    uncertainty_slice_list[np.argmax(uncertainty_list)])))
                print("The pat id with the lowest uncertainty: {}".format(np.argmin(uncertainty_list) + 0))
                if watch_pat:
                    print("The slice with the highest uncertainty in the pat {}: {}".format(watch_pat, np.argmax(
                        uncertainty_slice_list[watch_pat - 0])))
                    print("Uncertainty of the slices of pat {}: {}".format(watch_pat, uncertainty_slice_list[
                        watch_pat - 0]))
                print("Uncertainty list: {}".format(np.round(uncertainty_list, 5)))
                print("The patient with the highest DC: {}".format(np.argmax(endo_dc) + 0))
                print("The patient with the lowest DC: {}".format(np.argmin(endo_dc) + 0))
                print("DC list: {}".format(np.round(endo_dc, 3)))
            measures = self.calculate_messages(endo_dc, rv_dc, myo_dc, endo_hd, rv_hd, myo_hd, endo_asd, rv_asd,
                                               myo_asd, toprint, modality, phase, ifhd, ifasd)
        if verbose:
            return measures, results, sample_range, []
        else:
            return measures

    def calculate_messages(self, endo_dc, rv_dc, myo_dc, endo_hd, rv_hd, myo_hd, endo_asd, rv_asd, myo_asd,
                           toprint, modality, phase, ifhd, ifasd):
        mean_endo_dc = np.around(np.mean(np.array(endo_dc)), 3)
        mean_rv_dc = np.around(np.mean(np.array(rv_dc)), 3)
        mean_myo_dc = np.around(np.mean(np.array(myo_dc)), 3)
        std_endo_dc = np.around(np.std(np.array(endo_dc)), 3)
        std_rv_dc = np.around(np.std(np.array(rv_dc)), 3)
        std_myo_dc = np.around(np.std(np.array(myo_dc)), 3)
        if toprint:
            print("Modality: {}, Phase: {}".format(modality, phase))
            print("Ave endo DC: {:.3f}, {:.3f}, Ave rv DC: {:.3f}, {:.3f}, Ave myo DC: {:.3f}, {:.3f}".format(
                mean_endo_dc, std_endo_dc,
                mean_rv_dc,
                std_rv_dc, mean_myo_dc,
                std_myo_dc))
            print("Ave Dice: {:.3f}, {:.3f}".format((mean_endo_dc + mean_rv_dc + mean_myo_dc) / 3.,
                                                    (std_endo_dc + std_rv_dc + std_myo_dc) / 3.))

        if ifhd:
            mean_endo_hd = np.around(np.mean(np.array(endo_hd)), 3)
            mean_rv_hd = np.around(np.mean(np.array(rv_hd)), 3)
            mean_myo_hd = np.around(np.mean(np.array(myo_hd)), 3)
            std_endo_hd = np.around(np.std(np.array(endo_hd)), 3)
            std_rv_hd = np.around(np.std(np.array(rv_hd)), 3)
            std_myo_hd = np.around(np.std(np.array(myo_hd)), 3)
            if toprint:
                print("Ave endo HD: {:.3f}, {:.3f}, Ave rv HD: {:.3f}, {:.3f}, Ave myo HD: {:.3f}, {:.3f}".format(
                    mean_endo_hd, std_endo_hd,
                    mean_rv_hd, std_rv_hd,
                    mean_myo_hd, std_myo_hd))
                print("Ave HD: {:.3f}, {:.3f}".format((mean_endo_hd + mean_rv_hd + mean_myo_hd) / 3.,
                                                      (std_endo_hd + std_rv_hd + std_myo_hd) / 3.))
        else:
            mean_myo_hd, std_myo_hd, mean_endo_hd, std_endo_hd, mean_rv_hd, std_rv_hd = 0, 0, 0, 0, 0, 0
        if ifasd:
            mean_endo_asd = np.around(np.mean(np.array(endo_asd)), 3)
            mean_rv_asd = np.around(np.mean(np.array(rv_asd)), 3)
            mean_myo_asd = np.around(np.mean(np.array(myo_asd)), 3)
            std_endo_asd = np.around(np.std(np.array(endo_asd)), 3)
            std_rv_asd = np.around(np.std(np.array(rv_asd)), 3)
            std_myo_asd = np.around(np.std(np.array(myo_asd)), 3)
            if toprint:
                print(
                    "Ave endo ASD: {:.3f}, {:.3f}, Ave rv ASD: {:.3f}, {:.3f}, Ave myo ASD: {:.3f}, {:.3f}".format(
                        mean_endo_asd, std_endo_asd,
                        mean_rv_asd, std_rv_asd,
                        mean_myo_asd, std_myo_asd))
                print("Ave ASD: {:.3f}, {:.3f}".format((mean_endo_asd + mean_rv_asd + mean_myo_asd) / 3.,
                                                       (std_endo_asd + std_rv_asd + std_myo_asd) / 3.))
        else:
            mean_myo_asd, std_myo_asd, mean_endo_asd, std_endo_asd, mean_rv_asd, std_rv_asd = 0, 0, 0, 0, 0, 0

        if toprint:
            print(
                '{}DC{}: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(self._color_preffixG,
                                                                                self._color_suffix, mean_myo_dc,
                                                                                std_myo_dc, mean_endo_dc, std_endo_dc,
                                                                                mean_rv_dc,
                                                                                std_rv_dc))
            if ifhd:
                print(
                    '{}HD{}: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(self._color_preffixG,
                                                                                    self._color_suffix, mean_myo_hd,
                                                                                    std_myo_hd, mean_endo_hd,
                                                                                    std_endo_hd, mean_rv_hd,
                                                                                    std_rv_hd))
            if ifasd:
                print('{}ASD{}: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(self._color_preffixG,
                                                                                       self._color_suffix, mean_myo_asd,
                                                                                       std_myo_asd,
                                                                                       mean_endo_asd, std_endo_asd,
                                                                                       mean_rv_asd,
                                                                                       std_rv_asd))

        return {'dc': [mean_myo_dc, std_myo_dc, mean_endo_dc, std_endo_dc, mean_rv_dc, std_rv_dc],
                'hd': [mean_myo_hd, std_myo_hd, mean_endo_hd, std_endo_hd, mean_rv_hd, std_rv_hd],
                'asd': [mean_myo_asd, std_myo_asd, mean_endo_asd, std_endo_asd, mean_rv_asd, std_rv_asd]}

    @timeit
    def evaluate(self, seg_model, ifhd=True, ifasd=True, weight_dir=None, bs=16, lge_train_test_split=None):
        bssfp_train = self.evaluate_single_dataset(seg_model=seg_model, modality='bssfp', phase='train', ifhd=ifhd,
                                                   ifasd=ifasd, save_csv=False, weight_dir=weight_dir, bs=bs,
                                                   toprint=False)
        bssfp_val = self.evaluate_single_dataset(seg_model=seg_model, modality='bssfp', phase='valid', ifhd=ifhd,
                                                 ifasd=ifasd, save_csv=False, weight_dir=weight_dir, bs=bs,
                                                 toprint=False)
        lge_val = self.evaluate_single_dataset(seg_model=seg_model, modality='lge', phase='valid', ifhd=ifhd,
                                               ifasd=ifasd, save_csv=False, weight_dir=weight_dir, bs=bs, toprint=False)
        lge_test = self.evaluate_single_dataset(seg_model=seg_model, modality='lge', phase='test', ifhd=ifhd,
                                                ifasd=ifasd, save_csv=False, weight_dir=weight_dir, bs=bs,
                                                toprint=False,
                                                lge_train_test_split=lge_train_test_split)

        return bssfp_train, bssfp_val, lge_val, lge_test


if __name__ == '__main__':
    start = datetime.now()
    import argparse
    from model.DRUNet import Segmentation_model as DR_UNet
    from torch.cuda import get_device_name

    print("Device name: {}".format(get_device_name(0)))
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--restore_from", type=str,
                        default='pretrained/best_DR_UNet.fewshot.lr0.0003.cw0.002.poly.pat_10_lge.adam.e63.Scr0.674.pt',
                        help="Where restore model parameters from.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of images sent to the network in one step.")
    parser.add_argument("--data_dir", type=str, default='../../data/mscmrseg/origin')
    parser.add_argument("--raw_data_dir", type=str, default='../../data/mscmrseg/raw_data')
    parser.add_argument("--modality", type=str, default='lge')
    parser.add_argument("--phase", type=str, default='test')
    parser.add_argument("--klc", action='store_true')
    parser.add_argument("--torch", action='store_true')
    parser.add_argument("--hd", action='store_true')
    parser.add_argument("--asd", action='store_true')
    args = parser.parse_args()
    evaluator = Evaluator(data_dir=args.data_dir, raw_data_dir=args.raw_data_dir, normalization='zscore')
    segmentor = DR_UNet(n_class=4).cuda()
    # if args.torch:
    #     evaluator.evaluate_single_dataset_torch(segmentor, model_name='best_model', modality=args.modality, phase=args.phase, ifhd=args.hd,
    #                                       ifasd=args.asd, save=False, weight_dir=args.restore_from, bs=args.batch_size,
    #                                       toprint=True, lge_train_test_split=None, cal_unctnty=False, watch_pat=None,
    #                                       klc=args.klc)
    # else:
    evaluator.evaluate_single_dataset(segmentor, model_name='best_model', modality=args.modality, phase=args.phase,
                                      ifhd=args.hd, ifhd95=True,
                                      ifasd=args.asd, save_csv=False, weight_dir=args.restore_from,
                                      bs=args.batch_size,
                                      toprint=True, lge_train_test_split=None, cal_unctnty=False,
                                      watch_pat=None,
                                      klc=args.klc)
    end = datetime.now()
    print('Time elapsed: {}'.format(end - start))
