from datetime import datetime
import os
import numpy as np
from tqdm import tqdm

import torch

#from dataset.data_generator_mscmrseg import prepare_dataset
from dataset.data_generator_mmwhs import prepare_dataset as prepare_dataset_mmwhs
from dataset.data_generator_mmwhs_raw import prepare_dataset as prepare_dataset_mmwhs_raw
from utils.lr_adjust import adjust_learning_rate, adjust_learning_rate_custom
from utils.utils_ import save_batch_data
from utils import timer
import config
from utils.loss import loss_calc
from trainer.Trainer import Trainer


class Trainer_baseline(Trainer):
    def __init__(self):
        super().__init__()

    def add_additional_arguments(self):
        """
        :param parser:
        :return:
        """

        """dataset configuration"""
        self.parser.add_argument("-train_with_t", action='store_true')
        self.parser.add_argument("-train_with_s", action='store_true')
        """evaluation configuration"""
        self.parser.add_argument("-eval_bs", type=int, default=config.EVAL_BS,
                                 help="Number of images sent to the network in a batch during evaluation.")
        self.parser.add_argument('-toggle_klc',
                                 help='Whether to apply keep_largest_component in evaluation during training.',
                                 action='store_false')
        self.parser.add_argument('-hd95', action='store_true')
        self.parser.add_argument('-multilvl', help='if apply multilevel network', action='store_true')
        self.parser.add_argument('-estop', help='if apply early stop', action='store_true')
        self.parser.add_argument('-stop_epoch', type=int, default=200,
                                 help='The number of epochs as the tolerance to stop the training.')

    @timer.timeit
    def get_arguments_apdx(self):
        """
        :return:
        """
        assert self.args.train_with_s or self.args.train_with_t, "at least train on one domain."

        super(Trainer_baseline, self).get_basic_arguments_apdx(name='Base')
        self.apdx += f".bs{self.args.bs}"
        self.apdx += '.trainW'
        if self.args.train_with_s:
            self.apdx += 's'
        if self.args.train_with_t:
            self.apdx += 't'
        if self.args.normalization == 'zscore':
            self.apdx += '.zscr'
        elif self.args.normalization == 'minmax':
            self.apdx += '.mnmx'
        print(f'apdx: {self.apdx}')

    @timer.timeit
    def prepare_dataloader(self):
        if self.dataset == 'mscmrseg':
            pass
            #self.scratch, self.scratch_raw, self.content_loader, self.style_loader = prepare_dataset(self.args)
        elif self.dataset == 'mmwhs':
            print('importing raw data...')
            if self.args.raw:
                from pathlib import Path
                self.args.data_dir = str(Path(self.args.data_dir).parent.joinpath('CT_MR_2D_Dataset_DA-master'))
                self.scratch, self.scratch_raw, self.content_loader, self.style_loader = prepare_dataset_mmwhs_raw(self.args)
            else:
                self.scratch, self.scratch_raw, self.content_loader, self.style_loader = prepare_dataset_mmwhs(self.args)
        else:
            raise NotImplementedError

    @timer.timeit
    def prepare_model(self):
        if self.args.backbone == 'unet':
            from model.unet_model import UNet
            self.segmentor = UNet(n_channels=3, n_classes=self.args.num_classes)
        elif self.args.backbone == 'drunet':
            from model.DRUNet import Segmentation_model as DR_UNet
            self.segmentor = DR_UNet(filters=self.args.filters, n_block=self.args.nb, bottleneck_depth=self.args.bd,
                                     n_class=self.args.num_classes, multilvl=self.args.multilvl, args=self.args)
            if self.args.restore_from:
                checkpoint = torch.load(self.args.restore_from)
                try:
                    self.segmentor.load_state_dict(checkpoint['model_state_dict'], strict=True)
                except:
                    self.segmentor.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif self.args.backbone == 'deeplabv2':
            from model.deeplabv2 import get_deeplab_v2
            self.segmentor = get_deeplab_v2(num_classes=self.args.num_classes, multi_level=self.args.multilvl,
                                            input_size=224)
            if self.args.restore_from:
                checkpoint = torch.load(self.args.restore_from)
                if self.args.pretrained:
                    new_params = self.segmentor.state_dict().copy()
                    for i in checkpoint:
                        i_parts = i.split('.')
                        if not i_parts[1] == 'layer5':
                            new_params['.'.join(i_parts[1:])] = checkpoint[i]
                    self.segmentor.load_state_dict(new_params)
                else:
                    self.segmentor.load_state_dict(checkpoint['model_state_dict'])
        elif 'resnet' in self.args.backbone or 'efficientnet' in self.args.backbone or \
                'mobilenet' in self.args.backbone or 'densenet' in self.args.backbone or 'ception' in self.args.backbone or \
                'se_resnet' in self.args.backbone or 'skresnext' in self.args.backbone:
            from model.segmentation_models import segmentation_models
            self.segmentor = segmentation_models(name=self.args.backbone, pretrained=False,
                                                 decoder_channels=(512, 256, 128, 64, 32), in_channel=3,
                                                 classes=4, multilvl=self.args.multilvl, args=self.args)
            if self.args.restore_from:
                checkpoint = torch.load(self.args.restore_from)
                try:
                    self.segmentor.load_state_dict(checkpoint['model_state_dict'], strict=True)
                    print('model loaded strict')
                except:
                    self.segmentor.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print('model loaded no strict')
            elif self.args.pretrained:
                from utils.utils_ import get_pretrained_checkpoint
                checkpoint = get_pretrained_checkpoint(self.args.backbone)
                self.segmentor.encoder.load_state_dict(checkpoint)
        else:
            raise NotImplementedError

        if self.args.restore_from and (not self.args.pretrained) and 'epoch' in checkpoint.keys():
            try:
                self.start_epoch = self.start_epoch if self.args.pretrained else checkpoint['epoch']
            except Exception as e:
                self.start_epoch = 0
                print(f'Error when loading the epoch number: {e}')

        self.segmentor.train()
        self.segmentor.to(self.device)

    @timer.timeit
    def prepare_checkpoints(self, mode='max'):
        from utils.callbacks import ModelCheckPointCallback, EarlyStopCallback
        weight_root_dir = './weights/'
        if not os.path.exists(weight_root_dir):
            os.mkdir(weight_root_dir)
        weight_dir = os.path.join(weight_root_dir, self.apdx + '.pt')
        best_weight_dir = os.path.join(weight_root_dir, "best_" + self.apdx + '.pt')
        # create the model check point
        self.mcp_segmentor = ModelCheckPointCallback(n_epochs=self.args.epochs, save_best=True,
                                                     mode=mode,
                                                     best_model_dir=best_weight_dir,
                                                     save_last_model=True,
                                                     model_name=weight_dir,
                                                     entire_model=False)
        self.earlystop = EarlyStopCallback(mode=mode, stop_criterion_len=self.args.stop_epoch)
        print('model checkpoint created')

    @timer.timeit
    def prepare_optimizers(self):
        if self.args.backbone == 'deeplabv2':
            params = self.segmentor.optim_parameters(self.args.lr)
        # self.args.backbone == 'drunet' or ('resnet' in self.args.backbone)
        else:
            params = self.segmentor.parameters()
        if self.args.optim == 'sgd':
            self.opt = torch.optim.SGD(params, lr=self.args.lr, momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optim == 'adam':
            self.opt = torch.optim.Adam(params, lr=self.args.lr, betas=(0.9, 0.99))
        else:
            raise NotImplementedError
        if self.args.restore_from:
            checkpoint = torch.load(self.args.restore_from)
            if 'optimizer_state_dict' in checkpoint.keys():
                try:
                    self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Optimizer loaded from state dict: {}".format(os.path.basename(self.args.restore_from)))
                except Exception as e:
                    print(f'Error when loading the optimizer: {e}')
        self.opt.zero_grad()
        print('Segmentor optimizer created')

    def adjust_lr(self, epoch):
        if self.args.lr_decay_method == 'poly':
            adjust_learning_rate(optimizer=self.opt, epoch=epoch, lr=self.args.lr, warmup_epochs=0,
                                 power=self.args.power,
                                 epochs=self.args.epochs)
        elif self.args.lr_decay_method == 'linear':
            adjust_learning_rate_custom(optimizer=self.opt, lr=self.args.lr, lr_decay=self.args.lr_decay,
                                        epoch=epoch)
        elif self.args.lr_decay_method is None:
            pass
        else:
            raise NotImplementedError

    def eval(self, modality='target', phase='valid', toprint=None, fold=None):
        if phase == 'valid':
            results = self.evaluator.evaluate_single_dataset(seg_model=self.segmentor, ifhd=False, ifasd=False,
                                                             modality=self.trgt_modality if modality == 'target' else self.src_modality,
                                                             phase=phase, bs=self.args.eval_bs, toprint=True if toprint is None else toprint,
                                                             klc=self.args.toggle_klc, crop_size=self.args.crop, spacing=self.args.spacing,
                                                             percent=self.args.percent)
        elif phase == 'test':
            results = self.evaluator.evaluate_single_dataset(seg_model=self.segmentor, ifhd=True, ifasd=True,
                                                             modality=self.trgt_modality if modality == 'target' else self.src_modality,
                                                             phase=phase, bs=self.args.eval_bs, toprint=True if toprint is None else toprint,
                                                             spacing=self.args.spacing,
                                                             ifhd95=self.args.hd95, save_csv=False,
                                                             weight_dir=None, klc=True if self.dataset == 'mscmrseg' else False,
                                                             lge_train_test_split=None, crop_size=self.args.crop,
                                                             pred_index=0, fold_num=self.args.fold if fold is None else fold,
                                                             split=self.args.split, percent=self.args.percent)
        else:
            raise NotImplementedError
        return results

    def stop_training(self, epoch, epoch_start, monitor):
        tobreak = self.check_time_elapsed(epoch, epoch_start) or (self.earlystop.step(monitor) and self.args.estop)
        return tobreak

    def train_epoch(self, epoch):
        print(f'start to train epoch: {epoch}')
        results = {}
        loss_seg_list = []

        if self.args.train_with_s:
            for batch_content in self.content_loader:
                self.segmentor.train()
                self.opt.zero_grad()
                img_s, labels_s, names = batch_content
                img_s, labels_s = img_s.to(self.device, non_blocking=self.args.pin_memory), labels_s.to(self.device,
                                                                                                        non_blocking=self.args.pin_memory)

                out = self.segmentor(img_s)
                pred = out[0] if type(out) == tuple else out
                # calculate the segmentation loss
                loss_seg = loss_calc(pred, labels_s, self.device, jaccard=True)
                loss_seg_list.append(loss_seg.item())
                loss_seg.backward()
                self.opt.step()
            results['seg_s'] = sum(loss_seg_list) / len(loss_seg_list)

        if self.args.train_with_t:
            loss_seg_t_list = []
            for batch_style in self.style_loader:
                self.segmentor.train()
                self.opt.zero_grad()
                img_t, labels_t, namet = batch_style
                if self.args.save_data:
                    save_batch_data(self.args.data_dir, img_t.numpy(), labels_t.numpy(), namet, self.args.normalization, self.args.aug_mode)
                img_t, labels_t = img_t.to(self.device, non_blocking=self.args.pin_memory), labels_t.to(self.device,
                                                                                                        non_blocking=self.args.pin_memory)

                out = self.segmentor(img_t)
                pred = out[0] if type(out) == tuple else out
                # calculate the segmentation loss
                loss_seg = loss_calc(pred, labels_t, self.device, jaccard=True)
                loss_seg_t_list.append(loss_seg.item())
                loss_seg.backward()
                self.opt.step()
            results['seg_t'] = sum(loss_seg_t_list) / len(loss_seg_t_list)

        return results

    @timer.timeit
    def train(self):
        # results = self.eval(modality='target', phase='test')
        for epoch in tqdm(range(self.start_epoch, self.args.epochs)):
            """adjust learning rate and save the value for tensorboard"""
            self.adjust_lr(epoch=epoch)
            epoch_start = datetime.now()

            train_results = self.train_epoch(epoch)

            msg = f'Epoch = {epoch + 1:6d}/{self.args.epochs:6d}'
            if self.args.train_with_s:
                msg += f', loss_seg = {train_results["seg_s"]:.4f}'
            if self.args.train_with_t:
                msg += f', loss_seg_t = {train_results["seg_t"]: .4f}'
            print(msg)
            results = self.eval(modality='target', phase='valid')
            lge_dice = np.round((results['dc'][0] + results['dc'][2] + results['dc'][4]) / 3, 3)
            if self.args.evalT:
                results = self.eval(modality='target', phase='test')
                lge_dice_test = np.round((results['dc'][0] + results['dc'][2] + results['dc'][4]) / 3, 3)
                self.writer.add_scalars('Dice/LGE', {'Valid': lge_dice, 'Test': lge_dice_test}, epoch + 1)
            else:
                self.writer.add_scalar('Dice/LGE_valid', lge_dice, epoch + 1)

            tobreak = self.stop_training(epoch, epoch_start, lge_dice)

            self.mcp_segmentor.step(monitor=lge_dice, model=self.segmentor, epoch=epoch + 1,
                                    optimizer=self.opt,
                                    tobreak=tobreak)

            if self.args.train_with_s:
                if self.args.train_with_t:
                    self.writer.add_scalars('Loss Seg',
                                            {'Source': train_results['seg_s'], 'Target': train_results['seg_t']},
                                            epoch + 1)
                else:
                    self.writer.add_scalar('Loss Seg/Source', train_results['seg_s'], epoch + 1)
            else:
                self.writer.add_scalar('Loss Seg/Target', train_results['seg_t'], epoch + 1)
            self.writer.add_scalar('LR/Seg_LR', self.opt.param_groups[0]['lr'], epoch + 1)

            if tobreak:
                break

        self.writer.close()
        best_epoch = self.mcp_segmentor.epoch
        best_score = self.mcp_segmentor.best_result
        log_dir_new = 'runs/{}.e{}.Scr{}'.format(self.apdx, best_epoch,
                                                 np.around(best_score, 3))
        os.rename(self.log_dir, log_dir_new)
        # load the weights with the bext validation score and do the evaluation
        print("the weight of the best unet model: {}".format(self.mcp_segmentor.best_model_save_dir))
        try:
            self.segmentor.load_state_dict(torch.load(self.mcp_segmentor.best_model_save_dir)['model_state_dict'])
            print("segmentor load from state dict")
        except:
            self.segmentor.load_state_dict(torch.load(self.mcp_segmentor.best_model_save_dir))
        print("model loaded")
        self.eval(modality='target', phase='test')
        if self.args.train_with_s:
            self.eval(modality='target', phase='test', fold=1 - self.args.fold)
        self.eval(modality='source', phase='test')
        return
