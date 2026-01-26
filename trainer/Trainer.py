import torch.backends.cudnn as cudnn
import torch

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import os
import argparse
import random
import numpy as np

from utils.utils_ import get_device, check_bit_generator, print_device_info, get_summarywriter
import config
from evaluator import Evaluator


class Trainer(ABC):
    def __init__(self):
        self.start_time = datetime.now()
        self.max_epoch_time = 0
        self.start_epoch = 0
        self.max_duration = 24 * 3600 - 5 * 60
        self.device = get_device()
        self.prepare_grocery()
        self.cores = os.cpu_count()

        self.get_argparser()
        self.get_arguments_apdx()
        self.args.num_workers = min(self.cores, self.args.bs) if self.args.num_workers == -1 else self.args.num_workers
        self.prepare_dataloader()
        self.prepare_model()
        self.prepare_optimizers()
        self.prepare_checkpoints()
        self.prepare_losses()
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        self.writer, self.log_dir = get_summarywriter(self.apdx)
        """create the evaluator"""
        self.evaluator = Evaluator(data_dir=self.scratch, raw_data_dir=self.scratch_raw, raw=self.args.raw,
                                   normalization=self.args.normalization, dataset=self.dataset)

    def get_argparser(self):
        # Basic options
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-apdx', type=str, default='')
        """training configuration"""
        self.parser.add_argument('-evalT', action='store_true')
        self.parser.add_argument('-val_num', type=int, default=0)
        """dataset configuration"""
        self.parser.add_argument('-spacing', help='pixel spacing of the mmwhs dataset', type=float, default=1)
        self.parser.add_argument('-noM3AS', action='store_false')
        self.parser.add_argument('-data_dir', type=str, default=config.DATA_DIRECTORY)
        self.parser.add_argument("-raw_data_dir", type=str, default=config.RAW_DATA_DIRECTORY,
                                 help="Path to the directory containing the source dataset.")
        self.parser.add_argument('-rev', action='store_true')
        self.parser.add_argument('-fold', type=int, help='fold number. Only for MMWHS dataset.', default=0)
        self.parser.add_argument('-split', type=int, help='split set', default=0)
        self.parser.add_argument('-scratch', action='store_true')
        self.parser.add_argument('-bs', type=int, default=config.BATCH_SIZE)
        self.parser.add_argument('-crop', type=int, default=config.INPUT_SIZE)
        self.parser.add_argument('-aug_s', action='store_true')
        self.parser.add_argument('-aug_t', action='store_true')
        self.parser.add_argument('-aug_mode', type=str, default='simple')
        self.parser.add_argument('-pin_memory', action='store_true')
        self.parser.add_argument('-num_workers', type=int, default=-1)
        self.parser.add_argument('-normalization', type=str, default='minmax')
        self.parser.add_argument('-clahe', action='store_true')
        self.parser.add_argument("-raw", action='store_true')
        self.parser.add_argument("-percent", type=float, default=100)
        self.parser.add_argument("-save_data", action='store_true')
        """model configuration"""
        self.parser.add_argument("-backbone", help='the model for training.', type=str, default='drunet')
        self.parser.add_argument("-pretrained", action='store_true',
                                 help="whether the loaded model is pretrained (the epoch number will not be loaded if True).")
        self.parser.add_argument("-restore_from", type=str, default=None,
                                 help="Where restore model parameters from (the epoch number will be loaded).")
        self.parser.add_argument("-num_classes", type=int, default=config.NUM_CLASSES,
                                 help="Number of classes to predict (including background).")
        self.parser.add_argument("-nb", type=int, default=4,
                                 help="Number of blocks.")
        self.parser.add_argument("-bd", type=int, default=4,
                                 help="Bottleneck depth.")
        self.parser.add_argument("-filters", type=int, default=32,
                                 help="Number of filters for the first layer.")
        """optimization options"""
        self.parser.add_argument('-optim', help='The optimizer.', type=str, default='sgd')
        self.parser.add_argument('-lr_decay_method', type=str, default=None)
        self.parser.add_argument('-lr', type=float, default=config.LEARNING_RATE)
        self.parser.add_argument('-lr_decay', type=float, default=config.LEARNING_RATE_DECAY)
        self.parser.add_argument('-lr_end', help='the minimum LR when using polynomial decay', type=float, default=0)
        self.parser.add_argument("-momentum", type=float, default=config.MOMENTUM,
                                 help="Momentum component of the optimiser.")
        self.parser.add_argument("-power", type=float, default=config.POWER,
                                 help="Decay parameter to compute the learning rate.")
        self.parser.add_argument("-weight_decay", type=float, default=config.WEIGHT_DECAY,
                                 help="Regularisation parameter for L2-loss.")
        # parser.add_argument('-ns', type=int, default=700)
        self.parser.add_argument('-epochs', type=int, default=config.EPOCHS)
        """weight directory"""
        self.parser.add_argument('-vgg', help='the path to the directory of the weight', type=str,
                                 default='pretrained/vgg_normalised.pth')
        """stylized image"""
        self.parser.add_argument('-style_dir', type=str, default='./style_track')
        self.parser.add_argument('-save_every_epochs', type=int, default=config.SAVE_PRED_EVERY,
                                 help='save the stylized images and checkpoint for every certain epochs')
        """miscellaneous"""
        self.parser.add_argument("-seed", type=int, default=config.RANDOM_SEED,
                                 help="Random seed to have reproducible results.")
        self.add_additional_arguments()
        self.args = self.parser.parse_args()
        del self.parser
        self.args.spacing = .5 if 'DDFSeg' in self.args.data_dir else 1
        if 'mscmrseg' in self.args.data_dir:
            self.dataset = 'mscmrseg'
            self.trgt_modality = 'bssfp' if self.args.rev else 'lge'
            self.src_modality = 'lge' if self.args.rev else 'bssfp'
        elif 'mmwhs' in self.args.data_dir:
            self.dataset = 'mmwhs'
            self.trgt_modality = 'ct' if self.args.rev else 'mr'
            self.src_modality = 'ct' if not self.args.rev else 'mr'
        elif 'Processed_data_nii_uda' in self.args.data_dir:
            self.dataset = 'Processed_data_nii_uda'
            self.trgt_modality = 'BIDMC'
            self.src_modality = 'RUNMC'
        else:
            raise NotImplementedError

    @abstractmethod
    def add_additional_arguments(self):
        pass

    def get_basic_arguments_apdx(self, name):
        self.apdx = f"{name}.{self.dataset}.s{self.args.split}.f{self.args.fold}.v{self.args.val_num}.{self.args.backbone}"
        if not self.args.noM3AS:
            self.apdx += '.noM3AS'
        if self.args.apdx != '':
            self.apdx += f'.{self.args.apdx}'
        if self.args.backbone == 'drunet':
            self.apdx += f".{self.args.filters}.nb{self.args.nb}.bd{self.args.bd}"
        if self.args.clahe:
            self.apdx += '.clahe'
        self.apdx += f'.lr{self.args.lr}'
        if self.args.lr_decay_method is not None:
            self.apdx += f'.{self.args.lr_decay_method}'
            if self.args.lr_decay_method == 'linear':
                self.apdx += f'.decay{self.args.lr_decay}'
            if self.args.lr_decay_method == 'poly':
                self.apdx += f'.power{self.args.power}'
        if self.args.optim == 'sgd':
            self.apdx += f'.mmt{self.args.momentum}'
        if 'DDFSeg' in self.args.data_dir:
            self.apdx += '.res.5'
        if self.args.raw:
            self.apdx += '.raw'
            if self.args.percent != 100:
                self.apdx += f'.pct{self.args.percent}'
        if self.args.aug_s or self.args.aug_t:
            self.apdx += '.aug'
            if self.args.aug_s:
                self.apdx += 's'
            if self.args.aug_t:
                self.apdx += 't'
            if self.args.aug_mode == 'simple':
                self.apdx += 'Sim'
            elif '2' in self.args.aug_mode:
                self.apdx += 'Hvy2'
            else:
                self.apdx += 'Hvy'

    @abstractmethod
    def get_arguments_apdx(self):
        self.apdx = ''

    @abstractmethod
    def prepare_model(self):
        pass

    @abstractmethod
    def prepare_optimizers(self):
        pass

    @abstractmethod
    def prepare_checkpoints(self, **kwargs):
        pass

    @abstractmethod
    def prepare_dataloader(self):
        pass

    def prepare_grocery(self):
        check_bit_generator()
        print_device_info()
        cudnn.benchmark = True
        cudnn.enabled = True
        torch.autograd.set_detect_anomaly(True)

    @abstractmethod
    def train_epoch(self, **kwargs):
        pass

    @abstractmethod
    def train(self, **kwargs):
        pass

    def check_time_elapsed(self, epoch, epoch_start, margin=30 * 60):
        epoch_time_elapsed = datetime.now() - epoch_start
        print(f'Epoch {epoch + 1} time elapsed: {epoch_time_elapsed}')
        self.max_epoch_time = max(epoch_time_elapsed.seconds, self.max_epoch_time)
        if (datetime.now() - self.start_time).seconds > self.max_duration - self.max_epoch_time - margin:
            print("training time elapsed: {}".format(datetime.now() - self.start_time))
            print("max_epoch_time: {}".format(timedelta(seconds=self.max_epoch_time)))
            return True
        return False

    def prepare_losses(self):
        pass

    def stop_training(self, *args):
        pass

    @abstractmethod
    def eval(self, phase='valid'):
        pass
