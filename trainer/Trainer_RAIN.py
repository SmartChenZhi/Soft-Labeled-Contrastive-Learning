from datetime import datetime
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable

import model.RAIN as net
from utils.utils_ import save_transferred_images_RAIN, crop_normalize
from utils import timer
import config
from utils.loss import loss_calc
from trainer.Trainer_baseline import Trainer_baseline
from evaluator import Evaluator


class Trainer_RAIN(Trainer_baseline):
    def __init__(self):
        super().__init__()
        self.save_dir = str(Path(self.args.style_dir).joinpath(self.apdx))
        if Path(self.save_dir).exists():
            now = datetime.now()
            self.save_dir = str(self.save_dir) + ".{}.{}".format(now.hour, now.minute)

    def add_additional_arguments(self):
        """
        :param parser:
        :return:
        """
        super(Trainer_RAIN, self).add_additional_arguments()
        """training configuration"""
        self.parser.add_argument('-mulstyle', action='store_true')
        self.parser.add_argument('-mulstyle2', help='whether to use the 2nd type of mulstyle', action='store_true')
        """optimization configurations"""
        self.parser.add_argument("-lr_eps", type=float, default=config.LEARNING_RATE_EPS,
                                 help="Base learning rate for epsilon(sampling).")
        self.parser.add_argument("-eps_iters", type=int, default=config.EPS_ITERS,
                                 help="Number of iterations for each epsilon. Will be used if only 'update_eps' is true")
        self.parser.add_argument("-warmup_epochs", type=int, default=config.WARMUP_EPOCHS,
                                 help="Number of training steps to warm up the segmentor model.")
        self.parser.add_argument('-consist_w', help='the weight for the consistent loss', type=float,
                                 default=config.WEIGHT_CONSIST)
        """weight directory"""
        self.parser.add_argument('-decoder', type=str, default='pretrained/best_decoder.pt')
        self.parser.add_argument('-fc_encoder', type=str, default='pretrained/best_fc_encoder.pt')
        self.parser.add_argument('-fc_decoder', type=str, default='pretrained/best_fc_decoder.pt')
        """choose slice(s), only for one-shot and few-shot training"""
        # pat_id 10, 13, 33, 38, 41
        self.parser.add_argument('-pat_id', help='The patient id to choose.', type=int, default=-1)
        # slice_id 13, 11, 14, 7, 3
        self.parser.add_argument('-slice_id', help='The slice id in the volume.', type=int, default=-1)

    @timer.timeit
    def get_arguments_apdx(self):
        """
        :return:
        """
        super(Trainer_RAIN, self).get_basic_arguments_apdx(name='RAIN')
        self.apdx += f'.eps{self.args.eps_iters}.lrs{self.args.lr_eps}.consist_w{self.args.consist_w}'
        if self.args.mulstyle:
            self.apdx += '.ms'
        if self.args.mulstyle2:
            self.apdx += '.ms2'
        if self.args.pat_id != -1:
            self.apdx += f".pat_{self.args.pat_id}_lge"
        elif self.args.slice_id != -1:
            self.apdx += f"{self.args.slice_id}"

    def prepare_losses(self):
        self.mse_loss = torch.nn.MSELoss()

    @timer.timeit
    def prepare_model(self):
        from model.RAIN import load_rain_models
        super(Trainer_RAIN, self).prepare_model()

        self.vgg_encoder, self.decoder, self.fc_encoder, self.fc_decoder = load_rain_models(self.args.vgg,
                                                                                            self.args.decoder,
                                                                                            self.args.fc_encoder,
                                                                                            self.args.fc_decoder,
                                                                                            device=self.device)
        self.network = net.Net(self.vgg_encoder, self.decoder, self.fc_encoder, self.fc_decoder, init=False)

    def train_epoch(self, epoch):
        print(f'start to train epoch: {epoch}')
        to_save_pic = True
        loss_seg_list = []
        loss_consistent_list = []
        for batch_content, batch_style in zip(self.content_loader, self.style_loader):
            self.segmentor.train()
            sampling = None
            self.opt.zero_grad()
            img_s, labels_s, names = batch_content
            img_s, labels_s = img_s.to(self.device, non_blocking=self.args.pin_memory), labels_s.to(self.device,
                                                                                                    non_blocking=self.args.pin_memory)
            img_t, labels_t, namet = batch_style
            img_t, labels_t = img_t.to(self.device, non_blocking=self.args.pin_memory), labels_t.to(self.device,
                                                                                                    non_blocking=self.args.pin_memory)
            eps_update_times = 1 if epoch < self.args.warmup_epochs else self.args.eps_iters
            for eps_iter in range(eps_update_times):
                if self.args.mulstyle2:
                    rain_content, rain_style = img_s, img_t[0:1]
                elif self.args.mulstyle:
                    rain_content, rain_style = img_s, img_t
                else:
                    rain_content, rain_style = img_s[0:1], img_t[0:1]
                img_style, sampling = self.network.style_transfer(rain_content,
                                                                  rain_style, sampling)
                img_style = torch.mean(img_style, dim=1)
                img_style = torch.stack([img_style, img_style, img_style], dim=1)
                img_style = crop_normalize(img_style, img_s, normalization=self.args.normalization)
                if to_save_pic and (epoch + 1) % self.args.save_every_epochs == 0:
                    save_transferred_images_RAIN(img_style, names, namet, epoch=epoch, iter=eps_iter,
                                                 idx_to_save=np.arange(min(8, len(names))),
                                                 save_dir=self.save_dir,
                                                 stage='warmup' if eps_update_times == 1 else 'RAIN',
                                                 normalization=self.args.normalization)
                """stop saving the images once the first batch has been saved for eps_update_times"""
                to_save_pic = False if (eps_iter + 1) == eps_update_times else True
                pred, pred_norm, _ = self.segmentor(torch.cat([img_style, img_s], dim=0))
                style_size = img_style.size()[0]
                pred_norm_style, pred_norm_s = pred_norm[:style_size], pred_norm[style_size: 2 * style_size]
                """calculate the consistency loss"""
                loss_norm = self.mse_loss(pred_norm_s - pred_norm_style,
                                          torch.zeros(pred_norm_style.size()).to(self.device))
                """calculate the segmentation loss"""
                label_tensor = torch.cat([labels_s[:style_size], labels_s], dim=0)
                loss_seg = loss_calc(pred, label_tensor, self.device, jaccard=True)
                loss_seg_list.append(loss_seg.item())
                loss_consistent_list.append(loss_norm.item())
                """check whether need to retain graph"""
                retain_graph = (epoch >= self.args.warmup_epochs)
                if retain_graph:
                    sampling.require_grad = True
                    sampling.retain_grad()
                    samp_loss = loss_seg
                    samp_loss.backward(retain_graph=retain_graph)
                    grad_data = sampling.grad.data
                    self.opt.zero_grad()
                loss = loss_seg + self.args.consist_w * loss_norm
                loss.backward()
                if retain_graph:
                    sampling = sampling + (self.args.lr_eps / samp_loss.item()) * grad_data
                    sampling = Variable(sampling.detach(), requires_grad=True)
                self.opt.step()

        return sum(loss_seg_list) / len(loss_seg_list), sum(loss_consistent_list) / len(loss_consistent_list)

    @timer.timeit
    def train(self):
        """
        :return:
        """

        """mkdir for the stylized images"""
        if not os.path.exists(self.args.style_dir):
            os.makedirs(self.args.style_dir)

        losses_seg = []
        losses_consistent = []
        lge_dice, lge_dice_test = [], []
        lr = []

        epoch = self.start_epoch

        for epoch in tqdm(range(self.start_epoch, self.args.epochs)):
            """adjust learning rate and save the value for tensorboard"""
            self.adjust_lr(epoch=epoch)
            lr.append(self.opt.param_groups[0]['lr'])
            epoch_start = datetime.now()

            mean_seg_loss, mean_consist_loss = self.train_epoch(epoch)

            losses_seg.append(mean_seg_loss)
            losses_consistent.append(mean_consist_loss)
            print('Epoch = {0:6d}/{1:6d}, loss_seg = {2:.4f} loss_con = {3:.4f}'.format(
                epoch + 1, self.args.epochs, losses_seg[-1], losses_consistent[-1]))
            results = self.eval(modality='target', phase='valid')
            lge_dice.append(np.round((results['dc'][0] + results['dc'][2] + results['dc'][4]) / 3, 3))
            if self.args.evalT:
                results = self.eval(modality='target', phase='test')
                lge_dice_test.append(np.round((results['dc'][0] + results['dc'][2] + results['dc'][4]) / 3, 3))

            tobreak = self.stop_training(epoch, epoch_start, lge_dice)

            self.mcp_segmentor.step(monitor=lge_dice[-1], model=self.segmentor, epoch=epoch + 1,
                                    optimizer=self.opt,
                                    tobreak=tobreak)
            if tobreak:
                break

        print("Writing summary")
        from torch.utils.tensorboard import SummaryWriter
        best_epoch = self.mcp_segmentor.epoch
        best_score = self.mcp_segmentor.best_result
        log_dir = 'runs/{}.e{}.Scr{}'.format(self.apdx, best_epoch,
                                             np.around(best_score, 3))
        writer = SummaryWriter(log_dir=log_dir)
        i = self.start_epoch
        for loss_seg, loss_con, dice, s_lr in zip(losses_seg, losses_consistent, lge_dice, lr):
            writer.add_scalar('Loss/Training_seg', loss_seg, i + 1)
            writer.add_scalar('Loss/Training_consistent', loss_con, i + 1)
            writer.add_scalar('Dice/LGE_valid', dice, i + 1)
            writer.add_scalar('LR/Seg_LR', s_lr, i + 1)
            i += 1
        i = self.start_epoch
        if self.args.evalT:
            for dice, dice_t in zip(lge_dice, lge_dice_test):
                writer.add_scalars('Dice/LGE', {'Valid': dice, 'Test': dice_t}, i + 1)
                i += 1
        else:
            for dice in lge_dice:
                writer.add_scalar('Dice/LGE_valid', dice, i + 1)
                i += 1
        writer.close()
        # load the weights with the bext validation score and do the evaluation
        model_name = '{}.e{}.Scr{}{}'.format(self.mcp_segmentor.best_model_name_base, best_epoch,
                                             np.around(best_score, 3), self.mcp_segmentor.ext)
        print("the weight of the best unet model: {}".format(model_name))
        try:
            self.segmentor.load_state_dict(torch.load(model_name)['model_state_dict'])
            print("segmentor load from state dict")
        except:
            self.segmentor.load_state_dict(torch.load(model_name))
        print("model loaded")

        self.eval(modality='target', phase='test')

        print(f'epoch: {epoch}, start epoch: {self.start_epoch}, save every epochs: {self.args.save_every_epochs}')
        if epoch - self.start_epoch >= self.args.save_every_epochs:
            os.rename(self.save_dir, f'{self.save_dir}.e{best_epoch}.Scr{best_score}')
            print(f'stylized images saved in {self.save_dir}.e{best_epoch}.Scr{best_score}')
        return
