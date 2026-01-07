"""torch import"""
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

import os
import numpy as np
from tqdm import tqdm
from datetime import datetime

from utils.loss import loss_calc
from utils.lr_adjust import adjust_learning_rate
from utils import timer

from evaluator import Evaluator

from trainer.Trainer_baseline import Trainer_baseline


class Trainer_AdapSeg(Trainer_baseline):
    def __init__(self):
        super().__init__()
        self.args.num_workers = min(self.args.num_workers, int(self.cores // 2))

    def add_additional_arguments(self):
        super(Trainer_AdapSeg, self).add_additional_arguments()
        self.parser.add_argument('-w_seg_aux', help='the lambda of the auxiliary segmentation losses.', type=float,
                                 default=0.1)
        self.parser.add_argument('-lr_dis', help='the learning rate for the discriminators', type=float, default=1e-4)
        # self.parser.add_argument('-lr_dis_aux', help='the learning rate for the discriminators', type=float, default=2e-4)
        self.parser.add_argument('-adjust_lr_dis', action='store_true')
        self.parser.add_argument('-mmt1', help='the momentum for the discriminators', type=float, default=0.9)
        self.parser.add_argument('-mmt', help='the momentum for the discriminators', type=float, default=0.99)
        self.parser.add_argument('-w_dis', help='the weight for the discriminators loss', type=float, default=1e-3)
        self.parser.add_argument('-w_dis_aux', type=float, default=2e-4)
        self.parser.add_argument('-restore_d', help='the weight dir of the discriminator', type=str, default=None)
        self.parser.add_argument('-restore_d_aux', help='the weight dir of the discriminator1', type=str, default=None)
        self.parser.add_argument('-d_label_smooth', help='label smoothing for discriminator', type=float, default=0.0)
        self.parser.add_argument('-d_update_freq', help='update frequency for discriminator', type=int, default=1)
        self.parser.add_argument('-adv_warmup_epochs', help='warmup epochs for segmentation before adversarial training', type=int, default=0)

    def get_arguments_apdx(self):
        super(Trainer_AdapSeg, self).get_basic_arguments_apdx(name='AdaptSeg')

        self.apdx += f".bs{self.args.bs}"
        self.apdx += f".lr_dis{self.args.lr_dis}.w_dis{self.args.w_dis}"
        if self.args.d_label_smooth > 0:
            self.apdx += f".dls{self.args.d_label_smooth}"
        if self.args.d_update_freq > 1:
            self.apdx += f".duf{self.args.d_update_freq}"
        if self.args.adv_warmup_epochs > 0:
            self.apdx += f".wup{self.args.adv_warmup_epochs}"
        if self.args.multilvl:
            self.apdx += f'.mutlvl.w_d_aux{self.args.w_dis_aux}'
            self.apdx += f'.wsegaux{self.args.w_seg_aux}'

    @timer.timeit
    def prepare_model(self):
        from model.GAN import UncertaintyDiscriminator
        super(Trainer_AdapSeg, self).prepare_model()

        self.d_main = UncertaintyDiscriminator(in_channel=self.args.num_classes)
        if self.args.restore_d:
            checkpoint = torch.load(self.args.restore_d)
            self.d_main.load_state_dict(checkpoint['model_state_dict'])
            print("Discriminator load from state dict: {}".format(os.path.basename(self.args.restore_d)))
        self.d_main.train()
        self.d_main.to(self.device)
        self.d_aux = UncertaintyDiscriminator(in_channel=self.args.num_classes)
        if self.args.restore_d_aux:
            checkpoint = torch.load(self.args.restore_d_aux)
            self.d_aux.load_state_dict(checkpoint['model_state_dict'])
            print("Discriminator1 load from state dict: {}".format(os.path.basename(self.args.restore_d_aux)))
        self.d_aux.train()
        self.d_aux.to(self.device)

    @timer.timeit
    def prepare_checkpoints(self, mode='max'):
        from utils.callbacks import ModelCheckPointCallback
        weight_root_dir = './weights/'
        super(Trainer_AdapSeg, self).prepare_checkpoints(mode=mode)

        """create the discriminator checkpoints"""
        d1_weight_dir = weight_root_dir + 'out_dis_{}.pt'.format(self.apdx)
        best_d1_weight_dir = weight_root_dir + 'best_out_dis_{}.pt'.format(self.apdx)
        self.modelcheckpoint_d = ModelCheckPointCallback(n_epochs=self.args.epochs, save_best=True,
                                                         mode=mode,
                                                         best_model_dir=best_d1_weight_dir,
                                                         save_last_model=True,
                                                         model_name=d1_weight_dir,
                                                         entire_model=False)
        d1_weight_dir = weight_root_dir + 'out_dis1_{}.pt'.format(self.apdx)
        best_d1_weight_dir = weight_root_dir + 'best_out_dis1_{}.pt'.format(self.apdx)
        self.modelcheckpoint_d_aux = ModelCheckPointCallback(n_epochs=self.args.epochs, save_best=True,
                                                             mode=mode,
                                                             best_model_dir=best_d1_weight_dir,
                                                             save_last_model=True,
                                                             model_name=d1_weight_dir,
                                                             entire_model=False)
        print('discriminator checkpoints created')

    @timer.timeit
    def prepare_optimizers(self):
        super(Trainer_AdapSeg, self).prepare_optimizers()
        # if self.args.multilvl:
        self.opt_d_aux = torch.optim.Adam(self.d_aux.parameters(), lr=self.args.lr_dis,
                                          betas=(self.args.mmt1, self.args.mmt))
        if self.args.restore_d_aux:
            checkpoint = torch.load(self.args.restore_d_aux)
            try:
                self.opt_d_aux.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer loaded from state dict: {}".format(os.path.basename(self.args.restore_d_aux)))
            except Exception as e:
                print(f'Error when loading the optimizer: {e}')
        self.opt_d_aux.zero_grad()

        self.opt_d = torch.optim.Adam(self.d_main.parameters(), lr=self.args.lr_dis,
                                      betas=(self.args.mmt1, self.args.mmt))
        if self.args.restore_d:
            checkpoint = torch.load(self.args.restore_d)
            try:
                self.opt_d.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer loaded from state dict: {}".format(os.path.basename(self.args.restore_d)))
            except Exception as e:
                print(f'Error when loading the optimizer: {e}')
            self.opt_d.zero_grad()
        print('Discriminators optimizers created')

    def adjust_lr(self, epoch):
        super(Trainer_AdapSeg, self).adjust_lr(epoch)
        if self.args.adjust_lr_dis:
            if self.args.lr_decay_method == 'poly' or self.args.lr_decay_method is None:
                adjust_learning_rate(self.opt_d, epoch, self.args.lr_dis, warmup_epochs=0,
                                     power=self.args.power, epochs=self.args.epochs)
                if self.args.multilvl:
                    adjust_learning_rate(self.opt_d_aux, epoch, self.args.lr_dis, warmup_epochs=0,
                                         power=self.args.power, epochs=self.args.epochs)

    def train_epoch(self, epoch):
        print(f'start to train epoch: {epoch}')
        self.segmentor.train()
        self.d_main.train()
        self.d_aux.train()
        resultls = {}
        source_domain_label = 1
        target_domain_label = 0
        loss_seg_list = []
        loss_seg_aux_list = []
        loss_adv_list, loss_adv_aux_list, loss_dis_list, loss_dis1_list = [], [], [], []
        d1_acc_s, d1_acc_t, d_acc_s, d_acc_t = [], [], [], []
        
        # Label smoothing
        d_source_label = 1.0 - self.args.d_label_smooth
        d_target_label = self.args.d_label_smooth

        for batch_idx, (batch_content, batch_style) in enumerate(zip(self.content_loader, self.style_loader)):
            self.opt_d.zero_grad()
            self.opt_d_aux.zero_grad()
            self.opt.zero_grad()
            for param in self.d_main.parameters():
                param.requires_grad = False
            for param in self.d_aux.parameters():
                param.requires_grad = False
            img_s, labels_s, names = batch_content
            img_s, labels_s = img_s.to(self.device, non_blocking=self.args.pin_memory), \
                              labels_s.to(self.device, non_blocking=self.args.pin_memory)

            pred_s, pred_s_aux, _ = self.segmentor(img_s)
            loss_seg = loss_calc(pred_s, labels_s, self.device, jaccard=True)
            """save the segmentation losses"""
            loss_seg_list.append(loss_seg.item())
            if self.args.multilvl:
                loss_seg_aux = loss_calc(pred_s_aux, labels_s, self.device, jaccard=True)
                loss_seg_aux_list.append(loss_seg_aux.item())
                loss_seg += self.args.w_seg_aux * loss_seg_aux

            loss_adv, loss_adv_aux = 0, 0
            if epoch >= self.args.adv_warmup_epochs:
                img_t, labels_t, namet = batch_style
                img_t = img_t.to(self.device, non_blocking=self.args.pin_memory)
                pred_t, pred_t_aux, _ = self.segmentor(img_t)
                D_out = self.d_main(F.softmax(pred_t, dim=1))
                loss_adv = F.binary_cross_entropy_with_logits(D_out, torch.FloatTensor(
                    D_out.data.size()).fill_(
                    source_domain_label).cuda())
                if self.args.multilvl:
                    D_out1 = self.d_aux(F.softmax(pred_t_aux, dim=1))
                    loss_adv_aux = F.binary_cross_entropy_with_logits(D_out1, torch.FloatTensor(
                        D_out1.data.size()).fill_(
                        source_domain_label).cuda())
                    loss_adv_aux_list.append(loss_adv_aux.item())
            """save the adversarial losses"""
            loss_adv_list.append(loss_adv.item() if torch.is_tensor(loss_adv) else loss_adv)
            (loss_seg + self.args.w_dis * loss_adv + self.args.w_dis_aux * loss_adv_aux).backward()

            # Discriminator update frequency and warmup
            if epoch < self.args.adv_warmup_epochs or batch_idx % self.args.d_update_freq != 0:
                self.opt.step()
                continue

            for param in self.d_main.parameters():
                param.requires_grad = True
            for param in self.d_aux.parameters():
                param.requires_grad = True

            pred_s = pred_s.detach()
            pred_s_aux = pred_s_aux.detach()
            pred_t_aux = pred_t_aux.detach()
            pred_t = pred_t.detach()

            if self.args.multilvl:
                D_out_s = self.d_aux(F.softmax(pred_s_aux, dim=1))
                loss_D_s = F.binary_cross_entropy_with_logits(D_out_s, torch.FloatTensor(D_out_s.data.size()).fill_(
                    d_source_label).cuda())
                loss_D_s = loss_D_s / 2
                loss_D_s.backward()
                D_out_s = torch.sigmoid(D_out_s.detach()).cpu().numpy()
                D_out_s = np.where(D_out_s >= .5, 1, 0)
                d1_acc_s.append(np.mean(D_out_s))

                D_out_t_aux = self.d_aux(F.softmax(pred_t_aux, dim=1))
                loss_D_t1 = F.binary_cross_entropy_with_logits(D_out_t_aux,
                                                               torch.FloatTensor(D_out_t_aux.data.size()).fill_(
                                                                   d_target_label).cuda())
                loss_D_t1 = loss_D_t1 / 2
                loss_D_t1.backward()
                D_out_t_aux = torch.sigmoid(D_out_t_aux.detach()).cpu().numpy()
                D_out_t_aux = np.where(D_out_t_aux >= .5, 1, 0)
                d1_acc_t.append(1 - np.mean(D_out_t_aux))

                loss_dis1_list.append((loss_D_s + loss_D_t1).item())

            D_out_s = self.d_main(F.softmax(pred_s, dim=1))
            loss_D_s = F.binary_cross_entropy_with_logits(D_out_s, torch.FloatTensor(D_out_s.data.size()).fill_(
                d_source_label).cuda())
            loss_D_s = loss_D_s / 2
            loss_D_s.backward()
            D_out_s = torch.sigmoid(D_out_s.detach()).cpu().numpy()
            D_out_s = np.where(D_out_s >= .5, 1, 0)
            d_acc_s.append(np.mean(D_out_s))

            D_out_t = self.d_main(F.softmax(pred_t, dim=1))
            loss_D_t = F.binary_cross_entropy_with_logits(D_out_t, torch.FloatTensor(D_out_t.data.size()).fill_(
                d_target_label).cuda())
            loss_D_t = loss_D_t / 2
            loss_D_t.backward()
            D_out_t = torch.sigmoid(D_out_t.detach()).cpu().numpy()
            D_out_t = np.where(D_out_t >= .5, 1, 0)
            d_acc_t.append(1 - np.mean(D_out_t))

            loss_dis_list.append((loss_D_s + loss_D_t).item())
            """update the discriminator optimizer"""
            self.opt_d_aux.step()
            self.opt_d.step()
            """update the segmentor optimizer"""
            self.opt.step()

        resultls['seg_s'] = sum(loss_seg_list) / len(loss_seg_list)
        resultls['dis_acc_s'] = sum(d_acc_s) / len(d_acc_s) if len(d_acc_s) > 0 else 0.0
        resultls['dis_acc_t'] = sum(d_acc_t) / len(d_acc_t) if len(d_acc_t) > 0 else 0.0
        resultls['loss_adv'] = sum(loss_adv_list) / len(loss_adv_list)
        resultls['loss_dis'] = sum(loss_dis_list) / len(loss_dis_list) if len(loss_dis_list) > 0 else 0.0
        if self.args.multilvl:
            resultls['seg_s_aux'] = sum(loss_seg_aux_list) / len(loss_seg_aux_list)
            resultls['dis1_acc_s'] = sum(d1_acc_s) / len(d1_acc_s) if len(d1_acc_s) > 0 else 0.0
            resultls['dis1_acc_t'] = sum(d1_acc_t) / len(d1_acc_t) if len(d1_acc_t) > 0 else 0.0
            resultls['loss_adv_aux'] = sum(loss_adv_aux_list) / len(loss_adv_aux_list)
            resultls['loss_dis1'] = sum(loss_dis1_list) / len(loss_dis1_list) if len(loss_dis1_list) > 0 else 0.0

        """save the visualization results"""
        if 'img_s' in locals():
            resultls['vis_img_s'] = img_s.detach().cpu()
            resultls['vis_pred_s'] = torch.argmax(pred_s, dim=1).detach().cpu()
            resultls['vis_label_s'] = labels_s.detach().cpu()
        if 'img_t' in locals():
            resultls['vis_img_t'] = img_t.detach().cpu()
            resultls['vis_pred_t'] = torch.argmax(pred_t, dim=1).detach().cpu()
            if 'labels_t' in locals():
                resultls['vis_label_t'] = labels_t.detach().cpu()

        return resultls

    def train(self):
        """
        :return:
        """

        """mkdir for the stylized images"""
        if not os.path.exists(self.args.style_dir):
            os.makedirs(self.args.style_dir)

        for epoch in tqdm(range(self.start_epoch, self.args.epochs)):
            epoch_start = datetime.now()
            """adjust learning rate"""
            self.adjust_lr(epoch)

            train_results = self.train_epoch(epoch)

            results = self.eval(phase='valid')
            lge_dice = np.round((results['dc'][0] + results['dc'][2] + results['dc'][4]) / 3, 3)
            if self.args.evalT:
                results = self.eval(modality='target', phase='test')
                lge_dice_test = np.round((results['dc'][0] + results['dc'][2] + results['dc'][4]) / 3, 3)

            """record all the experiment results into the tensorboard"""
            print("Writing summary")
            if self.args.evalT:
                self.writer.add_scalars('Dice/LGE', {'Valid': lge_dice, 'Test': lge_dice_test}, epoch + 1)
            else:
                self.writer.add_scalar('Dice/LGE_valid', lge_dice, epoch + 1)
            self.writer.add_scalars('Acc/Dis', {'source': train_results['dis_acc_s'],
                                                'target': train_results['dis_acc_t']}, epoch + 1)
            if self.args.multilvl:
                self.writer.add_scalars('Loss/Seg', {'main': train_results['seg_s'],
                                                     'aux': train_results['seg_s_aux']}, epoch + 1)
                self.writer.add_scalars('Loss/Adv', {'main': train_results['loss_adv'],
                                                     'aux': train_results['loss_adv_aux']}, epoch + 1)
                self.writer.add_scalars('Loss/Dis', {'dis': train_results['loss_dis'],
                                                     'dis1': train_results['loss_dis1']}, epoch + 1)
                self.writer.add_scalars('Acc/Dis_aux', {'source': train_results['dis1_acc_s'],
                                                        'target': train_results['dis1_acc_t']}, epoch + 1)
                self.writer.add_scalars('LR', {'Seg': self.opt.param_groups[0]['lr'],
                                               'Dis': self.opt_d.param_groups[0]['lr'],
                                               'Dis_aux': self.opt_d_aux.param_groups[0]['lr']}, epoch + 1)
            else:
                self.writer.add_scalar('Loss/Seg', train_results['seg_s'], epoch + 1)
                self.writer.add_scalar('Loss/Adv', train_results['loss_adv'], epoch + 1)
                self.writer.add_scalar('Loss/Dis', train_results['loss_dis'], epoch + 1)
                self.writer.add_scalars('LR', {'Seg': self.opt.param_groups[0]['lr'],
                                               'Dis': self.opt_d.param_groups[0]['lr']}, epoch + 1)

            """visualize the segmentation results"""
            if 'vis_img_s' in train_results:
                self.writer.add_image('Train/Image_Source',
                                      make_grid(train_results['vis_img_s'][:4], normalize=True), epoch + 1)
                self.writer.add_image('Train/Label_Source',
                                      make_grid(train_results['vis_label_s'][:4].unsqueeze(1).float(),
                                                normalize=True), epoch + 1)
                self.writer.add_image('Train/Pred_Source',
                                      make_grid(train_results['vis_pred_s'][:4].unsqueeze(1).float(),
                                                normalize=True), epoch + 1)
            if 'vis_img_t' in train_results:
                self.writer.add_image('Train/Image_Target',
                                      make_grid(train_results['vis_img_t'][:4], normalize=True), epoch + 1)
                self.writer.add_image('Train/Pred_Target',
                                      make_grid(train_results['vis_pred_t'][:4].unsqueeze(1).float(),
                                                normalize=True), epoch + 1)
            if 'vis_label_t' in train_results:
                self.writer.add_image('Train/Label_Target',
                                      make_grid(train_results['vis_label_t'][:4].unsqueeze(1).float(),
                                                normalize=True), epoch + 1)

            print(
                f'Epoch = {epoch + 1:4d}/{self.args.epochs:4d}, loss_seg = {train_results["seg_s"]:.4f}, dc_valid = {lge_dice:.4f}')

            tobreak = self.stop_training(epoch, epoch_start, lge_dice)

            self.mcp_segmentor.step(monitor=lge_dice, model=self.segmentor, epoch=epoch + 1,
                                    optimizer=self.opt,
                                    tobreak=tobreak)
            self.modelcheckpoint_d.step(monitor=lge_dice, model=self.d_main, epoch=epoch + 1,
                                        optimizer=self.opt_d,
                                        tobreak=tobreak)
            if self.args.multilvl:
                self.modelcheckpoint_d_aux.step(monitor=lge_dice, model=self.d_aux, epoch=epoch + 1,
                                                optimizer=self.opt_d_aux,
                                                tobreak=tobreak)
            if tobreak:
                break

        self.writer.close()
        best_epoch = self.mcp_segmentor.epoch
        best_score = self.mcp_segmentor.best_result
        log_dir = 'runs/{}.e{}.Scr{}'.format(self.apdx, best_epoch,
                                             np.around(best_score, 3))
        os.rename(self.log_dir, log_dir)
        # load the weights with the bext validation score and do the evaluation
        model_name = '{}.e{}.Scr{}{}'.format(self.mcp_segmentor.best_model_name_base, best_epoch,
                                             np.around(best_score, 3), self.mcp_segmentor.ext)
        print("the weight of the best unet model: {}".format(model_name))

        """test the model with the test data"""
        try:
            self.segmentor.load_state_dict(torch.load(model_name)['model_state_dict'])
            print("segmentor load from state dict")
        except:
            self.segmentor.load_state_dict(torch.load(model_name))
        print("model loaded")
        self.eval(modality='target', phase='test')
        return
