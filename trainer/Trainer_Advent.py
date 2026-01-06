"""torch import"""
import torch
import torch.nn.functional as F

import os
import numpy as np
from tqdm import tqdm
from datetime import datetime

from utils.loss import loss_calc, loss_entropy, loss_class_prior

from evaluator import Evaluator

from trainer.Trainer_AdaptSeg import Trainer_AdapSeg


class Trainer_Advent(Trainer_AdapSeg):
    def __init__(self):
        super().__init__()
        if self.args.cls_prior:
            self.calculate_class_prior()

    def calculate_class_prior(self):
        print("Calculating class prior from source dataset...")
        class_counts = torch.zeros(self.args.num_classes).to(self.device)
        
        with torch.no_grad():
            for batch in tqdm(self.content_loader, desc="Calculating Prior"):
                _, labels_s, _ = batch
                labels_s = labels_s.to(self.device)
                for c in range(self.args.num_classes):
                    class_counts[c] += (labels_s == c).sum()
        
        self.class_prior = class_counts / class_counts.sum()
        print(f"Calculated class prior: {self.class_prior.cpu().numpy()}")

    def add_additional_arguments(self):
        super(Trainer_Advent, self).add_additional_arguments()
        self.parser.add_argument('-ent_min', help='whether to apply direct entropy minimization', action='store_true')
        self.parser.add_argument('-w_ent', help='the weight for the entropy loss', type=float, default=1e-3)

        self.parser.add_argument('-cls_prior', help='whether to apply class prior', action='store_true')
        self.parser.add_argument('-w_prior', help='the weight for the class prior', type=float, default=.5)

    def get_arguments_apdx(self):
        super(Trainer_Advent, self).get_basic_arguments_apdx(name='Advent')
        if self.args.multilvl:
            self.apdx += f'.mutlvl.w_seg_aux{self.args.w_seg_aux}'
        self.apdx += f".lr_dis{self.args.lr_dis}"
        if self.args.adjust_lr_dis:
            self.apdx += '.decay_lr_dis'
        self.apdx += f".w_dis{self.args.w_dis}"
        if self.args.multilvl:
            self.apdx += f'.w_d_aux{self.args.w_dis_aux}'
        if self.args.ent_min:
            self.apdx += f".w_ent{self.args.w_ent}"
        if self.args.cls_prior:
            self.apdx += f'.prior{self.args.w_prior}'
        self.apdx += f".bs{self.args.bs}"

    def train_epoch(self, epoch):
        print(f'start to train epoch: {epoch}')
        self.segmentor.train()
        self.d_main.train()
        if self.args.multilvl:
            self.d_aux.train()
        smooth = 1e-10
        results = {}
        source_domain_label = 1
        target_domain_label = 0
        loss_seg_list, loss_seg_aux_list, loss_uncertainty, loss_prior_list = [], [], [], []
        loss_adv_list, loss_adv_aux_list, loss_dis_list, loss_dis_aux_list = [], [], [], []
        d_acc_s, d_acc_t, d_aux_acc_s, d_aux_acc_t = [], [], [], []
        for batch_content, batch_style in zip(self.content_loader, self.style_loader):
            self.opt_d.zero_grad()
            self.opt.zero_grad()
            for param in self.d_main.parameters():
                param.requires_grad = False
            if self.args.multilvl:
                self.opt_d_aux.zero_grad()
                for param in self.d_aux.parameters():
                    param.requires_grad = False

            img_s, labels_s, names = batch_content
            img_s, labels_s = img_s.to(self.device, non_blocking=self.args.pin_memory), \
                              labels_s.to(self.device, non_blocking=self.args.pin_memory)
            img_t, labels_t, namet = batch_style
            img_t = img_t.to(self.device, non_blocking=self.args.pin_memory)

            pred_s, pred_s_aux, _ = self.segmentor(img_s)
            loss_seg = loss_calc(pred_s, labels_s, self.device, jaccard=True)
            """save the segmentation loss"""
            loss_seg_list.append(loss_seg.item())
            if self.args.multilvl:
                loss_seg_aux = loss_calc(pred_s_aux, labels_s, self.device, jaccard=True)
                """save the aux segmentation loss"""
                loss_seg_aux_list.append(loss_seg_aux.item())
                loss_seg += self.args.w_seg_aux * loss_seg_aux

            pred_t, pred_t_aux, _ = self.segmentor(img_t)
            pred_t_softmax = F.softmax(pred_t, dim=1)
            uncertainty_mapT = -1.0 * pred_t_softmax * torch.log(pred_t_softmax + smooth)
            D_out = self.d_main(uncertainty_mapT)
            loss_adv = F.binary_cross_entropy_with_logits(D_out, torch.FloatTensor(
                D_out.data.size()).fill_(
                source_domain_label).cuda())
            """save the main adversarial loss"""
            loss_adv_list.append(loss_adv.item())
            loss_seg += self.args.w_dis * loss_adv

            if self.args.multilvl:
                pred_t_softmax = F.softmax(pred_t_aux)
                uncertainty_mapT_aux = -1.0 * pred_t_softmax * torch.log(pred_t_softmax + smooth)
                D_out = self.d_aux(uncertainty_mapT_aux)
                loss_adv_aux = F.binary_cross_entropy_with_logits(D_out, torch.FloatTensor(
                    D_out.data.size()).fill_(
                    source_domain_label).cuda())
                """save the auxiliary adversarial loss"""
                loss_adv_aux_list.append(loss_adv_aux.item())
                loss_seg += self.args.w_dis_aux * loss_adv_aux

            if self.args.ent_min:
                loss_ent = loss_entropy(pred_t_softmax, smooth=smooth, device=self.device)
                """save the entropy minimization loss"""
                loss_uncertainty.append(loss_ent.item())
                loss_seg += self.args.w_ent * loss_ent

            if self.args.cls_prior:
                loss_prior = loss_class_prior(pred_t_softmax, self.class_prior, self.args.w_prior, self.device)
                """save the class prior loss"""
                loss_prior_list.append(loss_prior.item())
                loss_seg += loss_prior

            loss_seg.backward()

            """train discriminators"""
            for param in self.d_main.parameters():
                param.requires_grad = True
            if self.args.multilvl:
                for param in self.d_aux.parameters():
                    param.requires_grad = True

            pred_s_softmax = F.softmax(pred_s, dim=1)
            uncertainty_mapS = (-1.0 * pred_s_softmax * torch.log(pred_s_softmax + smooth)).detach()
            uncertainty_mapT = uncertainty_mapT.detach()

            D_out_s = self.d_main(uncertainty_mapS)
            loss_D_s = F.binary_cross_entropy_with_logits(D_out_s, torch.FloatTensor(D_out_s.data.size()).fill_(
                source_domain_label).cuda())
            loss_D_s = loss_D_s / 2
            loss_D_s.backward()
            D_out_s = torch.sigmoid(D_out_s.detach()).cpu().numpy()
            D_out_s = np.where(D_out_s >= .5, 1, 0)
            d_acc_s.append(np.mean(D_out_s))

            D_out_t = self.d_main(uncertainty_mapT)
            loss_D_t = F.binary_cross_entropy_with_logits(D_out_t, torch.FloatTensor(D_out_t.data.size()).fill_(
                target_domain_label).cuda())
            loss_D_t = loss_D_t / 2
            loss_D_t.backward()
            D_out_t = torch.sigmoid(D_out_t.detach()).cpu().numpy()
            D_out_t = np.where(D_out_t >= .5, 1, 0)
            d_acc_t.append(1 - np.mean(D_out_t))

            """save the main discirminator loss"""
            loss_dis_list.append((loss_D_s + loss_D_t).item())

            if self.args.multilvl:
                pred_s_softmax = F.softmax(pred_s_aux, dim=1)
                uncertainty_mapS_aux = (-1.0 * pred_s_softmax * torch.log(pred_s_softmax + smooth)).detach()
                uncertainty_mapT_aux = uncertainty_mapT_aux.detach()

                D_out_s = self.d_aux(uncertainty_mapS_aux)
                loss_D_s = F.binary_cross_entropy_with_logits(D_out_s, torch.FloatTensor(D_out_s.data.size()).fill_(
                    source_domain_label).cuda())
                loss_D_s = loss_D_s / 2
                loss_D_s.backward()
                D_out_s = torch.sigmoid(D_out_s.detach()).cpu().numpy()
                D_out_s = np.where(D_out_s >= .5, 1, 0)
                d_aux_acc_s.append(np.mean(D_out_s))

                D_out_t = self.d_aux(uncertainty_mapT_aux)
                loss_D_t = F.binary_cross_entropy_with_logits(D_out_t, torch.FloatTensor(D_out_t.data.size()).fill_(
                    target_domain_label).cuda())
                loss_D_t = loss_D_t / 2
                loss_D_t.backward()
                D_out_t = torch.sigmoid(D_out_t.detach()).cpu().numpy()
                D_out_t = np.where(D_out_t >= .5, 1, 0)
                d_aux_acc_t.append(1 - np.mean(D_out_t))

                """save the main discirminator loss"""
                loss_dis_aux_list.append((loss_D_s + loss_D_t).item())

            self.opt.step()
            self.opt_d.step()
            if self.args.multilvl:
                self.opt_d_aux.step()

        results['seg_s'] = sum(loss_seg_list) / len(loss_seg_list)
        results['dis_acc_s'] = sum(d_acc_s) / len(d_acc_s)
        results['dis_acc_t'] = sum(d_acc_t) / len(d_acc_t)
        results['loss_adv'] = sum(loss_adv_list) / len(loss_adv_list)
        results['loss_dis'] = sum(loss_dis_list) / len(loss_dis_list)
        if self.args.ent_min:
            results['loss_uncertainty'] = sum(loss_uncertainty) / len(loss_uncertainty)
        if self.args.cls_prior:
            results['loss_prior'] = sum(loss_prior_list) / len(loss_prior_list)
        if self.args.multilvl:
            results['seg_s_aux'] = sum(loss_seg_aux_list) / len(loss_seg_aux_list)
            results['dis1_acc_s'] = sum(d_aux_acc_s) / len(d_aux_acc_s)
            results['dis1_acc_t'] = sum(d_aux_acc_t) / len(d_aux_acc_t)
            results['loss_adv_aux'] = sum(loss_adv_aux_list) / len(loss_adv_aux_list)
            results['loss_dis1'] = sum(loss_dis_aux_list) / len(loss_dis_aux_list)

        return results

    def train(self):
        """
        :return:
        """
        print('start to train')
        print("Evaluator created.")

        """mkdir for the stylized images"""
        if not os.path.exists(self.args.style_dir):
            os.makedirs(self.args.style_dir)

        for epoch in tqdm(range(self.start_epoch, self.args.epochs)):
            epoch_start = datetime.now()
            """adjust learning rate with polynomial decay"""
            self.adjust_lr(epoch)

            train_results = self.train_epoch(epoch)

            results = self.eval(modality='target', phase='valid')
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
            if self.args.ent_min:
                self.writer.add_scalar('Loss/Uncertainty', train_results['loss_uncertainty'], epoch + 1)
            if self.args.cls_prior:
                self.writer.add_scalar('Loss/Prior', train_results['loss_prior'], epoch + 1)
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
                                               'Dis': self.opt_d.param_groups[0]['lr']}, epoch + 1)
            else:
                self.writer.add_scalar('Loss/Seg', train_results['seg_s'], epoch + 1)
                self.writer.add_scalars('Loss/Adv', {'adv': train_results['loss_adv']}, epoch + 1)
                self.writer.add_scalars('Loss/Dis', {'dis': train_results['loss_dis']}, epoch + 1)
                self.writer.add_scalars('LR', {'Segmentor': self.opt.param_groups[0]['lr'],
                                               'Discriminator': self.opt_d.param_groups[0]['lr']}, epoch + 1)

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
        d_name = f'{self.modelcheckpoint_d.best_model_name_base}.e{best_epoch}.Scr{np.around(best_score, 3)}' \
                 f'{self.modelcheckpoint_d.ext}'
        d_msg = f"The corresponding discriminators: \n\t{d_name}"
        if self.args.multilvl:
            d_aux_name = f'{self.modelcheckpoint_d_aux.best_model_name_base}.e{best_epoch}.Scr{np.around(best_score, 3)}' \
                         f'{self.modelcheckpoint_d_aux.ext}'
            d_msg += f"\n\t{d_aux_name}"
        print(d_msg)

        """test the model with the test data"""
        try:
            self.segmentor.load_state_dict(torch.load(model_name)['model_state_dict'])
            print("segmentor load from state dict")
        except:
            self.segmentor.load_state_dict(torch.load(model_name))
        print("model loaded")

        self.eval(modality='target', phase='test')
        return
