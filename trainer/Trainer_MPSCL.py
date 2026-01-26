import os
import numpy as np
from tqdm import tqdm
from datetime import datetime

"""torch import"""
import torch
import torch.nn.functional as F

"""utils import"""
from utils.loss import loss_calc, MPCL, dice_loss, mpcl_loss_calc
from utils.utils_ import update_class_center_iter, generate_pseudo_label, prob_2_entropy

"""evaluator import"""
from evaluator import Evaluator

"""trainer import"""
from trainer.Trainer_Advent import Trainer_Advent


class Trainer_MPSCL(Trainer_Advent):
    def __init__(self):
        super().__init__()

    def add_additional_arguments(self):
        super(Trainer_MPSCL, self).add_additional_arguments()
        self.parser.add_argument('-adjust_lr', action='store_true')
        self.parser.add_argument('-uda', action='store_true')

        self.parser.add_argument('-src_temp', type=float, default=1)
        self.parser.add_argument('-src_base_temp', type=float, default=1)
        self.parser.add_argument('-trg_temp', type=float, default=1)
        self.parser.add_argument('-trg_base_temp', type=float, default=1)
        self.parser.add_argument('-src_margin', type=float, default=.4)
        self.parser.add_argument('-trg_margin', type=float, default=.2)

        self.parser.add_argument('-class_center_m', type=float, default=.2)
        self.parser.add_argument('-pixel_sel_th', type=float, default=.25)

        self.parser.add_argument('-w_mpcl_s', type=float, default=1)
        self.parser.add_argument('-w_mpcl_t', type=float, default=.1)

        self.parser.add_argument('-dis_type', type=str, default='origin')

        # self.parser.add_argument('-class_center_init', type=str, default='./class_center_bssfp.npy')

    def get_arguments_apdx(self):
        super(Trainer_MPSCL, self).get_basic_arguments_apdx(name='MPSCL')
        if self.args.multilvl:
            self.apdx += f'.w_seg_aux{self.args.w_seg_aux}'
        self.apdx += f".bs{self.args.bs}"
        self.apdx += f".lr_dis{self.args.lr_dis}.w_dis{self.args.w_dis}"
        if self.args.multilvl:
            self.apdx += f'.w_d_aux{self.args.w_dis_aux}'
        self.apdx += f'.w_mpscl_s{self.args.w_mpcl_s}.t{self.args.w_mpcl_t}'

    def prepare_losses(self):
        self.mpcl_loss_src = MPCL(self.device, num_class=self.args.num_classes, temperature=self.args.src_temp,
                                  base_temperature=self.args.src_base_temp, m=self.args.src_margin)

        self.mpcl_loss_trg = MPCL(self.device, num_class=self.args.num_classes, temperature=self.args.trg_temp,
                                  base_temperature=self.args.trg_base_temp, m=self.args.trg_margin)

    def _log_images(self, img_s, labels_s, pred_s_main, img_t, pred_t_main, epoch):
        num_samples = min(4, img_s.size(0))
        img_s = img_s[:num_samples].detach().cpu()
        labels_s = labels_s[:num_samples].detach().cpu()
        pred_s = torch.argmax(pred_s_main[:num_samples], dim=1).detach().cpu()
        img_t = img_t[:num_samples].detach().cpu()
        pred_t = torch.argmax(pred_t_main[:num_samples], dim=1).detach().cpu()

        def process_image(img):
            if img.dim() == 3:
                img = img.unsqueeze(1)
            min_val = img.min()
            max_val = img.max()
            if max_val > min_val:
                img = (img - min_val) / (max_val - min_val)
            return img

        def process_mask(mask):
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = mask.float()
            if self.args.num_classes > 1:
                mask = mask / (self.args.num_classes - 1)
            return mask

        img_s_proc = process_image(img_s)
        labels_s_proc = process_mask(labels_s)
        pred_s_proc = process_mask(pred_s)
        img_t_proc = process_image(img_t)
        pred_t_proc = process_mask(pred_t)

        pred_t_softmax = F.softmax(pred_t_main[:num_samples].detach().cpu(), dim=1)
        uncertainty_map = prob_2_entropy(pred_t_softmax)
        uncertainty_map = uncertainty_map.sum(dim=1, keepdim=True)
        uncertainty_proc = process_image(uncertainty_map)

        if img_s_proc.size(1) == 1:
            img_s_proc = img_s_proc.repeat(1, 3, 1, 1)
        if labels_s_proc.size(1) == 1:
            labels_s_proc = labels_s_proc.repeat(1, 3, 1, 1)
        if pred_s_proc.size(1) == 1:
            pred_s_proc = pred_s_proc.repeat(1, 3, 1, 1)
        if img_t_proc.size(1) == 1:
            img_t_proc = img_t_proc.repeat(1, 3, 1, 1)
        if pred_t_proc.size(1) == 1:
            pred_t_proc = pred_t_proc.repeat(1, 3, 1, 1)
        if uncertainty_proc.size(1) == 1:
            uncertainty_proc = uncertainty_proc.repeat(1, 3, 1, 1)

        step = epoch + 1
        self.writer.add_images('Images/Source', img_s_proc, step)
        self.writer.add_images('Images/Source_Label', labels_s_proc, step)
        self.writer.add_images('Images/Source_Pred', pred_s_proc, step)
        self.writer.add_images('Images/Target', img_t_proc, step)
        self.writer.add_images('Images/Target_Pred', pred_t_proc, step)
        self.writer.add_images('Images/Target_Uncertainty', uncertainty_proc, step)

    def train_epoch(self, epoch):
        print(f'start to train epoch: {epoch}')
        self.segmentor.train()
        self.d_main.train()
        self.d_aux.train()
        results = {}
        source_domain_label = 1
        target_domain_label = 0
        loss_seg_list, loss_uncertainty, loss_prior_list = [], [], []
        loss_seg_aux_list = []
        loss_adv_list, loss_adv_aux_list, loss_dis_list, loss_dis_aux_list = [], [], [], []
        loss_mpcl_tr_list, loss_mpcl_tg_list = [], []
        d_acc_s, d_acc_t = [], []
        d_aux_acc_s, d_aux_acc_t = [], []
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
            if labels_s.dim() == 4 and labels_s.size(1) == 1:
                labels_s = labels_s.squeeze(1)
            img_t, labels_t, namet = batch_style
            img_t = img_t.to(self.device, non_blocking=self.args.pin_memory)

            pred_s_main, pred_s_aux, dcdr_ft_s = self.segmentor(img_s)
            pred_t_main, pred_t_aux, dcdr_ft_t = self.segmentor(img_t)
            if len(loss_seg_list) == 0:
                print("MPSCL debug - img_s:", img_s.size(), "img_t:", img_t.size())
                print("MPSCL debug - labels_s:", labels_s.size())
                print("MPSCL debug - dcdr_ft_s:", dcdr_ft_s.size(), "dcdr_ft_t:", dcdr_ft_t.size())
                self._log_images(img_s, labels_s, pred_s_main, img_t, pred_t_main, epoch)
            loss_seg = loss_calc(pred_s_main, labels_s, self.device, False) + dice_loss(pred_s_main, labels_s)
            """save the main segmentation loss"""
            loss_seg_list.append(loss_seg.item())
            if self.args.multilvl:
                loss_seg_aux = loss_calc(pred_s_aux, labels_s, self.device, False) + dice_loss(pred_s_aux, labels_s)
                loss_seg += self.args.w_seg_aux * loss_seg_aux
                """save the auxiliary segmentation loss"""
                loss_seg_aux_list.append(loss_seg_aux.item())

            self.centroid_s = update_class_center_iter(dcdr_ft_s, labels_s, self.centroid_s,
                                                       m=self.args.class_center_m,
                                                       num_class=self.args.num_classes)
            hard_pixel_label, pixel_mask = generate_pseudo_label(dcdr_ft_t, self.centroid_s, self.args.pixel_sel_th)
            if len(loss_mpcl_tr_list) == 0:
                print("MPSCL debug - hard_pixel_label shape:", hard_pixel_label.shape, "pixel_mask shape:", pixel_mask.shape)

            mpcl_loss_tr = mpcl_loss_calc(feas=dcdr_ft_s, labels=labels_s,
                                          class_center_feas=self.centroid_s,
                                          loss_func=self.mpcl_loss_src, tag='source')
            loss_mpcl_tr_list.append(mpcl_loss_tr.item())

            mpcl_loss_tg = mpcl_loss_calc(feas=dcdr_ft_t, labels=hard_pixel_label,
                                          class_center_feas=self.centroid_s,
                                          loss_func=self.mpcl_loss_trg,
                                          pixel_sel_loc=pixel_mask, tag='target')
            loss_mpcl_tg_list.append(mpcl_loss_tg.item())

            pred_t_softmax = F.softmax(pred_t_main, dim=1)
            uncertainty_mapT = prob_2_entropy(pred_t_softmax)
            D_out = self.d_main(uncertainty_mapT)
            loss_adv = F.binary_cross_entropy_with_logits(D_out, torch.FloatTensor(
                D_out.data.size()).fill_(
                source_domain_label).cuda())
            # loss_ent = loss_entropy(pred_t_softmax, smooth=smooth, device=self.device)
            # loss_prior = loss_class_prior(pred_t_softmax, self.class_prior, self.args.w_prior, self.device)
            loss_adv_list.append(loss_adv.item())
            loss_adv = self.args.w_dis * loss_adv
            if self.args.multilvl:
                pred_t_softmax_aux = F.softmax(pred_t_aux, dim=1)
                uncertainty_mapT_aux = prob_2_entropy(pred_t_softmax_aux)
                D_out = self.d_aux(uncertainty_mapT_aux)
                loss_adv_aux = F.binary_cross_entropy_with_logits(D_out, torch.FloatTensor(
                    D_out.data.size()).fill_(
                    source_domain_label).cuda())
                loss_adv += self.args.w_dis_aux * loss_adv_aux
                loss_adv_aux_list.append(loss_adv_aux.item())

            (loss_seg + loss_adv + self.args.w_mpcl_s * mpcl_loss_tr + self.args.w_mpcl_t * mpcl_loss_tg).backward()

            for param in self.d_main.parameters():
                param.requires_grad = True
            if self.args.multilvl:
                for param in self.d_aux.parameters():
                    param.requires_grad = True

            if self.args.multilvl:
                pred_s_aux = pred_s_aux.detach()
                d_out_aux = self.d_aux(prob_2_entropy(F.softmax(pred_s_aux, dim=1)))
                loss_d_aux = F.binary_cross_entropy_with_logits(d_out_aux, torch.FloatTensor(
                    d_out_aux.data.size()).fill_(
                    source_domain_label).cuda())
                loss_d_aux = loss_d_aux / 2
                loss_d_aux.backward()
                D_out_s = torch.sigmoid(d_out_aux.detach()).cpu().numpy()
                D_out_s = np.where(D_out_s >= .5, 1, 0)
                d_aux_acc_s.append(np.mean(D_out_s))

            pred_s_main = pred_s_main.detach()
            d_out_main = self.d_main(prob_2_entropy(F.softmax(pred_s_main, dim=1)))
            loss_d_main = F.binary_cross_entropy_with_logits(d_out_main, torch.FloatTensor(
                d_out_main.data.size()).fill_(
                source_domain_label).cuda())
            loss_d_main = loss_d_main / 2
            loss_d_main.backward()
            D_out_s = torch.sigmoid(d_out_main.detach()).cpu().numpy()
            D_out_s = np.where(D_out_s >= .5, 1, 0)
            d_acc_s.append(np.mean(D_out_s))

            # second we train with target
            if self.args.multilvl:
                pred_t_aux = pred_t_aux.detach()
                d_out_aux = self.d_aux(prob_2_entropy(F.softmax(pred_t_aux, dim=1)))
                loss_d_aux = F.binary_cross_entropy_with_logits(d_out_aux, torch.FloatTensor(
                    d_out_aux.data.size()).fill_(
                    target_domain_label).cuda())
                loss_d_aux = loss_d_aux / 2
                loss_d_aux.backward()
                """save the aux discriminator loss"""
                loss_dis_aux_list.append(loss_d_aux.item())

                D_out_t = torch.sigmoid(d_out_aux.detach()).cpu().numpy()
                D_out_t = np.where(D_out_t >= .5, 1, 0)
                d_aux_acc_t.append(1 - np.mean(D_out_t))

            pred_t_main = pred_t_main.detach()
            d_out_main = self.d_main(prob_2_entropy(F.softmax(pred_t_main, dim=1)))
            loss_d_main = F.binary_cross_entropy_with_logits(d_out_main, torch.FloatTensor(
                d_out_main.data.size()).fill_(
                target_domain_label).cuda())
            loss_d_main = loss_d_main / 2
            loss_d_main.backward()
            """save the main discriminator loss"""
            loss_dis_list.append(loss_d_main.item())

            D_out_t = torch.sigmoid(d_out_main.detach()).cpu().numpy()
            D_out_t = np.where(D_out_t >= .5, 1, 0)
            d_acc_t.append(1 - np.mean(D_out_t))

            self.opt.step()
            self.opt_d.step()
            if self.args.multilvl:
                self.opt_d_aux.step()

        results['seg_s'] = sum(loss_seg_list) / len(loss_seg_list)
        results['dis_acc_s'] = sum(d_acc_s) / len(d_acc_s)
        results['dis_acc_t'] = sum(d_acc_t) / len(d_acc_t)
        results['loss_adv'] = sum(loss_adv_list) / len(loss_adv_list)
        results['loss_dis'] = sum(loss_dis_list) / len(loss_dis_list)
        results['loss_mpscl_tr'] = sum(loss_mpcl_tr_list) / len(loss_mpcl_tr_list)
        results['loss_mpscl_tg'] = sum(loss_mpcl_tg_list) / len(loss_mpcl_tg_list)
        # results['loss_uncertainty'] = sum(loss_uncertainty) / len(loss_uncertainty)
        # results['loss_prior'] = sum(loss_prior_list) / len(loss_prior_list)
        if self.args.multilvl:
            results['seg_s_aux'] = sum(loss_seg_aux_list) / len(loss_seg_aux_list)
            results['loss_adv_aux'] = sum(loss_adv_aux_list) / len(loss_adv_aux_list)
            results['loss_dis_aux'] = sum(loss_dis_aux_list) / len(loss_dis_aux_list)
            results['dis_aux_acc_s'] = sum(d_aux_acc_s) / len(d_aux_acc_s)
            results['dis_aux_acc_t'] = sum(d_aux_acc_t) / len(d_aux_acc_t)

        return results

    def train(self):
        """
        :return:
        """

        """mkdir for the stylized images"""
        if not os.path.exists(self.args.style_dir):
            os.makedirs(self.args.style_dir)

        self.args.class_center_init = f'class_center_{"bssfp" if "mscmrseg" in self.args.data_dir else "ct"}_f{self.args.fold}.npy'
        self.centroid_s = np.load(self.args.class_center_init)
        if self.centroid_s.shape[0] != self.args.num_classes:
            if self.centroid_s.shape[0] > self.args.num_classes:
                self.centroid_s = self.centroid_s[:self.args.num_classes]
            else:
                repeat_num = self.args.num_classes - self.centroid_s.shape[0]
                self.centroid_s = np.concatenate(
                    [self.centroid_s, np.repeat(self.centroid_s[-1][None, :], repeat_num, axis=0)],
                    axis=0,
                )
        self.centroid_s = torch.from_numpy(self.centroid_s).float().to(self.device)
        for epoch in tqdm(range(self.start_epoch, self.args.epochs)):
            epoch_start = datetime.now()
            """adjust learning rate with polynomial decay"""
            self.adjust_lr(epoch)

            train_results = self.train_epoch(epoch)

            results = self.eval(modality='target', phase='valid')
            if len(results['dc']) == 2:
                 lge_dice = results['dc'][0]
            else:
                 lge_dice = np.round((results['dc'][0] + results['dc'][2] + results['dc'][4]) / 3, 3)
            
            if self.args.evalT:
                results = self.eval(modality='target', phase='test')
                if len(results['dc']) == 2:
                     lge_dice_test = results['dc'][0]
                else:
                     lge_dice_test = np.round((results['dc'][0] + results['dc'][2] + results['dc'][4]) / 3, 3)

            """record all the experiment results into the tensorboard"""
            print("Writing summary")
            if self.args.evalT:
                self.writer.add_scalars('Dice/LGE', {'Valid': lge_dice, 'Test': lge_dice_test}, epoch + 1)
            else:
                self.writer.add_scalar('Dice/LGE_valid', lge_dice, epoch + 1)
            if self.args.multilvl:
                self.writer.add_scalars('Loss/Seg', {'main': train_results['seg_s'],
                                                     'aux': train_results['seg_s_aux']}, epoch + 1)
                self.writer.add_scalars('Loss/Adv', {'main': train_results['loss_adv'],
                                                     'aux': train_results['loss_adv_aux']}, epoch + 1)
                self.writer.add_scalars('Loss/Dis', {'main': train_results['loss_dis'],
                                                     'aux': train_results['loss_dis_aux']}, epoch + 1)
                self.writer.add_scalars('Acc/main', {'source': train_results['dis_acc_s'],
                                                     'target': train_results['dis_acc_t']}, epoch + 1)
                self.writer.add_scalars('Acc/aux', {'source': train_results['dis_aux_acc_s'],
                                                    'target': train_results['dis_aux_acc_t']}, epoch + 1)
            else:
                self.writer.add_scalar('Loss/Seg', train_results['seg_s'], epoch + 1)
                self.writer.add_scalars('Loss/Adv', {'adv': train_results['loss_adv']}, epoch + 1)
                self.writer.add_scalars('Loss/Dis', {'dis': train_results['loss_dis']}, epoch + 1)
                # self.writer.add_scalar('Loss/Uncertainty', train_results['loss_uncertainty'], epoch + 1)
                # self.writer.add_scalar('Loss/Prior', train_results['loss_prior'], epoch + 1)
                self.writer.add_scalars('Acc/Dis', {'source': train_results['dis_acc_s'],
                                                    'target': train_results['dis_acc_t']}, epoch + 1)
            self.writer.add_scalars('LR', {'Segmentor': self.opt.param_groups[0]['lr'],
                                           'Discriminator': self.opt_d.param_groups[0]['lr']}, epoch + 1)
            self.writer.add_scalars('Loss/MPSCL', {'Source': train_results['loss_mpscl_tr'],
                                                   'Target': train_results['loss_mpscl_tg']}, epoch + 1)

            message = f'Epoch = {epoch + 1:4d}/{self.args.epochs:4d}, loss_seg = {train_results["seg_s"]:.4f}'
            if self.args.multilvl:
                message += f', aux seg = {train_results["seg_s_aux"]:.4f}'
            message += f', dc_valid = {lge_dice:.4f}'
            print(message)

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
