"""torch import"""
import torch
import torch.nn.functional as F

import os
import numpy as np
from tqdm import tqdm
from datetime import datetime

from dataset.data_generator_mscmrseg import prepare_dataset
from dataset.data_generator_mmwhs import prepare_dataset as prepare_dataset_mmwhs
from dataset.data_generator_mmwhs_raw import prepare_dataset as prepare_dataset_mmwhs_raw
from utils.loss import loss_calc, batch_NN_loss, batch_pairwise_dist
from utils.lr_adjust import adjust_learning_rate
from utils import timer
from model.segmentation_models import segmentation_model_point
from trainer.Trainer_AdaptSeg import Trainer_AdapSeg


class Trainer_AdaptEvery(Trainer_AdapSeg):
    def __init__(self):
        super().__init__()
        self.args.num_workers = min(self.args.num_workers, int(self.cores // 2))

    def add_additional_arguments(self):
        super(Trainer_AdaptEvery, self).add_additional_arguments()
        # self.parser.add_argument('-w_dis', help='the weight for the discriminators loss', type=float, default=1e-3)
        # self.parser.add_argument('-w_dis_aux', type=float, default=2e-4)
        self.parser.add_argument('-w_d_ent', help='the weight for the discriminators loss', type=float, default=1e-3)
        self.parser.add_argument('-w_d_point', help='the weight for the discriminators loss', type=float, default=1e-3)
        self.parser.add_argument('-wp', type=float, default=1.)
        # self.parser.add_argument('-restore_d', help='the weight dir of the discriminator', type=str, default=None)
        # self.parser.add_argument('-restore_d_aux', help='the weight dir of the discriminator1', type=str, default=None)
        self.parser.add_argument('-restore_d_ent', help='the weight dir of the entropy discriminator1', type=str,
                                 default=None)
        self.parser.add_argument('-restore_d_point', help='the weight dir of the point cloud discriminator1', type=str,
                                 default=None)

    def get_arguments_apdx(self):
        super(Trainer_AdaptEvery, self).get_basic_arguments_apdx(name='AdaptEvery')

        self.apdx += f".bs{self.args.bs}"
        self.apdx += f".lr_d{self.args.lr_dis}.wp{self.args.wp}.d{self.args.w_dis}"
        # if self.args.multilvl:
        self.apdx += f'.aux{self.args.w_dis_aux}'
        self.apdx += f'.ent{self.args.w_d_ent}'
        self.apdx += f'.point{self.args.w_d_point}'
        self.apdx += f'.wsegaux{self.args.w_seg_aux}'

    @timer.timeit
    def prepare_model(self):
        from model.GAN import UncertaintyDiscriminator
        from model.PointNetCls import PointNetCls

        self.segmentor = segmentation_model_point(name=self.args.backbone, pretrained=False,
                                                  decoder_channels=(512, 256, 128, 64, 32), in_channel=3,
                                                  classes=4, multilvl=True, fc_inch=4, extpn=False)
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

        if self.args.restore_from and 'epoch' in checkpoint.keys():
            try:
                self.start_epoch = self.start_epoch if self.args.pretrained else checkpoint['epoch']
            except Exception as e:
                self.start_epoch = 0
                print(f'Error when loading the epoch number: {e}')

        self.segmentor.train()
        self.segmentor.to(self.device)

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

        self.d_ent = UncertaintyDiscriminator(in_channel=self.args.num_classes)
        if self.args.restore_d_ent:
            checkpoint = torch.load(self.args.restore_d_ent)
            self.d_ent.load_state_dict(checkpoint['model_state_dict'])
            print("Discriminator load from state dict: {}".format(os.path.basename(self.args.restore_d_ent)))
        self.d_ent.train()
        self.d_ent.to(self.device)

        self.d_point = PointNetCls(kernel_size=2)
        if self.args.restore_d_point:
            checkpoint = torch.load(self.args.restore_d_point)
            self.d_point.load_state_dict(checkpoint['model_state_dict'])
            print("Point Discriminator load from state dict: {}".format(os.path.basename(self.args.restore_d_point)))
        self.d_point.train()
        self.d_point.to(self.device)

    @timer.timeit
    def prepare_checkpoints(self, mode='max'):
        from utils.callbacks import ModelCheckPointCallback
        weight_root_dir = './weights/'
        super(Trainer_AdaptEvery, self).prepare_checkpoints(mode=mode)

        """create the discriminator checkpoints"""
        d2_weight_dir = weight_root_dir + 'ent_d_{}.pt'.format(self.apdx)
        best_d2_weight_dir = weight_root_dir + 'best_ent_d_{}.pt'.format(self.apdx)
        self.modelcheckpoint_d_ent = ModelCheckPointCallback(n_epochs=self.args.epochs, save_best=True,
                                                             mode=mode,
                                                             best_model_dir=best_d2_weight_dir,
                                                             save_last_model=True,
                                                             model_name=d2_weight_dir,
                                                             entire_model=False)
        d3_weight_dir = weight_root_dir + 'point_d_{}.pt'.format(self.apdx)
        best_d3_weight_dir = weight_root_dir + 'best_point_d_{}.pt'.format(self.apdx)
        self.modelcheckpoint_d_point = ModelCheckPointCallback(n_epochs=self.args.epochs, save_best=True,
                                                               mode=mode,
                                                               best_model_dir=best_d3_weight_dir,
                                                               save_last_model=True,
                                                               model_name=d3_weight_dir,
                                                               entire_model=False)
        print('discriminator checkpoints created')

    @timer.timeit
    def prepare_optimizers(self):
        super(Trainer_AdaptEvery, self).prepare_optimizers()

        self.opt_d_ent = torch.optim.Adam(self.d_ent.parameters(), lr=self.args.lr_dis,
                                          betas=(self.args.mmt1, self.args.mmt))
        if self.args.restore_d:
            checkpoint = torch.load(self.args.restore_d)
            try:
                self.opt_d_ent.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer loaded from state dict: {}".format(os.path.basename(self.args.restore_d)))
            except Exception as e:
                print(f'Error when loading the optimizer: {e}')
            self.opt_d_ent.zero_grad()
        print('Discriminators optimizers created')

        self.opt_d_point = torch.optim.Adam(self.d_point.parameters(), lr=self.args.lr_dis,
                                            betas=(self.args.mmt1, self.args.mmt))
        if self.args.restore_d:
            checkpoint = torch.load(self.args.restore_d)
            try:
                self.opt_d_point.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer loaded from state dict: {}".format(os.path.basename(self.args.restore_d)))
            except Exception as e:
                print(f'Error when loading the optimizer: {e}')
            self.opt_d_point.zero_grad()
        print('Discriminators optimizers created')

    def adjust_lr(self, epoch):
        super(Trainer_AdaptEvery, self).adjust_lr(epoch)
        if self.args.adjust_lr_dis:
            if self.args.lr_decay_method == 'poly':
                adjust_learning_rate(self.opt_d_ent, epoch, self.args.lr_dis, warmup_epochs=0,
                                     power=self.args.power, epochs=self.args.epochs)
                adjust_learning_rate(self.opt_d_point, epoch, self.args.lr_dis, warmup_epochs=0,
                                     power=self.args.power, epochs=self.args.epochs)

    @timer.timeit
    def prepare_dataloader(self):
        if self.dataset == 'mscmrseg':
            self.scratch, self.scratch_raw, self.content_loader, self.style_loader = prepare_dataset(self.args,
                                                                                                     aug_counter=True,
                                                                                                     vert=True)
        elif self.dataset == 'mmwhs':
            print('importing raw data...')
            if self.args.raw:
                from pathlib import Path
                self.args.data_dir = str(Path(self.args.data_dir).parent.joinpath('CT_MR_2D_Dataset_DA-master'))
                self.scratch, self.scratch_raw, self.content_loader, self.style_loader = prepare_dataset_mmwhs_raw(self.args,
                                                                                                           aug_counter=True,
                                                                                                           vert=True)
            else:
                self.scratch, self.scratch_raw, self.content_loader, self.style_loader = prepare_dataset_mmwhs(self.args,
                                                                                                           aug_counter=True,
                                                                                                           vert=True)
        else:
            raise NotImplementedError

    def train_epoch(self, epoch):
        print(f'start to train epoch: {epoch}')
        smooth = 1e-10
        self.segmentor.train()
        self.d_main.train()
        self.d_aux.train()
        self.d_ent.train()
        self.d_point.train()
        resultls = {}
        source_domain_label = 1
        target_domain_label = 0
        loss_seg_list, loss_point_list = [], []
        loss_seg_aux_list = []
        loss_adv_list, loss_adv_aux_list, loss_adv_ent_list, loss_adv_point_list = [], [], [], []
        loss_d_list, loss_d_aux_list, loss_d_ent_list, loss_d_point_list = [], [], [], []
        d1_acc_s, d1_acc_t, d_acc_s, d_acc_t, d_ent_acc_s, d_ent_acc_t, d_p_acc_s, d_p_acc_t = [], [], [], [], [], [], [], []
        for batch_content, batch_style in zip(self.content_loader, self.style_loader):
            self.opt_d.zero_grad()
            self.opt_d_aux.zero_grad()
            self.opt.zero_grad()
            self.opt_d_ent.zero_grad()
            self.opt_d_point.zero_grad()
            for param in self.d_main.parameters():
                param.requires_grad = False
            for param in self.d_aux.parameters():
                param.requires_grad = False
            for param in self.d_ent.parameters():
                param.requires_grad = False
            for param in self.d_point.parameters():
                param.requires_grad = False
            img_s, labels_s, vert_s = batch_content
            vert_s = vert_s.float().to(self.device)
            img_s, labels_s = img_s.to(self.device, non_blocking=self.args.pin_memory), \
                              labels_s.to(self.device, non_blocking=self.args.pin_memory)

            pred_s, pred_s_aux, pred_vert_s = self.segmentor(img_s)
            loss_seg = loss_calc(pred_s, labels_s, self.device, jaccard=True)
            """save the segmentation losses"""
            loss_seg_list.append(loss_seg.item())
            # if self.args.multilvl:
            loss_seg_aux = loss_calc(pred_s_aux, labels_s, self.device, jaccard=True)
            loss_seg_aux_list.append(loss_seg_aux.item())
            loss_seg += self.args.w_seg_aux * loss_seg_aux
            loss_seg.backward(retain_graph=True)

            loss_point = batch_NN_loss(pred_vert_s, vert_s)
            try:
                (self.args.wp * loss_point).backward(retain_graph=True)
                loss_point_list.append(loss_point.item())
            except Exception as e:
                print(f'error: {e}')
                print(f'pred size: {pred_vert_s.size()}, label size: {vert_s.size()}')
                print(f'pred device: {pred_vert_s.device}, label device: {vert_s.device}')
                print(f'pred dtype: {pred_vert_s.dtype}, label vert dtype: {vert_s.dtype}')
                dist = batch_pairwise_dist(pred_vert_s.detach(), vert_s.detach())
                dist = dist.min(dim=2).values
                print(f'dist size: {dist.size()}')
                dist = torch.sqrt(dist + 1e-10)
                dist1 = batch_pairwise_dist(vert_s.detach(), pred_vert_s.detach())
                dist1 = dist1.min(dim=2).values
                print(f'dist1 size: {dist1.size()}')
                dist1 = torch.sqrt(dist1 + 1e-10)
                print(torch.count_nonzero(dist == 0) + torch.count_nonzero(dist1 == 0))

            """calculate the adversarial losses"""
            """adaptseg"""
            img_t, labels_t, namet = batch_style
            img_t = img_t.to(self.device, non_blocking=self.args.pin_memory)
            pred_t, pred_t_aux, pred_vert_t = self.segmentor(img_t)
            pred_t_softmax = F.softmax(pred_t, dim=1)
            D_out = self.d_main(pred_t_softmax)
            loss_adv = F.binary_cross_entropy_with_logits(D_out, torch.FloatTensor(
                D_out.data.size()).fill_(
                source_domain_label).cuda())
            # if self.args.multilvl:
            pred_t_aux_softmax = F.softmax(pred_t_aux, dim=1)
            D_out1 = self.d_aux(pred_t_aux_softmax)
            loss_adv_aux = F.binary_cross_entropy_with_logits(D_out1, torch.FloatTensor(
                D_out1.data.size()).fill_(
                source_domain_label).cuda())
            loss_adv_aux_list.append(loss_adv_aux.item())
            """advent"""
            uncertainty_mapT = -1.0 * pred_t_softmax * torch.log(pred_t_softmax + smooth)
            D_out = self.d_ent(uncertainty_mapT)
            loss_adv_ent = F.binary_cross_entropy_with_logits(D_out, torch.FloatTensor(
                D_out.data.size()).fill_(
                source_domain_label).cuda())
            """pointnet"""
            D_out4 = self.d_point(pred_vert_t.transpose(2, 1))[0]
            loss_adv_point = F.binary_cross_entropy_with_logits(D_out4, torch.FloatTensor(
                D_out4.data.size()).fill_(
                source_domain_label).cuda())

            """save the adversarial losses"""
            loss_adv_list.append(loss_adv.item())
            loss_adv_aux_list.append(loss_adv_aux.item())
            loss_adv_ent_list.append(loss_adv_ent.item())
            loss_adv_point_list.append(loss_adv_point.item())
            (self.args.w_dis * loss_adv + self.args.w_d_ent * loss_adv_ent + self.args.w_d_point * loss_adv_point +
             self.args.w_dis_aux * loss_adv_aux).backward()

            for param in self.d_main.parameters():
                param.requires_grad = True
            for param in self.d_aux.parameters():
                param.requires_grad = True
            for param in self.d_ent.parameters():
                param.requires_grad = True
            for param in self.d_point.parameters():
                param.requires_grad = True

            pred_s = pred_s.detach()
            pred_s_aux = pred_s_aux.detach()
            pred_t_softmax = pred_t_softmax.detach()
            pred_s_softmax = F.softmax(pred_s, dim=1).detach()
            uncertainty_mapS = (-1.0 * pred_s_softmax * torch.log(pred_s_softmax + smooth)).detach()
            uncertainty_mapT = uncertainty_mapT.detach()
            pred_vert_s = pred_vert_s.detach()
            pred_vert_t = pred_vert_t.detach()

            """aux output discriminator"""
            # if self.args.multilvl:
            D_out_s = self.d_aux(F.softmax(pred_s_aux, dim=1))
            loss_D_s = F.binary_cross_entropy_with_logits(D_out_s, torch.FloatTensor(D_out_s.data.size()).fill_(
                source_domain_label).cuda()) / 2
            loss_D_s.backward()
            D_out_s = torch.sigmoid(D_out_s.detach()).cpu().numpy()
            D_out_s = np.where(D_out_s >= .5, 1, 0)
            d1_acc_s.append(np.mean(D_out_s))

            D_out_t_aux = self.d_aux(pred_t_aux_softmax.detach())
            loss_D_t1 = F.binary_cross_entropy_with_logits(D_out_t_aux,
                                                           torch.FloatTensor(D_out_t_aux.data.size()).fill_(
                                                               target_domain_label).cuda()) / 2
            loss_D_t1.backward()
            D_out_t_aux = torch.sigmoid(D_out_t_aux.detach()).cpu().numpy()
            D_out_t_aux = np.where(D_out_t_aux >= .5, 1, 0)
            d1_acc_t.append(1 - np.mean(D_out_t_aux))

            loss_d_aux_list.append((loss_D_s + loss_D_t1).item())
            """output discriminator"""
            D_out_s = self.d_main(pred_s_softmax)
            loss_D_s = F.binary_cross_entropy_with_logits(D_out_s, torch.FloatTensor(D_out_s.data.size()).fill_(
                source_domain_label).cuda()) / 2
            loss_D_s.backward()
            D_out_s = torch.sigmoid(D_out_s.detach()).cpu().numpy()
            D_out_s = np.where(D_out_s >= .5, 1, 0)
            d_acc_s.append(np.mean(D_out_s))

            D_out_t = self.d_main(pred_t_softmax)
            loss_D_t = F.binary_cross_entropy_with_logits(D_out_t, torch.FloatTensor(D_out_t.data.size()).fill_(
                target_domain_label).cuda()) / 2
            loss_D_t.backward()
            D_out_t = torch.sigmoid(D_out_t.detach()).cpu().numpy()
            D_out_t = np.where(D_out_t >= .5, 1, 0)
            d_acc_t.append(1 - np.mean(D_out_t))

            loss_d_list.append((loss_D_s + loss_D_t).item())
            """entropy discriminator"""
            D_out_s = self.d_ent(uncertainty_mapS)
            loss_D_s = F.binary_cross_entropy_with_logits(D_out_s, torch.FloatTensor(D_out_s.data.size()).fill_(
                source_domain_label).cuda()) / 2
            loss_D_s.backward()
            D_out_s = torch.sigmoid(D_out_s.detach()).cpu().numpy()
            D_out_s = np.where(D_out_s >= .5, 1, 0)
            d_ent_acc_s.append(np.mean(D_out_s))

            D_out_t = self.d_ent(uncertainty_mapT)
            loss_D_t = F.binary_cross_entropy_with_logits(D_out_t, torch.FloatTensor(D_out_t.data.size()).fill_(
                target_domain_label).cuda()) / 2
            loss_D_t.backward()
            D_out_t = torch.sigmoid(D_out_t.detach()).cpu().numpy()
            D_out_t = np.where(D_out_t >= .5, 1, 0)
            d_ent_acc_t.append(1 - np.mean(D_out_t))

            loss_d_ent_list.append((loss_D_s + loss_D_t).item())
            """point discriminator"""
            D_out_s = self.d_point(pred_vert_s.transpose(2, 1))[0]
            loss_D_s = F.binary_cross_entropy_with_logits(D_out_s, torch.FloatTensor(D_out_s.data.size()).fill_(
                source_domain_label).cuda()) / 2
            loss_D_s.backward()
            D_out_s = torch.sigmoid(D_out_s.detach()).cpu().numpy()
            D_out_s = np.where(D_out_s >= .5, 1, 0)
            d_p_acc_s.append(np.mean(D_out_s))

            D_out_t = self.d_point(pred_vert_t.transpose(2, 1))[0]
            loss_D_t = F.binary_cross_entropy_with_logits(D_out_t, torch.FloatTensor(D_out_t.data.size()).fill_(
                target_domain_label).cuda()) / 2
            loss_D_t.backward()
            D_out_t = torch.sigmoid(D_out_t.detach()).cpu().numpy()
            D_out_t = np.where(D_out_t >= .5, 1, 0)
            d_p_acc_t.append(1 - np.mean(D_out_t))

            loss_d_point_list.append((loss_D_s + loss_D_t).item())
            """update the discriminator optimizer"""
            self.opt_d_aux.step()
            self.opt_d.step()
            self.opt_d_ent.step()
            self.opt_d_point.step()
            """update the segmentor optimizer"""
            self.opt.step()

        resultls['seg_s'] = sum(loss_seg_list) / len(loss_seg_list)
        resultls['seg_s_aux'] = sum(loss_seg_aux_list) / len(loss_seg_aux_list)
        resultls['point'] = sum(loss_point_list) / len(loss_point_list)

        resultls['loss_adv'] = sum(loss_adv_list) / len(loss_adv_list)
        resultls['loss_adv_aux'] = sum(loss_adv_aux_list) / len(loss_adv_aux_list)
        resultls['loss_adv_ent'] = sum(loss_adv_ent_list) / len(loss_adv_ent_list)
        resultls['loss_adv_point'] = sum(loss_adv_point_list) / len(loss_adv_point_list)

        resultls['loss_dis'] = sum(loss_d_list) / len(loss_d_list)
        resultls['loss_dis1'] = sum(loss_d_aux_list) / len(loss_d_aux_list)
        resultls['loss_dis_ent'] = sum(loss_d_ent_list) / len(loss_d_ent_list)
        resultls['loss_dis_point'] = sum(loss_d_point_list) / len(loss_d_point_list)

        resultls['dis_acc_s'] = sum(d_acc_s) / len(d_acc_s)
        resultls['dis_acc_t'] = sum(d_acc_t) / len(d_acc_t)
        resultls['dis1_acc_s'] = sum(d1_acc_s) / len(d1_acc_s)
        resultls['dis1_acc_t'] = sum(d1_acc_t) / len(d1_acc_t)
        resultls['dis_ent_acc_s'] = sum(d_ent_acc_s) / len(d_ent_acc_s)
        resultls['dis_ent_acc_t'] = sum(d_ent_acc_t) / len(d_ent_acc_t)
        resultls['dis_point_acc_s'] = sum(d_p_acc_s) / len(d_p_acc_s)
        resultls['dis_point_acc_t'] = sum(d_p_acc_t) / len(d_p_acc_t)

        return resultls

    def train(self):
        """
        :return:
        """
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
            """record accuracies"""
            self.writer.add_scalars('Acc/Dis', {'source': train_results['dis_acc_s'],
                                                'target': train_results['dis_acc_t']}, epoch + 1)
            self.writer.add_scalars('Acc/Dis_aux', {'source': train_results['dis1_acc_s'],
                                                    'target': train_results['dis1_acc_t']}, epoch + 1)
            self.writer.add_scalars('Acc/D_ent', {'source': train_results['dis_ent_acc_s'],
                                                  'target': train_results['dis_ent_acc_t']}, epoch + 1)
            self.writer.add_scalars('Acc/D_point', {'source': train_results['dis_point_acc_s'],
                                                    'target': train_results['dis_point_acc_t']}, epoch + 1)
            # if self.args.multilvl:
            """record losses"""
            self.writer.add_scalars('Loss/Seg', {'main': train_results['seg_s'],
                                                 'aux': train_results['seg_s_aux']}, epoch + 1)
            self.writer.add_scalar('Loss/Point', train_results['point'], epoch + 1)
            self.writer.add_scalars('Loss/Adv', {'main': train_results['loss_adv'],
                                                 'aux': train_results['loss_adv_aux']}, epoch + 1)
            self.writer.add_scalars('Loss/Adv_ent', {'main': train_results['loss_adv_ent']}, epoch + 1)
            self.writer.add_scalars('Loss/Adv_point', {'main': train_results['loss_adv_point']}, epoch + 1)
            self.writer.add_scalars('Loss/Dis', {'dis': train_results['loss_dis'],
                                                 'dis1': train_results['loss_dis1']}, epoch + 1)
            self.writer.add_scalars('Loss/Dis_ent', {'dis_ent': train_results['loss_dis_ent']}, epoch + 1)
            self.writer.add_scalars('Loss/Dis_point', {'dis_point': train_results['loss_dis_point']}, epoch + 1)
            self.writer.add_scalars('LR', {'Seg': self.opt.param_groups[0]['lr'],
                                           'Dis': self.opt_d.param_groups[0]['lr']}, epoch + 1)

            print(
                f'Epoch = {epoch + 1:4d}/{self.args.epochs:4d}, loss_seg = {train_results["seg_s"]:.4f}, dc_valid = {lge_dice:.4f}')

            tobreak = self.stop_training(epoch, epoch_start, lge_dice)

            self.mcp_segmentor.step(monitor=lge_dice, model=self.segmentor, epoch=epoch + 1,
                                    optimizer=self.opt,
                                    tobreak=tobreak)
            self.modelcheckpoint_d.step(monitor=lge_dice, model=self.d_main, epoch=epoch + 1,
                                        optimizer=self.opt_d,
                                        tobreak=tobreak)
            # if self.args.multilvl:
            self.modelcheckpoint_d_aux.step(monitor=lge_dice, model=self.d_aux, epoch=epoch + 1,
                                            optimizer=self.opt_d_aux,
                                            tobreak=tobreak)
            self.modelcheckpoint_d_ent.step(monitor=lge_dice, model=self.d_ent, epoch=epoch + 1,
                                            optimizer=self.opt_d_ent,
                                            tobreak=tobreak)
            self.modelcheckpoint_d_point.step(monitor=lge_dice, model=self.d_point, epoch=epoch + 1,
                                              optimizer=self.opt_d_point,
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
