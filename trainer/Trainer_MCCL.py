from datetime import datetime
import os
import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn.functional as functional
from torch import autograd

from dataset.data_generator_mscmrseg import prepare_dataset
from dataset.data_generator_mmwhs import prepare_dataset as prepare_dataset_mmwhs
from dataset.data_generator_mmwhs_raw import prepare_dataset as prepare_dataset_mmwhs_raw
from utils.utils_ import save_transferred_images_RAIN, cal_centroid, crop_normalize
from utils import timer
import config
from utils.loss import loss_calc, ContrastiveLoss
from trainer.Trainer_RAIN import Trainer_RAIN


class Trainer_MCCL(Trainer_RAIN):
    def __init__(self):
        super().__init__()
        self.retain_graph = False
        self.args.num_workers = min(self.args.num_workers, self.args.bs)

    def add_additional_arguments(self):
        """
        :param parser:
        :return:
        """
        super(Trainer_MCCL, self).add_additional_arguments()
        # parser.add_argument('-fp16', action='store_true',
        #                     help='use float16 instead of float32, which will save about 50% memory')
        """rain configuration"""
        self.parser.add_argument('-rain', help='whether to use rain', action='store_true')
        """contrastive loss configuration"""
        self.parser.add_argument('-clda', help='whether to apply contrastive loss', action='store_true')
        self.parser.add_argument('-stdmin', help='whether to minimize the stddev of the features', action='store_true')
        self.parser.add_argument('-w_stdmin', help='The weight of the stdmin loss', type=float, default=.1)
        self.parser.add_argument('-clbg', help='whether to include background in contrastive loss', action='store_true')
        self.parser.add_argument('-phead', help='whether to include projection head', action='store_true')
        self.parser.add_argument('-seg_pseudo', help='whether to include pseudo segmentation loss', action='store_true')
        self.parser.add_argument('-tau', help='The temperature value for the contrastive loss', type=float, default=5.)
        # self.parser.add_argument('-clwn', help='Whether to use norm as the denominator in the contrastive loss',
        #                     action='store_true')
        self.parser.add_argument("-ctd_mmt", type=float, default=0.95, help='The momentum of the source centroid.')
        self.parser.add_argument('-inter_w', help='the weight for the inter contrastive loss', type=float,
                                 default=config.WEIGHT_INTER_LOSS)
        self.parser.add_argument('-intra', help='Whether to apply intra contrastive loss.', action='store_true')
        self.parser.add_argument('-intra_w', help='the weight for the intra contrastive loss.', type=float,
                                 default=config.WEIGHT_INTRA_LOSS)
        # self.parser.add_argument('-inst', action='store_true')
        # self.parser.add_argument('-inst_w', type=float, default=config.WEIGHT_INST_LOSS)
        self.parser.add_argument('-CNR', help='whether to apply l2 regularization to the centroids (CNR).',
                                 action='store_true')
        # self.parser.add_argument('-mse0', help='whether to set label of mse loss as 0.', action='store_true')
        self.parser.add_argument('-CNR_w', help='The weight for the l2 regularization.', type=float,
                                 default=config.WEIGHT_MSE)
        self.parser.add_argument('-thd', help='The threshold for calculating the centroids. '
                                              '1 represents the adaptive threshold, (0, 1) refers to a defined threshold',
                                 type=float, default=None)
        self.parser.add_argument('-low_thd', help='The lowest threshold when using curriculum learning',
                                 type=float, default=0)
        self.parser.add_argument('-high_thd', help='The highest threshold when using curriculum learning',
                                 type=float, default=0.99)
        self.parser.add_argument('-thd_w', help='The weight for the adaptive threshold.', type=float,
                                 default=config.WEIGHT_THD)
        self.parser.add_argument('-part', help='number of partitions to split decoder_ft', type=int, default=1)
        self.parser.add_argument('-wtd_ave',
                                 help='Whether to calculated the weighted average of the features or the "global" '
                                      'average of the features as the centroids.', action='store_true')
        self.parser.add_argument('-contrast_split', help='Whether to split the source and target feature in the '
                                                         'nominator of the contrastive loss', action='store_true')
        """epsilon configuration"""
        self.parser.add_argument('-update_eps', help='Whether to update eps (sampling)', action='store_true')
        # self.parser.add_argument('-eps_cts', help='whether to apply contrastive loss to the update of epsilon.',
        #                     action='store_true')
        # self.parser.add_argument('-eps_cts_w', help='The weight for the contrastive loss in the updates of epsilon.',
        #                     type=float, default=config.WEIGHT_EPS_CTS)
        """evaluation configuration"""
        # self.parser.add_argument('-evl_s', help='Whether to evaluate the model in source domain.', action='store_true')
        # self.parser.add_argument('-eval', help='Whether to evaluate the best model at the end of training.',
        #                     action='store_true')
        """recording related experiment data"""
        self.parser.add_argument('-grad', help='Whether to record gradient of the features and the centroids',
                                 action='store_true')

    @timer.timeit
    def get_arguments_apdx(self):
        super(Trainer_MCCL, self).get_basic_arguments_apdx(name='MCCL')
        self.apdx += f'.cw{self.args.consist_w}.bs{self.args.bs}'
        if self.args.mulstyle:
            self.apdx += '.ms'
        if self.args.mulstyle2:
            self.apdx += '.ms2'
        if self.args.ctd_mmt != 0.95:
            self.apdx += '.mmt{}'.format(self.args.ctd_mmt)
        if self.args.rain is False:
            self.apdx += '.norain'
        else:
            if self.args.update_eps:
                self.apdx += '.eps{}.LSeg'.format(self.args.eps_iters) + '.lrs{}'.format(self.args.lr_eps)
                # if self.args.eps_cts:
                #     self.apdx += '.epcts.w{}'.format(self.args.eps_cts_w)
            if self.args.warmup_epochs > 0:
                self.apdx += '.wup{}'.format(self.args.warmup_epochs)
            if self.args.pat_id != -1:
                self.apdx += f".pat_{self.args.pat_id}_lge"
            elif self.args.slice_id != -1:
                self.apdx += f"{self.args.slice_id}"
        if self.args.stdmin:
            self.apdx += '.w{}.stdmin'.format(self.args.w_stdmin)
        if self.args.clda:
            if self.args.phead:
                self.apdx += '.ph'
            if self.args.seg_pseudo:
                self.apdx += '.segPdo'
            if self.args.thd is not None:
                self.apdx += '.thd{}.{}'.format(self.args.thd, self.args.thd_w)
                if self.args.thd == -2:
                    if self.args.low_thd is not None:
                        self.apdx += '.lth{}'.format(self.args.low_thd)
                    if self.args.high_thd is not None:
                        self.apdx += '.hth{}'.format(self.args.high_thd)
            self.apdx += '.clda.itew{}.t{}'.format(self.args.inter_w, self.args.tau)
            if self.args.intra:
                self.apdx += '.itrw{}'.format(self.args.intra_w)
            # if self.args.inst:
            #     self.apdx += '.instw{}'.format(self.args.inst_w)
            if self.args.wtd_ave:
                self.apdx += '.w_ave'
            if self.args.clbg:
                self.apdx += '.bg'
            self.apdx += '.p{}'.format(self.args.part)
        if self.args.CNR:
            self.apdx += '.CNR.w{}'.format(self.args.CNR_w)
        self.apdx += '.{}'.format(self.args.optim)
        if self.args.normalization == 'zscore':
            self.apdx += '.zscr'

    def prepare_losses(self):
        super(Trainer_MCCL, self).prepare_losses()
        self.contrastive_loss = ContrastiveLoss(tau=self.args.tau)

    @timer.timeit
    def prepare_dataloader(self):
        if self.dataset == 'mscmrseg':
            self.scratch, self.scratch_raw, self.content_loader, self.style_loader = prepare_dataset(self.args,
                                                                                                     aug_counter=True)
        elif self.dataset == 'mmwhs':
            print('importing raw data...')
            if self.args.raw:
                from pathlib import Path
                self.args.data_dir = str(Path(self.args.data_dir).parent.joinpath('CT_MR_2D_Dataset_DA-master'))
                self.scratch, self.scratch_raw, self.content_loader, self.style_loader = prepare_dataset_mmwhs_raw(
                    self.args,
                    aug_counter=True)
            else:
                self.scratch, self.scratch_raw, self.content_loader, self.style_loader = prepare_dataset_mmwhs(
                    self.args,
                    aug_counter=True)
        else:
            raise NotImplementedError

    def _log_images(self, visual_dict, step):
        def process_image(img):
            # img: [B, C, H, W]
            if img.dim() == 3:
                img = img.unsqueeze(1)
            # Normalize to 0-1 for visualization
            min_val = img.min()
            max_val = img.max()
            if max_val > min_val:
                img = (img - min_val) / (max_val - min_val)
            return img

        def process_mask(mask):
            # mask: [B, H, W]
            mask = mask.float()
            mask = mask.unsqueeze(1)  # [B, 1, H, W]
            mask = mask / (self.args.num_classes - 1 if self.args.num_classes > 1 else 1)
            return mask

        img_s = visual_dict.get('img_s')
        num_samples = min(4, img_s.size(0))

        def add_to_list(key, process_func):
            if key in visual_dict:
                data = visual_dict[key][:num_samples]
                data = process_func(data)
                if data.size(1) == 1:
                    data = data.repeat(1, 3, 1, 1)
                return data
            return None

        img_s_proc = add_to_list('img_s', process_image)
        labels_s_proc = add_to_list('labels_s', process_mask)
        pred_seg_s_proc = add_to_list('pred_seg_s', process_mask)
        img_t_proc = add_to_list('img_t', process_image)
        pred_t_proc = add_to_list('pred_t', process_mask)
        img_style_proc = add_to_list('img_style', process_image)
        pred_seg_style_proc = add_to_list('pred_seg_style', process_mask)

        if img_s_proc is not None:
            self.writer.add_images('Images/Source', img_s_proc, step)
        if labels_s_proc is not None:
            self.writer.add_images('Images/Source_Label', labels_s_proc, step)
        if pred_seg_s_proc is not None:
            self.writer.add_images('Images/Source_Pred', pred_seg_s_proc, step)

        if img_t_proc is not None:
            self.writer.add_images('Images/Target', img_t_proc, step)
        if pred_t_proc is not None:
            self.writer.add_images('Images/Target_Pred', pred_t_proc, step)

        if img_style_proc is not None:
            self.writer.add_images('Images/Style', img_style_proc, step)
        if pred_seg_style_proc is not None:
            self.writer.add_images('Images/Style_Pred', pred_seg_style_proc, step)

    def train_epoch(self, epoch):
        print(f'start to train epoch: {epoch}')
        self.segmentor.train()
        results = {}
        to_save_pic = True
        loss_seg_style_list, loss_seg_s_list, loss_pseudo_list = [], [], []
        loss_consistent_list = []
        ratio_t_list, ratio_t_aug_list = [], []
        stddev_t_list = []
        CNR_loss_list = []
        inter_cons_loss_list, intra_cons_loss_list = [], []
        grads_centroid, grads_ft = [], []
        """initialize the source centroid"""
        centroid_s = None
        visual_dict = {}
        for i_batch, (batch_content, batch_style) in enumerate(zip(self.content_loader, self.style_loader)):
            self.opt.zero_grad()
            sampling = None
            img_s, labels_s, names = batch_content
            img_s, labels_s = img_s.to(self.device, non_blocking=self.args.pin_memory), labels_s.to(self.device,
                                                                                                    non_blocking=self.args.pin_memory)
            img_t, img_t_aug, namet = batch_style
            img_t, img_t_aug = img_t.to(self.device, non_blocking=self.args.pin_memory), img_t_aug.to(self.device,
                                                                                                      non_blocking=self.args.pin_memory)
            eps_update_times = 1 if (
                    epoch < self.args.warmup_epochs or (self.args.update_eps is False) or (self.args.rain is False)) \
                else self.args.eps_iters
            for eps_iter in range(eps_update_times):
                s_size = img_s.size()[0]
                t_size = img_t.size()[0]
                loss_seg = 0
                if self.args.rain:
                    if self.args.mulstyle2:
                        rain_content, rain_style = img_s, img_t[0:1]
                    elif self.args.mulstyle:
                        rain_content, rain_style = img_s, img_t
                    else:
                        rain_content, rain_style = img_s[0:1], img_t[0:1]
                    img_style, sampling = self.network.style_transfer(rain_content,
                                                                      rain_style,
                                                                      sampling)
                    img_style = torch.mean(img_style, dim=1)
                    img_style = torch.stack([img_style, img_style, img_style], dim=1)
                    img_style = crop_normalize(img_style, img_s, normalization=self.args.normalization)
                    if to_save_pic and (epoch + 1 - self.start_epoch) % self.args.save_every_epochs == 0:
                        save_transferred_images_RAIN(img_style, names, namet, epoch=epoch, iter=eps_iter,
                                                     idx_to_save=np.arange(min(8, len(img_style))),
                                                     save_dir=self.save_dir,
                                                     stage='warmup' if eps_update_times == 1 else 'RAIN',
                                                     normalization=self.args.normalization)
                    """stop saving the images once the first batch has been saved for eps_update_times"""
                    to_save_pic = False if (eps_iter + 1) == eps_update_times else True
                    pred, bottleneck, dcdr_ft = self.segmentor(torch.cat([img_style, img_s, img_t, img_t_aug], dim=0))
                    style_size = img_style.size()[0]
                    pred_seg_size = style_size + s_size
                    pred_seg, pred_t = pred[:pred_seg_size], pred[pred_seg_size:]
                    """***********************The RAIN Part***********************"""
                    """calculate the consistency loss"""
                    btnk_seg = bottleneck[:pred_seg_size]
                    bottle_s_style = btnk_seg[: style_size]
                    bottle_s = btnk_seg[style_size: 2 * style_size]
                    loss_consist = self.mse_loss(bottle_s - bottle_s_style,
                                                 torch.zeros(bottle_s_style.size()).to(self.device))
                    loss_consistent_list.append(loss_consist.item())
                    loss_seg_style = loss_calc(pred_seg[:style_size], labels_s[:style_size], self.device, jaccard=True)
                    loss_seg_style_list.append(loss_seg_style.item())
                    loss_seg += loss_seg_style
                    """check whether need to retain graph"""
                    if self.args.update_eps:
                        self.retain_graph = (epoch >= self.args.warmup_epochs)
                        if self.retain_graph:
                            sampling.require_grad = True
                            sampling.retain_grad()
                            samp_loss = loss_seg
                            # if self.args.clda and self.args.eps_cts:
                            #     samp_loss = samp_loss - self.args.eps_cts_w * contrastive_loss_whole
                            samp_loss.backward(retain_graph=self.retain_graph)
                            grad_data = sampling.grad.data
                            self.opt.zero_grad()
                    loss_seg += self.args.consist_w * loss_consist
                else:
                    pred, bottleneck, dcdr_ft = self.segmentor(torch.cat([img_s, img_t, img_t_aug], dim=0))
                    pred_seg_size = s_size
                    style_size = 0
                    pred_seg, pred_t = pred[:s_size], pred[s_size:]

                """calculate the segmentation loss"""
                loss_seg_s = loss_calc(pred_seg[style_size:], labels_s, self.device, jaccard=True)
                loss_seg_s_list.append(loss_seg_s.item())
                loss_seg += loss_seg_s

                if i_batch == 0 and eps_iter == 0:
                    visual_dict['img_s'] = img_s.detach().cpu()
                    visual_dict['img_t'] = img_t.detach().cpu()
                    visual_dict['labels_s'] = labels_s.detach().cpu()
                    visual_dict['pred_seg_s'] = torch.argmax(pred_seg[style_size:], dim=1).detach().cpu()
                    visual_dict['pred_t'] = torch.argmax(pred_t, dim=1).detach().cpu()
                    if self.args.rain:
                        visual_dict['img_style'] = img_style.detach().cpu()
                        visual_dict['pred_seg_style'] = torch.argmax(pred_seg[:style_size], dim=1).detach().cpu()

                """***********************The Contrastive Learning Part***********************"""
                """get target pseudo label"""
                pseudo_label_t = functional.softmax(input=pred_t, dim=1)
                pseudo_label_t, pseudo_label_t_aug = pseudo_label_t[:t_size], pseudo_label_t[t_size:]
                if self.args.seg_pseudo:
                    calibrated_pseudo = pseudo_label_t * self.args.num_classes / torch.e
                    loss_seg_pseudo = -calibrated_pseudo.detach() * torch.log(calibrated_pseudo)
                    pseudo_mask = pseudo_label_t.max(dim=1, keepdims=True).values > self.args.thd
                    loss_seg_pseudo = loss_seg_pseudo * pseudo_mask
                    loss_seg_pseudo = loss_seg_pseudo.mean()
                    loss_pseudo_list.append(loss_seg_pseudo)
                    loss_seg += .5 * loss_seg_pseudo
                """get the respective decoder features"""
                if self.args.rain:
                    dcdr_ft_style, dcdr_ft_s = dcdr_ft[: style_size], dcdr_ft[style_size: pred_seg_size]
                else:
                    dcdr_ft_s = dcdr_ft[: s_size]
                dcdr_ft_t, dcdr_ft_t_aug = dcdr_ft[pred_seg_size: -t_size], dcdr_ft[-t_size:]
                """calculate the respective centroid (prototypes)"""
                centroid_s, _, _ = cal_centroid(decoder_ft=dcdr_ft_s, label=labels_s.cuda(), momentum=self.args.ctd_mmt,
                                                previous_centroid=centroid_s, weighted_ave=self.args.wtd_ave)
                centroid_s = centroid_s.detach()
                # centroid_s_style, _ = cal_centroid(decoder_ft=dcdr_ft_style, label=labels_s.cuda(),
                #                                    weighted_ave=self.args.wtd_ave)
                # centroid_t (4, 32) or (P, 4, 32)
                centroid_t, ratio_t, stddev_t = cal_centroid(decoder_ft=dcdr_ft_t, label=pseudo_label_t,
                                                             pseudo_label=True,
                                                             partition=self.args.part, threshold=self.args.thd,
                                                             thd_w=self.args.thd_w, low_thd=self.args.low_thd,
                                                             high_thd=self.args.high_thd,
                                                             weighted_ave=self.args.wtd_ave,
                                                             epoch=epoch, max_epoch=1000, stdmin=self.args.stdmin)
                centroid_t_aug, ratio_t_aug, _ = cal_centroid(decoder_ft=dcdr_ft_t_aug,
                                                                         label=pseudo_label_t_aug,
                                                                         pseudo_label=True, threshold=self.args.thd,
                                                                         thd_w=self.args.thd_w,
                                                                         low_thd=self.args.low_thd,
                                                                         high_thd=self.args.high_thd,
                                                                         weighted_ave=self.args.wtd_ave,
                                                                         epoch=epoch, max_epoch=1000)
                if type(centroid_t_aug) is list:
                    centroid_t_aug = centroid_t_aug[0]

                ratio_t_list.append(ratio_t)
                ratio_t_aug_list.append(ratio_t_aug)
                stddev_t_list.append(np.array([stddev.detach().cpu().numpy() for stddev in stddev_t]))

                """calculate the centroid norm regularizer (CNR)"""
                centroid_s_norm = torch.norm(centroid_s, p=2, dim=1)
                # centroid_t_aug_norm = torch.norm(centroid_t_aug, p=2, dim=1)
                # CNR_loss = self.mse_loss(centroid_t_aug_norm, centroid_s_norm.cuda()) if self.args.CNR else 0
                CNR_loss = 0
                for cent_t in centroid_t:
                    centroid_t_norm = torch.norm(cent_t, p=2, dim=1)
                    # if torch.max(centroid_t_norm) > 1e2:
                    #     print('centroid_t norm > 1e2. {}'.format(centroid_t_norm.detach().cpu().numpy()))
                    # else:
                    CNR_loss = CNR_loss + self.mse_loss(centroid_t_norm,
                                                        centroid_s_norm.to(self.device)) / self.args.part
                CNR_loss_list.append(CNR_loss.item())

                """calculate (inter / intra)contrastive loss"""
                inter_cons_loss, intra_cons_loss = 0, 0
                # inst_cons_loss = 0
                for cent_t in centroid_t:
                    inter_cons_loss = inter_cons_loss + self.contrastive_loss.forward(centroid_s=centroid_s,
                                                                                      centroid_t=cent_t,
                                                                                      split=self.args.contrast_split) / self.args.part
                    intra_cons_loss = intra_cons_loss + self.contrastive_loss.forward(centroid_s=cent_t,
                                                                                      centroid_t=centroid_t_aug,
                                                                                      split=self.args.contrast_split) / self.args.part
                inter_cons_loss_list.append(inter_cons_loss.item())
                intra_cons_loss_list.append(intra_cons_loss.item())
                # inst_loss_list.append(inst_cons_loss.item())
                contrastive_loss_whole = self.args.inter_w * inter_cons_loss
                if self.args.intra:
                    contrastive_loss_whole = contrastive_loss_whole + self.args.intra_w * intra_cons_loss

                """save the gradients"""
                # if self.args.grad:
                #     grads = torch.mean(torch.abs(autograd.grad(inter_cons_loss, centroid_t, retain_graph=True)[0]))
                #     grads_centroid.append(grads.detach().cpu().numpy())
                #     grads = torch.mean(torch.abs(autograd.grad(inter_cons_loss, dcdr_ft_t, retain_graph=True)[0]))
                #     grads_ft.append(grads.detach().cpu().numpy())
                if epoch >= self.args.warmup_epochs:
                    if self.args.clda:
                        # self.segmentor.decoder.decoder2_1[-3].weight
                        loss_seg = loss_seg + contrastive_loss_whole
                    if self.args.CNR:
                        loss_seg = loss_seg + self.args.CNR_w * CNR_loss
                    if self.args.stdmin:
                        loss_seg += self.args.w_stdmin * sum(stddev_t)
                loss_seg.backward()
                if self.args.update_eps and self.args.rain and self.retain_graph:
                    sampling = sampling + (self.args.lr_eps / samp_loss.item()) * grad_data
                    sampling = Variable(sampling.detach(), requires_grad=True)
                self.opt.step()

        """segmentation losses"""
        if self.args.rain:
            results['seg_style'] = sum(loss_seg_style_list) / len(loss_seg_style_list)
        results['seg_s'] = sum(loss_seg_s_list) / len(loss_seg_s_list)

        """consistency loss for content consistency"""
        if self.args.rain:
            results['loss_consist'] = sum(loss_consistent_list) / len(loss_consistent_list)

        """contrastive learning related losses"""
        results['inter_c_loss'] = sum(inter_cons_loss_list) / len(inter_cons_loss_list)
        results['intra_c_loss'] = sum(intra_cons_loss_list) / len(intra_cons_loss_list)
        results['CNR'] = sum(CNR_loss_list) / len(CNR_loss_list)

        """save the gradients"""
        # if self.args.grad:
        #     results['grads_centroid'] = sum(grads_centroid) / len(grads_centroid)
        #     results['grads_ft'] = sum(grads_ft) / len(grads_ft)
        """save the ratio of the number of features larger than the thd"""
        if self.args.thd:
            results['ratio_t'] = sum(ratio_t_list) / len(ratio_t_list)
            results['ratio_t_aug'] = sum(ratio_t_aug_list) / len(ratio_t_aug_list)
        if self.args.stdmin:
            results['stddev'] = np.array(stddev_t_list).mean(0)
        if self.args.seg_pseudo:
            results['loss_pseudo'] = sum(loss_pseudo_list) / len(loss_pseudo_list)

        results['visual_dict'] = visual_dict
        return results

    @timer.timeit
    def train(self):
        """
        :return:
        """

        """mkdir for the stylized images"""
        if not os.path.exists(self.args.style_dir):
            os.makedirs(self.args.style_dir)

        epoch = self.start_epoch
        for epoch in tqdm(range(self.start_epoch, self.args.epochs)):
            """adjust learning rate"""
            self.adjust_lr(epoch)
            epoch_start = datetime.now()

            train_results = self.train_epoch(epoch)

            message = f'Epoch = {epoch + 1:4d}/{self.args.epochs:4d}, loss_seg_s = {train_results["seg_s"]:.3f}'
            if self.args.rain:
                message += f', loss_seg_style = {train_results["seg_style"]:.3f}, ' \
                           f'loss_consist = {train_results["loss_consist"]:.3f}'
            print(message)
            if self.args.clda:
                print(f'Inter Contrastive loss = {train_results["inter_c_loss"]:.3f}, '
                      f'Intra Contrastive loss = {train_results["intra_c_loss"]:.3f}')
            results = self.eval(modality='target', phase='valid')
            if type(results) == tuple or type(results) == list:
                results = results[0]
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
            if self.args.rain:
                self.writer.add_scalars('Loss_seg', {'source': train_results['seg_s'],
                                                     'stylized': train_results['seg_style']}, epoch + 1)
                self.writer.add_scalar('Loss_consist', train_results['loss_consist'], epoch + 1)
            else:
                self.writer.add_scalar('Loss_seg', train_results['seg_s'], epoch + 1)
            if self.args.seg_pseudo:
                self.writer.add_scalars('Loss_seg', {'Pseudo': train_results['loss_pseudo']}, epoch + 1)
            self.writer.add_scalars('Loss_contrast', {'Inter': train_results['inter_c_loss'],
                                                      'Intra': train_results['intra_c_loss']}, epoch + 1)
            self.writer.add_scalar('CNR', train_results['CNR'], epoch + 1)
            self.writer.add_scalar('lr', self.opt.param_groups[0]['lr'], epoch + 1)
            # if self.args.grad:
            #     self.writer.add_scalar('Grad/Centroid', train_results['grads_centroid'], epoch + 1)
            #     self.writer.add_scalar('Grad/feature', train_results['grads_ft'], epoch + 1)
            if self.args.thd:
                self.writer.add_scalar('Ratio/target', train_results['ratio_t'], epoch + 1)
                self.writer.add_scalar('Ratio/target_aug', train_results['ratio_t_aug'], epoch + 1)
            if self.args.stdmin:
                for i in range(len(train_results['stddev'])):
                    self.writer.add_scalar('StdDev/c{}'.format(i + 1), train_results['stddev'][i], epoch + 1)

            if 'visual_dict' in train_results:
                self._log_images(train_results['visual_dict'], epoch + 1)

            tobreak = self.stop_training(epoch, epoch_start, lge_dice)

            self.mcp_segmentor.step(monitor=lge_dice, model=self.segmentor, epoch=epoch + 1,
                                    optimizer=self.opt,
                                    tobreak=tobreak)
            if tobreak:
                break

        self.writer.close()
        best_epoch = self.mcp_segmentor.epoch
        best_score = self.mcp_segmentor.best_result
        log_dir_new = 'runs/{}.e{}.Scr{}'.format(self.apdx, best_epoch,
                                                 np.around(best_score, 3))
        os.rename(self.log_dir, log_dir_new)
        # load the weights with the bext validation score and do the evaluation
        print(f'best model name base: {self.mcp_segmentor.best_model_name_base}, best epoch: {best_epoch}')
        print("the weight of the best unet model: {}".format(self.mcp_segmentor.best_model_save_dir))
        try:
            self.segmentor.load_state_dict(torch.load(self.mcp_segmentor.best_model_save_dir)['model_state_dict'])
            print("segmentor load from state dict")
        except:
            self.segmentor.load_state_dict(torch.load(self.mcp_segmentor.best_model_save_dir))
        print("model loaded")

        self.eval(modality='target', phase='test')
        if (epoch - self.start_epoch >= self.args.save_every_epochs) and self.args.rain:
            os.rename(self.save_dir, f'{self.save_dir}.e{best_epoch}.Scr{best_score}')
        return
