from datetime import datetime, timedelta
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch

import model.RAIN as net
from utils.utils_ import save_transferred_images_RAIN, crop_normalize
from utils.lr_adjust import adjust_learning_rate_custom
from utils import timer

from trainer.Trainer_baseline import Trainer_baseline


class Pretrainer_RAIN(Trainer_baseline):
    def __init__(self):
        super().__init__()
        self.save_dir = str(Path(self.args.style_dir).joinpath(self.apdx))
        if Path(self.save_dir).exists():
            now = datetime.now()
            self.save_dir = str(self.save_dir) + ".{}.{}".format(now.hour, now.minute)

    def add_additional_arguments(self):
        """weight for the losses"""
        self.parser.add_argument('-sw', help='style weight', type=float, default=5.0)
        self.parser.add_argument('-cw', help='content weight', type=float, default=5.0)
        self.parser.add_argument('-lw', help='latent weight', type=float, default=1.0)
        self.parser.add_argument('-rw', help='reconstruct weight', type=float, default=5.0)

        self.parser.add_argument('-restore', action='store_true')
        self.parser.add_argument('-task', type=str, default='pretrain_RAIN')

    @timer.timeit
    def get_arguments_apdx(self):
        assert self.args.task == 'pretrain_RAIN' or self.args.task == 'self_recon'
        super(Trainer_baseline, self).get_basic_arguments_apdx(name='Reco' if self.args.task == 'self_recon' else 'PreRAIN')
        if self.args.task == 'pretrain_RAIN':
            self.apdx += f".{self.src_modality}2{self.trgt_modality}.lr{self.args.lr}.sw{self.args.sw}." \
                         f"cw{self.args.cw}.lw{self.args.lw}.rw{self.args.rw}"
        if self.args.normalization == 'zscore':
            self.apdx += '.zscr'
        elif self.args.normalization == 'minmax':
            self.apdx += '.mnmx'
        else:
            raise NotImplementedError

    @timer.timeit
    def prepare_model(self):
        if self.args.restore:
            decoder_model_dir = 'weights/decoder.{}.pt'.format(self.apdx)
            fc_encoder_dir = 'weights/fc_encoder.{}.pt'.format(self.apdx)
            fc_decoder_dir = 'weights/fc_decoder.{}.pt'.format(self.apdx)
            self.start_epoch = torch.load(decoder_model_dir)['epoch']
        else:
            decoder_model_dir, fc_encoder_dir, fc_decoder_dir = None, None, None
        self.vgg, self.decoder, self.fc_encoder, self.fc_decoder = net.load_rain_models(encoder_weight=self.args.vgg,
                                                                                        decoder_weights=decoder_model_dir,
                                                                                        fc_encoder_weights=fc_encoder_dir,
                                                                                        fc_decoder_weights=fc_decoder_dir,
                                                                                        device=self.device)
        print('start epoch: {}'.format(self.start_epoch))
        self.network = net.Net(self.vgg, self.decoder, self.fc_encoder, self.fc_decoder, init=not self.args.restore)
        self.network.train()
        self.network.to(self.device)

    @timer.timeit
    def prepare_checkpoints(self, decoder_best_model_dir=None, decoder_model_dir=None, mode='min'):
        from utils.callbacks import ModelCheckPointCallback
        encoder_best_model_dir = 'weights/best_encoder.{}.pt'.format(self.apdx)
        encoder_model_dir = 'weights/encoder.{}.pt'.format(self.apdx)
        self.encoder_checkpoint = ModelCheckPointCallback(n_epochs=self.args.epochs, save_best=True,
                                                          mode=mode,
                                                          best_model_dir=encoder_best_model_dir,
                                                          save_last_model=True,
                                                          model_name=encoder_model_dir,
                                                          entire_model=False,
                                                          save_every_epochs=self.args.save_every_epochs)

        decoder_best_model_dir = 'weights/best_decoder.{}.pt'.format(
            self.apdx) if decoder_best_model_dir is None else decoder_best_model_dir
        decoder_model_dir = 'weights/decoder.{}.pt'.format(
            self.apdx) if decoder_model_dir is None else decoder_model_dir
        self.decoder_checkpoint = ModelCheckPointCallback(n_epochs=self.args.epochs, save_best=True,
                                                          mode=mode,
                                                          best_model_dir=decoder_best_model_dir,
                                                          save_last_model=True,
                                                          model_name=decoder_model_dir,
                                                          entire_model=False,
                                                          save_every_epochs=self.args.save_every_epochs)

        fc_encoder_best_dir = 'weights/best_fc_encoder.{}.pt'.format(self.apdx)
        fc_encoder_dir = 'weights/fc_encoder.{}.pt'.format(self.apdx)
        self.fc_encoder_checkpoint = ModelCheckPointCallback(n_epochs=self.args.epochs, save_best=True,
                                                             mode=mode,
                                                             best_model_dir=fc_encoder_best_dir,
                                                             save_last_model=True,
                                                             model_name=fc_encoder_dir,
                                                             entire_model=False,
                                                             save_every_epochs=self.args.save_every_epochs)

        fc_decoder_best_dir = 'weights/best_fc_decoder.{}.pt'.format(self.apdx)
        fc_decoder_dir = 'weights/fc_decoder.{}.pt'.format(self.apdx)
        self.fc_decoder_checkpoint = ModelCheckPointCallback(n_epochs=self.args.epochs, save_best=True,
                                                             mode=mode,
                                                             best_model_dir=fc_decoder_best_dir,
                                                             save_last_model=True,
                                                             model_name=fc_decoder_dir,
                                                             entire_model=False,
                                                             save_every_epochs=self.args.save_every_epochs)

    @timer.timeit
    def prepare_optimizers(self):
        self.vgg_optimizer = torch.optim.Adam(self.vgg.parameters(), lr=self.args.lr)
        print('vgg optimizer created')
        self.optimizer = torch.optim.Adam(self.network.decoder.parameters(), lr=self.args.lr)
        print('decoder optimizer created')
        self.optimizer_fc_encoder = torch.optim.Adam(self.network.fc_encoder.parameters(), lr=self.args.lr)
        print('fc encoder optimizer created')
        self.optimizer_fc_decoder = torch.optim.Adam(self.network.fc_decoder.parameters(), lr=self.args.lr)
        print('fc decoder optimizer created')

    def train_epoch(self, epoch):
        loss_c_list, loss_s_list, loss_l_list, loss_r_list = [], [], [], []
        to_save = True
        for (content_images, _, c_names), (style_images, _, s_names) in zip(self.content_loader, self.style_loader):
            if ((epoch + 1) % self.args.save_every_epochs == 0) and to_save:
                # print('transfer images and save')
                with torch.no_grad():
                    stylized, _ = self.network.style_transfer(content=content_images.cuda(), style=style_images.cuda())
                    stylized = torch.mean(stylized, dim=1)  # take the average of the 3 channels
                    stylized = torch.stack([stylized, stylized, stylized], dim=1)
                    stylized = crop_normalize(stylized, content_images, normalization=self.args.normalization)
                num_vis = min(8, stylized.size(0))
                self.writer.add_images('RAIN/Content', torch.clamp(content_images[:num_vis], 0, 1), epoch + 1)
                self.writer.add_images('RAIN/Style', torch.clamp(style_images[:num_vis], 0, 1), epoch + 1)
                self.writer.add_images('RAIN/Stylized', torch.clamp(stylized[:num_vis].detach().cpu(), 0, 1),
                                       epoch + 1)
                # print(f'content names: {c_names}')
                # print(f'style names: {s_names}')
                save_transferred_images_RAIN(stylized, c_names, s_names, epoch=epoch,
                                             idx_to_save=np.arange(min(8, len(c_names))),
                                             save_dir=self.save_dir, normalization=self.args.normalization)
                to_save = False
            for param in self.network.fc_encoder.parameters():
                param.requires_grad = True
            for param in self.network.fc_decoder.parameters():
                param.requires_grad = True
            loss_c, loss_s, loss_l, loss_r = self.network.forward(content_images.cuda(), style_images.cuda())
            # print('network forwarded')
            # collect losses
            loss_c_list.append(loss_c.item())
            loss_s_list.append(loss_s.item())
            loss_l_list.append(loss_l.item())
            loss_r_list.append(loss_r.item())

            loss_l = self.args.lw * loss_l
            loss_r = self.args.rw * loss_r
            loss_fc = loss_l + loss_r
            self.optimizer_fc_encoder.zero_grad()
            self.optimizer_fc_decoder.zero_grad()
            loss_fc.backward(retain_graph=True)
            # print('latent recons backwarded')
            for param in self.network.fc_encoder.parameters():
                param.requires_grad = False
            for param in self.network.fc_decoder.parameters():
                param.requires_grad = False

            loss_c = self.args.cw * loss_c
            loss_s = self.args.sw * loss_s
            self.optimizer.zero_grad()
            loss_de = loss_c + loss_s
            loss_de.backward()
            # print('content style backwarded')
            self.optimizer.step()
            self.optimizer_fc_encoder.step()
            self.optimizer_fc_decoder.step()
            # print('gradient updated')

        return sum(loss_c_list) / len(loss_c_list), sum(loss_s_list) / len(loss_s_list), \
               sum(loss_l_list) / len(loss_l_list), sum(loss_r_list) / len(loss_r_list)

    @timer.timeit
    def train(self):
        decoder_lr, fc_encoder, fc_decoder = [], [], []
        loss_c, loss_s, loss_l, loss_r, loss_combine = [], [], [], [], []
        monitor = 0
        for epoch in tqdm(range(self.start_epoch, self.args.epochs)):
            epoch_start = datetime.now()
            adjust_learning_rate_custom(self.optimizer, lr=self.args.lr, lr_decay=self.args.lr_decay, epoch=epoch)
            adjust_learning_rate_custom(self.optimizer_fc_encoder, lr=self.args.lr, lr_decay=self.args.lr_decay,
                                        epoch=epoch)
            adjust_learning_rate_custom(self.optimizer_fc_decoder, lr=self.args.lr, lr_decay=self.args.lr_decay,
                                        epoch=epoch)
            decoder_lr.append(self.optimizer.param_groups[0]['lr'])
            fc_encoder.append(self.optimizer_fc_encoder.param_groups[0]['lr'])
            fc_decoder.append(self.optimizer_fc_decoder.param_groups[0]['lr'])
            self.writer.add_scalars('Lr', {'Decoder': self.optimizer.param_groups[0]['lr'],
                                           'FC_encoder': self.optimizer_fc_encoder.param_groups[0]['lr'],
                                           'FC_decoder': self.optimizer_fc_decoder.param_groups[0]['lr']}, epoch + 1)

            mean_loss_c, mean_loss_s, mean_loss_l, mean_loss_r = self.train_epoch(epoch)

            loss_c.append(mean_loss_c)
            loss_s.append(mean_loss_s)
            loss_l.append(mean_loss_l)
            loss_r.append(mean_loss_r)
            loss_combine.append(self.args.lw * mean_loss_l + self.args.rw * mean_loss_r +
                                self.args.cw * mean_loss_c + self.args.sw * mean_loss_s)
            print(f"epoch: {epoch + 1:>4}, content {np.around(mean_loss_c, 3)}, style {np.around(mean_loss_s, 3)}, "
                  f"latent {np.around(mean_loss_l, 3)}, reconstruct {np.around(mean_loss_r, 3)}")
            self.writer.add_scalars('Loss', {'Content': mean_loss_c, 'Style': mean_loss_s, 'KL': mean_loss_l,
                                             'Reconstruct': mean_loss_r,
                                             'Combine': self.args.lw * mean_loss_l + self.args.rw * mean_loss_r +
                                                        self.args.cw * mean_loss_c + self.args.sw * mean_loss_s}, epoch + 1)

            if (datetime.now() - self.start_time).seconds > self.max_duration - self.max_epoch_time:
                epoch = self.args.epochs - 1

            monitor = round(loss_combine[-1], 3)
            self.decoder_checkpoint.step(monitor=monitor, optimizer=self.optimizer, model=self.network.decoder,
                                         epoch=epoch + 1)
            self.fc_encoder_checkpoint.step(monitor=monitor, optimizer=self.optimizer_fc_encoder,
                                            model=self.network.fc_encoder,
                                            epoch=epoch + 1)
            self.fc_decoder_checkpoint.step(monitor=monitor, optimizer=self.optimizer_fc_decoder,
                                            model=self.network.fc_decoder,
                                            epoch=epoch + 1)

            if self.check_time_elapsed(epoch, epoch_start):
                break

        self.writer.close()
        log_dir_new = '{}.loss{}'.format(self.log_dir, monitor)
        os.rename(self.log_dir, log_dir_new)

        print(f'best model at epoch {self.decoder_checkpoint.epoch}, loss {self.decoder_checkpoint.best_result}')

        if os.path.exists(self.save_dir):
            new_save_dir = self.save_dir + f'.loss{self.decoder_checkpoint.best_result}'
            os.rename(self.save_dir, new_save_dir)
            print(f'{self.save_dir} renamed to {new_save_dir}')

    @timer.timeit
    def train_selfrecon(self):
        """
        train the vgg and decoder with self-reconstruction task
        """
        lr = []
        loss_reco_list = []
        monitor = 0
        loss = torch.nn.MSELoss()
        self.vgg.train()
        self.decoder.train()
        for i in tqdm(range(self.start_epoch, self.args.epochs)):
            loss_recon = []
            epoch_start = datetime.now()
            adjust_learning_rate_custom(self.vgg_optimizer, lr=self.args.lr, lr_decay=self.args.lr_decay, epoch=i)
            adjust_learning_rate_custom(self.optimizer, lr=self.args.lr, lr_decay=self.args.lr_decay, epoch=i)


            for (content_images, _, c_names), (style_images, _, s_names) in zip(self.content_loader, self.style_loader):
                for param in self.network.fc_encoder.parameters():
                    param.requires_grad = True
                for param in self.network.fc_decoder.parameters():
                    param.requires_grad = True
                pred_s = self.decoder(self.vgg(content_images.to(self.device)))
                pred_t = self.decoder(self.vgg(style_images.to(self.device)))
                recon_loss = loss(pred_s, content_images.to(self.device)) + loss(pred_t, style_images.to(self.device))

                loss_recon.append(recon_loss.item())
                self.optimizer.zero_grad()
                recon_loss.backward()
                # print('content style backwarded')
                self.vgg_optimizer.step()
                self.optimizer.step()
                # print('gradient updated')
            loss_reco_list.append(sum(loss_recon) / len(loss_recon))
            lr.append(self.optimizer.param_groups[0]['lr'])
            print(f"epoch: {i + 1:>4}, recon_loss {np.around(loss_reco_list[-1], 3)}")

            if (datetime.now() - self.start_time).seconds > self.max_duration - self.max_epoch_time:
                i = self.args.epochs - 1

            monitor = loss_reco_list[-1]
            monitor = round(monitor, 3)
            self.encoder_checkpoint.step(monitor=monitor, optimizer=self.vgg_optimizer, model=self.vgg,
                                         epoch=i + 1)
            self.decoder_checkpoint.step(monitor=monitor, optimizer=self.optimizer, model=self.network.decoder,
                                         epoch=i + 1)

            if (datetime.now() - self.start_time).seconds > self.max_duration - self.max_epoch_time:
                print("training time elapsed: {}".format(datetime.now() - self.start_time))
                print("max_epoch_time: {}".format(timedelta(seconds=self.max_epoch_time)))
                break
            epoch_time_elapsed = datetime.now() - epoch_start
            print(f'Epoch {i + 1} time elapsed: {epoch_time_elapsed}')
            self.max_epoch_time = max(epoch_time_elapsed.seconds, self.max_epoch_time)

        print(f'write to summary writer. number of entries: {len(lr)}')
        for i in range(len(lr)):
            self.writer.add_scalars('Lr', lr[i], i + self.start_epoch + 1)
            self.writer.add_scalars('Loss', {'Reconstruct': loss_reco_list[i]}, i + self.start_epoch + 1)
        self.writer.close()
        log_dir_new = '{}.loss{}'.format(self.log_dir, monitor)
        os.rename(self.log_dir, log_dir_new)
