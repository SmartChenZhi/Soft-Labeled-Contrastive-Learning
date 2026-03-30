import os
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.loss import loss_calc, dice_loss, loss_entropy
from trainer.Trainer_baseline import Trainer_baseline


class Trainer_Causal(Trainer_baseline):
    def __init__(self):
        super().__init__()
        self.domain_criterion = nn.CrossEntropyLoss().to(self.device)

    def add_additional_arguments(self):
        super(Trainer_Causal, self).add_additional_arguments()
        self.parser.add_argument("-w_tde_sup", type=float, default=1.0)
        self.parser.add_argument("-w_ent", type=float, default=1e-3)
        self.parser.add_argument("-w_cons", type=float, default=0.1)
        self.parser.add_argument("-w_dom", type=float, default=0.1)
        self.parser.add_argument("-cons_thres", type=float, default=0.7)
        self.parser.add_argument("-xbar_momentum", type=float, default=0.2)
        self.parser.add_argument("-z_dim", type=int, default=32)
        self.parser.add_argument("-dom_emb_dim", type=int, default=16)
        self.parser.add_argument("-grl_lambda", type=float, default=1.0)

    def get_arguments_apdx(self):
        super(Trainer_Causal, self).get_basic_arguments_apdx(name="CausalTDE")
        self.apdx += f".bs{self.args.bs}"
        self.apdx += f".wtdesup{self.args.w_tde_sup}"
        self.apdx += f".went{self.args.w_ent}.wcons{self.args.w_cons}.wdom{self.args.w_dom}"
        self.apdx += f".cth{self.args.cons_thres}.xbm{self.args.xbar_momentum}"
        self.apdx += f".z{self.args.z_dim}.demb{self.args.dom_emb_dim}.grl{self.args.grl_lambda}"

    def prepare_model(self):
        if self.args.backbone != "drunet":
            raise ValueError("Trainer_Causal only supports -backbone drunet.")

        from model.CausalDRUNet import CausalSegmentationModel

        self.segmentor = CausalSegmentationModel(
            filters=self.args.filters,
            in_channels=3,
            n_block=self.args.nb,
            bottleneck_depth=self.args.bd,
            n_class=self.args.num_classes,
            z_dim=self.args.z_dim,
            dom_emb_dim=self.args.dom_emb_dim,
            grl_lambda=self.args.grl_lambda,
        )
        if self.args.restore_from:
            checkpoint = torch.load(self.args.restore_from)
            model_state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            self.segmentor.load_state_dict(model_state, strict=False)
            if "epoch" in checkpoint:
                self.start_epoch = checkpoint["epoch"]
        self.segmentor.train()
        self.segmentor.to(self.device)

    def _dice_ce_loss(self, logits, labels):
        return loss_calc(logits, labels, self.device, False) + dice_loss(logits, labels)

    def _consistency_loss(self, probs_fact, probs_tde):
        confidence = probs_tde.max(dim=1, keepdim=True).values > self.args.cons_thres
        if confidence.any():
            confidence = confidence.expand_as(probs_tde).float()
            return (torch.abs(probs_fact - probs_tde) * confidence).sum() / (confidence.sum() + 1e-6)
        return torch.zeros((), device=self.device)

    def train_epoch(self, epoch):
        print(f"start to train epoch: {epoch}")
        self.segmentor.train()

        results = {}
        loss_total_list = []
        loss_sup_list = []
        loss_sup_fact_list = []
        loss_sup_tde_list = []
        loss_ent_list = []
        loss_cons_list = []
        loss_dom_list = []

        for batch_content, batch_style in zip(self.content_loader, self.style_loader):
            self.opt.zero_grad()

            img_s, labels_s, _ = batch_content
            img_t, _, _ = batch_style

            img_s = img_s.to(self.device, non_blocking=self.args.pin_memory)
            labels_s = labels_s.to(self.device, non_blocking=self.args.pin_memory)
            img_t = img_t.to(self.device, non_blocking=self.args.pin_memory)

            if labels_s.dim() == 4 and labels_s.size(1) == 1:
                labels_s = labels_s.squeeze(1)

            b_s = img_s.size(0)
            b_t = img_t.size(0)
            domain_s = torch.zeros(b_s, dtype=torch.long, device=self.device)
            domain_t = torch.ones(b_t, dtype=torch.long, device=self.device)

            feat_x_s = self.segmentor.extract_x_feature(img_s)
            with torch.no_grad():
                x_batch_mean = feat_x_s.detach().mean(dim=(0, 2, 3))
                self.segmentor.update_x_bar(x_batch_mean, momentum=self.args.xbar_momentum)

            logits_tde_s, logits_fact_s, _, _, dom_logits_s = self.segmentor(
                img_s,
                domain_labels=domain_s,
                grl_lambda=self.args.grl_lambda,
                feat_x=feat_x_s,
            )
            logits_tde_t, logits_fact_t, _, _, dom_logits_t = self.segmentor(
                img_t,
                domain_labels=domain_t,
                grl_lambda=self.args.grl_lambda,
            )

            loss_sup_fact = self._dice_ce_loss(logits_fact_s, labels_s)
            loss_sup_tde = self._dice_ce_loss(logits_tde_s, labels_s)
            loss_sup = loss_sup_fact + self.args.w_tde_sup * loss_sup_tde

            probs_tde_t = F.softmax(logits_tde_t, dim=1)
            probs_fact_t = F.softmax(logits_fact_t, dim=1)
            loss_ent = loss_entropy(probs_tde_t, device=self.device, smooth=1e-10, mode="mean")
            loss_cons = self._consistency_loss(probs_fact_t, probs_tde_t)

            loss_dom_s = self.domain_criterion(dom_logits_s, domain_s)
            loss_dom_t = self.domain_criterion(dom_logits_t, domain_t)
            loss_dom = 0.5 * (loss_dom_s + loss_dom_t)

            loss_total = (
                loss_sup
                + self.args.w_ent * loss_ent
                + self.args.w_cons * loss_cons
                + self.args.w_dom * loss_dom
            )
            loss_total.backward()
            self.opt.step()

            loss_total_list.append(loss_total.item())
            loss_sup_list.append(loss_sup.item())
            loss_sup_fact_list.append(loss_sup_fact.item())
            loss_sup_tde_list.append(loss_sup_tde.item())
            loss_ent_list.append(loss_ent.item())
            loss_cons_list.append(loss_cons.item())
            loss_dom_list.append(loss_dom.item())

        results["loss_total"] = sum(loss_total_list) / len(loss_total_list)
        results["loss_sup"] = sum(loss_sup_list) / len(loss_sup_list)
        results["loss_sup_fact"] = sum(loss_sup_fact_list) / len(loss_sup_fact_list)
        results["loss_sup_tde"] = sum(loss_sup_tde_list) / len(loss_sup_tde_list)
        results["loss_ent"] = sum(loss_ent_list) / len(loss_ent_list)
        results["loss_cons"] = sum(loss_cons_list) / len(loss_cons_list)
        results["loss_dom"] = sum(loss_dom_list) / len(loss_dom_list)
        return results

    def train(self):
        if not os.path.exists(self.args.style_dir):
            os.makedirs(self.args.style_dir)

        for epoch in tqdm(range(self.start_epoch, self.args.epochs)):
            epoch_start = datetime.now()
            self.adjust_lr(epoch)

            train_results = self.train_epoch(epoch)

            results = self.eval(modality="target", phase="valid")
            if len(results["dc"]) == 2:
                lge_dice = results["dc"][0]
            else:
                lge_dice = np.round((results["dc"][0] + results["dc"][2] + results["dc"][4]) / 3, 3)

            if self.args.evalT:
                results_test = self.eval(modality="target", phase="test")
                if len(results_test["dc"]) == 2:
                    lge_dice_test = results_test["dc"][0]
                else:
                    lge_dice_test = np.round(
                        (results_test["dc"][0] + results_test["dc"][2] + results_test["dc"][4]) / 3, 3
                    )
                self.writer.add_scalars("Dice/LGE", {"Valid": lge_dice, "Test": lge_dice_test}, epoch + 1)
            else:
                self.writer.add_scalar("Dice/LGE_valid", lge_dice, epoch + 1)

            self.writer.add_scalar("Loss/Total", train_results["loss_total"], epoch + 1)
            self.writer.add_scalar("Loss/Sup", train_results["loss_sup"], epoch + 1)
            self.writer.add_scalar("Loss/Sup_fact", train_results["loss_sup_fact"], epoch + 1)
            self.writer.add_scalar("Loss/Sup_tde", train_results["loss_sup_tde"], epoch + 1)
            self.writer.add_scalar("Loss/Ent", train_results["loss_ent"], epoch + 1)
            self.writer.add_scalar("Loss/Cons", train_results["loss_cons"], epoch + 1)
            self.writer.add_scalar("Loss/Dom", train_results["loss_dom"], epoch + 1)
            self.writer.add_scalar("LR/Seg", self.opt.param_groups[0]["lr"], epoch + 1)

            print(
                f'Epoch = {epoch + 1:4d}/{self.args.epochs:4d}, '
                f'loss_total = {train_results["loss_total"]:.4f}, dc_valid = {lge_dice:.4f}'
            )

            if epoch + 1 >= 0 and (epoch + 1) % 5 == 0:
                save_path = os.path.join("results", f"checkpoint_epoch{epoch + 1}.pth")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.segmentor.state_dict(),
                    "optimizer_state_dict": self.opt.state_dict(),
                }, save_path)
                print(f"Checkpoint saved: {save_path}")

            tobreak = self.stop_training(epoch, epoch_start, lge_dice)
            self.mcp_segmentor.step(
                monitor=lge_dice,
                model=self.segmentor,
                epoch=epoch + 1,
                optimizer=self.opt,
                tobreak=tobreak,
            )
            if tobreak:
                break

        self.writer.close()

        best_epoch = self.mcp_segmentor.epoch
        best_score = self.mcp_segmentor.best_result
        log_dir_new = "runs/{}.e{}.Scr{}".format(self.apdx, best_epoch, np.around(best_score, 3))
        os.rename(self.log_dir, log_dir_new)
        print("the weight of the best model: {}".format(self.mcp_segmentor.best_model_save_dir))

        checkpoint = torch.load(self.mcp_segmentor.best_model_save_dir)
        model_state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        self.segmentor.load_state_dict(model_state, strict=False)
        print("model loaded")

        self.eval(modality="target", phase="test")
        return
