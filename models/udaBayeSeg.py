import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientunet import get_efficientunet_b2

from .Basic_module import Criterion, Visualization
from .ResNet import ResNet_appearance, ResNet_shape
from .Unet import UNet


class udaBayeSeg(nn.Module):
    def __init__(self, args):
        super(udaBayeSeg, self).__init__()

        self.args = args
        self.num_classes = args.num_classes

        self.res_shape = ResNet_shape(num_out_ch=2)
        self.res_appear_s = ResNet_appearance(num_out_ch=2, num_block=6, bn=True)
        self.res_appear_t = ResNet_appearance(num_out_ch=2, num_block=6, bn=True)
        # self.unet = get_efficientunet_b2(
        #     out_channels=2 * args.num_classes, pretrained=False
        # )
        self.unet = UNet(args,base_channels=45,output_channels=4,input_channels=1)
        self.unet_teacher = UNet(args,base_channels=45,output_channels=4,input_channels=1)
        self.unet_teacher.load_state_dict(self.unet.state_dict())  # 初始同步
        self.ema_decay = args.ema_decay
        for param in self.unet_teacher.parameters():
            param.requires_grad = False

        self.softmax = nn.Softmax(dim=1)

        Dx = torch.zeros([1, 1, 3, 3], dtype=torch.float)
        Dx[:, :, 1, 1] = 1
        Dx[:, :, 1, 0] = Dx[:, :, 1, 2] = Dx[:, :, 0, 1] = Dx[:, :, 2, 1] = -1 / 4
        self.Dx = nn.Parameter(data=Dx, requires_grad=False)

        #self.load_pretrained_parts("logs/model2/best_checkpoint.pth")

    def update_teacher(self):
        # 用 EMA 更新教师模型参数
        for student_param, teacher_param in zip(self.unet.parameters(), self.unet_teacher.parameters()):
            teacher_param.data.mul_(self.ema_decay).add_((1 - self.ema_decay) * student_param.data)


    def load_pretrained_parts(self, checkpoint_path):
        for param in self.res_shape.parameters():
            param.requires_grad = False
        for param in self.res_appear.parameters():
            param.requires_grad = False

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Extract model state_dict
        pretrained_state_dict = checkpoint["model"]

        # Load res_shape and res_appear weights
        self.res_shape.load_state_dict({k.replace("res_shape.", ""): v 
                                        for k, v in pretrained_state_dict.items() if k.startswith("res_shape.")})
        self.res_appear.load_state_dict({k.replace("res_appear.", ""): v 
                                         for k, v in pretrained_state_dict.items() if k.startswith("res_appear.")})

    @staticmethod
    def sample_normal_jit(mu, log_var):
        sigma = torch.exp(log_var / 2)
        eps = mu.mul(0).normal_()
        z = eps.mul_(sigma).add_(mu)
        return z, eps

    def generate_m_t(self, samples):
        feature = self.res_appear_t(samples)
        mu_m, log_var_m = torch.chunk(feature, 2, dim=1)
        log_var_m = torch.clamp(log_var_m, -20, 0)
        m, _ = self.sample_normal_jit(mu_m, log_var_m)
        return m, mu_m, log_var_m

    def generate_m_s(self, samples):
        feature = self.res_appear_s(samples)
        mu_m, log_var_m = torch.chunk(feature, 2, dim=1)
        log_var_m = torch.clamp(log_var_m, -20, 0)
        m, _ = self.sample_normal_jit(mu_m, log_var_m)
        return m, mu_m, log_var_m

    def generate_x(self, samples):
        feature = self.res_shape(samples)
        mu_x, log_var_x = torch.chunk(feature, 2, dim=1)
        log_var_x = torch.clamp(log_var_x, -20, 0)
        x, _ = self.sample_normal_jit(mu_x, log_var_x)
        return x, mu_x, log_var_x

    def generate_z(self, x):
        feature = self.unet(x)["pred_masks"]
        mu_z, log_var_z = torch.chunk(feature, 2, dim=1)
        log_var_z = torch.clamp(log_var_z, -20, 0)
        z, _ = self.sample_normal_jit(mu_z, log_var_z)
        if self.training:
            return F.gumbel_softmax(z, dim=1), F.gumbel_softmax(mu_z, dim=1), log_var_z
        else:
            return self.softmax(z), self.softmax(mu_z), log_var_z
        
    def generate_z_dummy(self, x):
        feature = self.unet_teacher(x)["pred_masks"]
        mu_z, log_var_z = torch.chunk(feature, 2, dim=1)
        log_var_z = torch.clamp(log_var_z, -20, 0)
        z, _ = self.sample_normal_jit(mu_z, log_var_z)
        return self.softmax(z), self.softmax(mu_z), log_var_z

    def getoutput(self, samples: torch.Tensor, isSource):
        x, mu_x, log_var_x = self.generate_x(samples)
        if isSource:
            m, mu_m, log_var_m = self.generate_m_s(samples)
            z, mu_z, log_var_z = self.generate_z(x)
        else:
            m, mu_m, log_var_m = self.generate_m_t(samples)
            z, mu_z, log_var_z = self.generate_z(x)
            z_dummy, mu_z_dummy, log_var_z_dummy = self.generate_z_dummy(x)
        

        K = self.num_classes
        _, _, W, H = samples.shape

        residual = samples - (x + m)
        mu_rho_hat = (2 * self.args.gamma_rho + 1) / (
            residual * residual + 2 * self.args.phi_rho
        )
        # mu_rho_hat = torch.clamp(mu_rho_hat, 1e4, 1e8)

        normalization = torch.sum(mu_rho_hat).detach()
        n, _ = self.sample_normal_jit(m, torch.log(1 / mu_rho_hat))

        # Image line upsilon
        alpha_upsilon_hat = 2 * self.args.gamma_upsilon + K
        difference_x = F.conv2d(mu_x, self.Dx, padding=1)
        beta_upsilon_hat = (
            torch.sum(
                mu_z * (difference_x * difference_x + 2 * torch.exp(log_var_x)),
                dim=1,
                keepdim=True,
            )
            + 2 * self.args.phi_upsilon
        )  # B x 1 x W x H
        mu_upsilon_hat = alpha_upsilon_hat / beta_upsilon_hat
        # mu_upsilon_hat = torch.clamp(mu_upsilon_hat, 1e6, 1e10)

        # Seg boundary omega
        difference_z = F.conv2d(
            mu_z, self.Dx.expand(K, 1, 3, 3), padding=1, groups=K
        )  # B x K x W x H
        alpha_omega_hat = 2 * self.args.gamma_omega + 1
        pseudo_pi = torch.mean(mu_z, dim=(2, 3), keepdim=True)
        beta_omega_hat = (
            pseudo_pi * (difference_z * difference_z + 2 * torch.exp(log_var_z))
            + 2 * self.args.phi_omega
        )
        mu_omega_hat = alpha_omega_hat / beta_omega_hat
        # mu_omega_hat = torch.clamp(mu_omega_hat, 1e2, 1e6)

        # Seg category probability pi
        _, _, W, H = samples.shape
        alpha_pi_hat = self.args.alpha_pi + W * H / 2
        beta_pi_hat = (
            torch.sum(
                mu_omega_hat * (difference_z * difference_z + 2 * torch.exp(log_var_z)),
                dim=(2, 3),
                keepdim=True,
            )
            / 2
            + self.args.beta_pi
        )
        digamma_pi = torch.special.digamma(
            alpha_pi_hat + beta_pi_hat
        ) - torch.special.digamma(beta_pi_hat)

        # compute loss-related
        kl_y = residual * mu_rho_hat.detach() * residual

        kl_mu_z = torch.sum(
            digamma_pi.detach() * difference_z * mu_omega_hat.detach() * difference_z,
            dim=1,
        )
        kl_sigma_z = torch.sum(
            digamma_pi.detach()
            * (2 * torch.exp(log_var_z) * mu_omega_hat.detach() - log_var_z),
            dim=1,
        )

        kl_mu_x = torch.sum(
            difference_x * difference_x * mu_upsilon_hat.detach() * mu_z.detach(), dim=1
        )
        kl_sigma_x = (
            torch.sum(
                2 * torch.exp(log_var_x) * mu_upsilon_hat.detach() * mu_z.detach(),
                dim=1,
            )
            - log_var_x
        )

        kl_mu_m = self.args.sigma_0 * mu_m * mu_m
        kl_sigma_m = self.args.sigma_0 * torch.exp(log_var_m) - log_var_m

        visualize = {
            "shape": torch.concat([x, mu_x, torch.exp(log_var_x / 2)]),
            "appearance": torch.concat([n, m, 1 / mu_rho_hat.sqrt()]),
            "logit": torch.concat(
                [
                    z[:, 1:2, ...],
                    mu_z[:, 1:2, ...],
                    torch.exp(log_var_z / 2)[:, 1:2, ...],
                ]
            ),
            "shape_boundary": mu_upsilon_hat,
            "seg_boundary": mu_omega_hat[:, 1:2, ...],
        }

        pred = z if self.training else mu_z
        out = {
            "pred_masks": pred,
            "kl_y": kl_y,
            "kl_mu_z": kl_mu_z,
            "kl_sigma_z": kl_sigma_z,
            "kl_mu_x": kl_mu_x,
            "kl_sigma_x": kl_sigma_x,
            "kl_mu_m": kl_mu_m,
            "kl_sigma_m": kl_sigma_m,
            "normalization": normalization,
            "rho": mu_rho_hat,
            "omega": mu_omega_hat * digamma_pi,
            "upsilon": mu_upsilon_hat * mu_z,
            "visualize": visualize,
        }
        if not isSource:
            out["dummy_label"] = mu_z_dummy
            out["visualize"] = {
                "shape_t": torch.concat([x, mu_x, torch.exp(log_var_x / 2)]),
                "appearance_t": torch.concat([n, m, 1 / mu_rho_hat.sqrt()]),
                "logit_t": torch.concat(
                    [
                        z[:, 1:2, ...],
                        mu_z[:, 1:2, ...],
                        torch.exp(log_var_z / 2)[:, 1:2, ...],
                    ]
                ),
                "shape_boundary_t": mu_upsilon_hat,
                "seg_boundary_t": mu_omega_hat[:, 1:2, ...],
            }
        return out
    
    def forward(self, samples: torch.Tensor, samples_t: torch.Tensor):
        out_s = self.getoutput(samples, True)
        out_t = self.getoutput(samples_t, False)
        out = [out_s,out_t]
        return out


class udaBayeSeg_Criterion(Criterion):
    def __init__(self, args):
        super(udaBayeSeg_Criterion, self).__init__(args)
        self.bayes_loss_coef = args.bayes_loss_coef
        self.mse = nn.MSELoss()

    def loss_Bayes(self, outputs):
        N = outputs["normalization"]
        loss_y = torch.sum(outputs["kl_y"]) / N
        loss_mu_m = torch.sum(outputs["kl_mu_m"]) / N
        loss_sigma_m = torch.sum(outputs["kl_sigma_m"]) / N
        loss_mu_x = torch.sum(outputs["kl_mu_x"]) / N
        loss_sigma_x = torch.sum(outputs["kl_sigma_x"]) / N
        loss_mu_z = torch.sum(outputs["kl_mu_z"]) / N
        loss_sigma_z = torch.sum(outputs["kl_sigma_z"]) / N
        loss_Bayes = (
            loss_y
            + loss_mu_m
            + loss_sigma_m
            + loss_mu_x
            + loss_sigma_x
            + loss_mu_z
            + loss_sigma_z
        )

        return loss_Bayes

    def forward(self, out, grnd):
        pred = out[0]
        pred_t = out[1]
        loss_dict = {
            "loss_Dice_CE": self.compute_dice_ce_loss(pred["pred_masks"], grnd),
            "Dice": self.compute_dice(pred["pred_masks"], grnd),
            "loss_Bayes": self.loss_Bayes(pred),
            "rho": torch.mean(pred["rho"]),
            "omega": torch.mean(pred["omega"]),
            "upsilon": torch.mean(pred["upsilon"]),
            "loss_Dice_CE_t": self.mse(pred_t["pred_masks"], pred_t["dummy_label"]),
            "loss_Bayes_t": self.loss_Bayes(pred_t),
        }
        losses = (
            loss_dict["loss_Dice_CE"] + 0.1 * loss_dict["loss_Dice_CE_t"] + self.bayes_loss_coef * (loss_dict["loss_Bayes"] + loss_dict["loss_Bayes_t"])
        )
        return losses, loss_dict


class udaBayeSegVis(Visualization):
    def __init__(self):
        super(udaBayeSegVis, self).__init__()

    def forward(self, inputs, inputs_t, outputs, outputs_t, labels, others, others_t, epoch, writer):
        self.save_image(inputs.as_tensor(), "inputs", epoch, writer)
        self.save_image(outputs.float().as_tensor(), "outputs", epoch, writer)
        self.save_image(labels.float().as_tensor(), "labels", epoch, writer)
        for key, value in others.items():
            self.save_image(value.float().as_tensor(), key, epoch, writer)
        self.save_image(inputs_t.as_tensor(), "inputs_t", epoch, writer)
        self.save_image(outputs_t.float().as_tensor(), "outputs_t", epoch, writer)
        for key, value in others_t.items():
            self.save_image(value.float().as_tensor(), key, epoch, writer)


def build(args):
    model = udaBayeSeg(args)
    criterion = udaBayeSeg_Criterion(args)
    visualizer = udaBayeSegVis()
    return model, criterion, visualizer
