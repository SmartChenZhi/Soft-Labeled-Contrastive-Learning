import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DRUNet import Encoder, Bottleneck, Decoder
from utils.utils_ import get_n_params


class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradientReverseFunction.apply(x, lambd)


class CausalSegmentationModel(nn.Module):
    def __init__(
        self,
        filters=32,
        in_channels=3,
        n_block=4,
        bottleneck_depth=4,
        n_class=4,
        z_dim=32,
        dom_emb_dim=16,
        grl_lambda=1.0,
    ):
        super().__init__()
        self.filters = filters
        self.z_dim = z_dim
        self.grl_lambda = grl_lambda

        # X branch: identical DRUNet backbone blocks
        self.encoder = Encoder(filters=filters, in_channels=in_channels, n_block=n_block)
        self.bottleneck = Bottleneck(filters=filters, n_block=n_block, depth=bottleneck_depth)
        self.decoder = Decoder(filters=filters, n_block=n_block)

        # Z branch: style encoder + domain embedding + MLP
        self.style_encoder = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters),
            nn.Conv2d(filters, filters, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters),
            nn.Conv2d(filters, filters * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters * 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.domain_embedding = nn.Embedding(2, dom_emb_dim)
        self.z_mlp = nn.Sequential(
            nn.Linear(filters * 2 + dom_emb_dim, z_dim),
            nn.ReLU(inplace=True),
            nn.Linear(z_dim, z_dim),
        )

        # Shared segmentation head for factual and counterfactual outputs
        self.seg_head = nn.Sequential(
            nn.Conv2d(filters + z_dim, filters, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(filters, n_class, kernel_size=1),
        )

        # Domain classifier with GRL
        self.domain_classifier = nn.Sequential(
            nn.Linear(filters, filters),
            nn.ReLU(inplace=True),
            nn.Linear(filters, 2),
        )

        # Running global structure feature X_bar
        self.register_buffer("x_bar", torch.zeros(filters))
        self.register_buffer("x_bar_initialized", torch.zeros(1, dtype=torch.uint8))

        self.number_params()

    def number_params(self):
        print(f"Number of params: {get_n_params(self):,}")

    def extract_x_feature(self, x):
        encoded, skip = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        feat_x = self.decoder(bottleneck, skip, adaptseg=False)
        return feat_x

    def _domain_tensor(self, domain_labels, batch_size, device):
        if domain_labels is None:
            return torch.ones(batch_size, dtype=torch.long, device=device)
        if isinstance(domain_labels, int):
            return torch.full((batch_size,), int(domain_labels), dtype=torch.long, device=device)
        return domain_labels.to(device=device, dtype=torch.long)

    def _build_z_map(self, x, domain_labels, h, w):
        batch_size = x.size(0)
        domain_labels = self._domain_tensor(domain_labels, batch_size, x.device)
        style_vec = self.style_encoder(x).flatten(1)
        dom_vec = self.domain_embedding(domain_labels)
        z = self.z_mlp(torch.cat([style_vec, dom_vec], dim=1))
        z_map = z.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        return z_map

    def _x_bar_map(self, batch_size, h, w, x_bar_override=None):
        if x_bar_override is None:
            x_bar_vec = self.x_bar
        else:
            x_bar_vec = x_bar_override
        if x_bar_vec.dim() == 1:
            return x_bar_vec.view(1, -1, 1, 1).expand(batch_size, -1, h, w)
        if x_bar_vec.dim() == 4:
            return x_bar_vec
        raise ValueError(f"Unsupported x_bar shape: {tuple(x_bar_vec.shape)}")

    @torch.no_grad()
    def update_x_bar(self, batch_mean, momentum=0.9):
        if batch_mean.dim() != 1:
            raise ValueError(f"batch_mean must be 1D, got {tuple(batch_mean.shape)}")
        if self.x_bar_initialized.item() == 0:
            self.x_bar.copy_(batch_mean)
            self.x_bar_initialized.fill_(1)
        else:
            self.x_bar.mul_(1 - momentum).add_(momentum * batch_mean)

    @torch.no_grad()
    def reset_x_bar(self):
        self.x_bar.zero_()
        self.x_bar_initialized.zero_()

    def forward(self, x, domain_labels=None, x_bar_override=None, grl_lambda=None, feat_x=None):
        if feat_x is None:
            feat_x = self.extract_x_feature(x)
        b, _, h, w = feat_x.shape

        z_map = self._build_z_map(x, domain_labels, h, w)
        x_bar_map = self._x_bar_map(b, h, w, x_bar_override=x_bar_override)

        logits_fact = self.seg_head(torch.cat([feat_x, z_map], dim=1))
        logits_cf = self.seg_head(torch.cat([x_bar_map, z_map], dim=1))
        logits_tde = logits_fact - logits_cf

        phi_tde = F.adaptive_avg_pool2d(feat_x, output_size=1).flatten(1)
        lambd = self.grl_lambda if grl_lambda is None else grl_lambda
        dom_logits = self.domain_classifier(grad_reverse(phi_tde, lambd))

        return logits_tde, logits_fact, logits_cf, feat_x, dom_logits
