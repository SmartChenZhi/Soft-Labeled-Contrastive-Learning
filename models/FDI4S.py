import torch
import torch.nn as nn
import torch.nn.functional as F

from .Basic_module import Criterion, Visualization


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        widths = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 8,
        ]
        self.blocks = nn.ModuleList(
            [
                ConvBlock(in_channels, widths[0]),
                ConvBlock(widths[0], widths[1]),
                ConvBlock(widths[1], widths[2]),
                ConvBlock(widths[2], widths[3]),
                ConvBlock(widths[3], widths[4]),
            ]
        )
        self.pool = nn.MaxPool2d(2)
        self.out_channels = widths[-1]

    def forward(self, x):
        skips = []
        out = x
        for idx, block in enumerate(self.blocks):
            out = block(out)
            if idx < len(self.blocks) - 1:
                skips.append(out)
                out = self.pool(out)
        return out, skips


class Decoder(nn.Module):
    def __init__(self, bottleneck_channels, base_channels, out_channels):
        super().__init__()
        widths = [
            base_channels * 8,
            base_channels * 4,
            base_channels * 2,
            base_channels,
        ]
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(bottleneck_channels, widths[0], kernel_size=2, stride=2),
                nn.ConvTranspose2d(widths[0], widths[1], kernel_size=2, stride=2),
                nn.ConvTranspose2d(widths[1], widths[2], kernel_size=2, stride=2),
                nn.ConvTranspose2d(widths[2], widths[3], kernel_size=2, stride=2),
            ]
        )
        self.blocks = nn.ModuleList(
            [
                ConvBlock(widths[0] + base_channels * 8, widths[0]),
                ConvBlock(widths[1] + base_channels * 4, widths[1]),
                ConvBlock(widths[2] + base_channels * 2, widths[2]),
                ConvBlock(widths[3] + base_channels, widths[3]),
            ]
        )
        self.head = nn.Conv2d(widths[-1], out_channels, kernel_size=1)

    def forward(self, x, skips):
        for upconv, block, skip in zip(self.upconvs, self.blocks, reversed(skips)):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return self.head(x)


class GlobalSamplingModule(nn.Module):
    def __init__(self, num_classes, feature_channels, base_channels):
        super().__init__()
        self.num_classes = num_classes
        hidden = max(base_channels // 2, 8)
        self.encoder = nn.Sequential(
            ConvBlock(1, hidden),
            nn.MaxPool2d(2),
            ConvBlock(hidden, hidden * 2),
            nn.MaxPool2d(2),
            ConvBlock(hidden * 2, hidden * 4),
            nn.MaxPool2d(2),
            ConvBlock(hidden * 4, feature_channels),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_channels, hidden * 4, kernel_size=2, stride=2),
            ConvBlock(hidden * 4, hidden * 4),
            nn.ConvTranspose2d(hidden * 4, hidden * 2, kernel_size=2, stride=2),
            ConvBlock(hidden * 2, hidden * 2),
            nn.ConvTranspose2d(hidden * 2, hidden, kernel_size=2, stride=2),
            ConvBlock(hidden, hidden),
            nn.ConvTranspose2d(hidden, 1, kernel_size=2, stride=2),
        )

    def encode(self, masks):
        batch_size, num_classes, height, width = masks.shape
        encoded = self.encoder(masks.reshape(batch_size * num_classes, 1, height, width))
        _, channels, h_small, w_small = encoded.shape
        return encoded.reshape(batch_size, num_classes, channels, h_small, w_small)

    def reconstruct(self, class_features):
        batch_size, num_classes, channels, h_small, w_small = class_features.shape
        decoded = self.decoder(class_features.reshape(batch_size * num_classes, channels, h_small, w_small))
        return decoded.reshape(batch_size, num_classes, decoded.shape[-2], decoded.shape[-1])

    def forward(self, masks):
        class_features = self.encode(masks)
        recon = self.reconstruct(class_features)
        return class_features, recon


class CausalIntervention(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, local_features, global_features):
        batch_size, channels, height, width = local_features.shape
        query = local_features.flatten(2).transpose(1, 2)
        global_tokens = global_features.reshape(-1, channels, height * width)
        global_tokens = global_tokens.permute(0, 2, 1).reshape(1, -1, channels)
        global_tokens = global_tokens.expand(batch_size, -1, -1)
        attended, _ = self.attn(query, global_tokens, global_tokens)
        attended = self.norm(attended + query)
        return attended.transpose(1, 2).reshape(batch_size, channels, height, width)


class FDI4S(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_classes = args.num_classes

        self.encoder = Encoder(args.in_channels, args.fdi_base_channels)
        bottleneck_channels = self.encoder.out_channels
        self.gs_module = GlobalSamplingModule(
            num_classes=args.num_classes,
            feature_channels=bottleneck_channels,
            base_channels=args.fdi_base_channels,
        )
        self.causal_intervention = CausalIntervention(
            channels=bottleneck_channels,
            num_heads=args.fdi_attn_heads,
        )
        self.decoder = Decoder(
            bottleneck_channels=bottleneck_channels * 2,
            base_channels=args.fdi_base_channels,
            out_channels=args.num_classes,
        )

        self.register_buffer("global_context", torch.zeros(0))
        self.register_buffer("global_context_ready", torch.zeros(1, dtype=torch.uint8))

    def one_hot_labels(self, labels):
        if labels.dim() == 4 and labels.size(1) == 1:
            labels = labels.squeeze(1)
        one_hot = F.one_hot(labels.long(), num_classes=self.num_classes)
        return one_hot.permute(0, 3, 1, 2).float()

    def gs_forward(self, labels):
        one_hot = self.one_hot_labels(labels)
        return self.gs_module(one_hot)

    @torch.no_grad()
    def build_global_context(self, loader, device):
        self.eval()
        total = None
        count = 0
        for batch in loader:
            labels = batch["label"].to(device)
            class_features, _ = self.gs_forward(labels)
            batch_mean = class_features.mean(dim=0)
            total = batch_mean if total is None else total + batch_mean
            count += 1
        if total is None or count == 0:
            raise RuntimeError("Cannot build global context from an empty loader.")
        global_context = total / count
        self.global_context = global_context
        self.global_context_ready.fill_(1)

    def forward(self, samples):
        if self.global_context_ready.item() == 0:
            raise RuntimeError("Global context is not initialized. Run GS pretraining first.")
        feature_z, skips = self.encoder(samples)
        feature_ci = self.causal_intervention(feature_z, self.global_context)
        pred = self.decoder(torch.cat([feature_z, feature_ci], dim=1), skips)
        return {
            "pred_masks": pred,
            "visualize": {
                "f_z": feature_z[:, :1],
                "f_ci": feature_ci[:, :1],
            },
        }


class FDI4SCriterion(Criterion):
    def __init__(self, args):
        super().__init__(args)

    def forward(self, pred, grnd):
        loss_dict = {
            "loss_Dice_CE": self.compute_dice_ce_loss(pred["pred_masks"], grnd),
            "Dice": self.compute_dice(pred["pred_masks"], grnd),
        }
        return loss_dict["loss_Dice_CE"], loss_dict


class FDI4SVis(Visualization):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, outputs, labels, others, epoch, writer):
        self.save_image(inputs.as_tensor(), "inputs", epoch, writer)
        self.save_image(outputs.float().as_tensor(), "outputs", epoch, writer)
        self.save_image(labels.float().as_tensor(), "labels", epoch, writer)
        for key, value in others.items():
            self.save_image(value.float().as_tensor(), key, epoch, writer)


def build(args):
    model = FDI4S(args)
    criterion = FDI4SCriterion(args)
    visualizer = FDI4SVis()
    return model, criterion, visualizer
