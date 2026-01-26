import torch
import torch.nn as nn
from .Basic_module import Criterion, Visualization
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, args, base_channels=45, output_channels=2, input_channels=1):
        super(UNet, self).__init__()
        self.args = args
        self.num_classes = args.num_classes
        # 编码器部分
        self.encoder1 = self.conv_block(input_channels, base_channels)
        self.encoder2 = self.conv_block(base_channels, base_channels * 2)
        self.encoder3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.encoder4 = self.conv_block(base_channels * 4, base_channels * 8)
        self.bottleneck = self.conv_block(base_channels * 8, base_channels * 16)
            
        # 解码器部分
        self.upconv4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(base_channels * 16, base_channels * 8)
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(base_channels * 8, base_channels * 4)
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(base_channels * 4, base_channels * 2)
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(base_channels * 2, base_channels)
        
        # 修改输出层为 2 通道
        self.output = nn.Conv2d(base_channels, output_channels, kernel_size=1)  # 输出 2 个通道

        # 初始化权重
        self.initialize_weights()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # 反卷积和解码
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        pred = self.output(dec1)

        visualize = {
            
        }

        out = {
            "pred_masks": pred,
            "visualize": visualize,
        }
        return out
    

class UNet_Criterion(Criterion):
    def __init__(self, args):
        super(UNet_Criterion, self).__init__(args)

    def forward(self, pred, grnd):
        loss_dict = {
            "loss_Dice_CE": self.compute_dice_ce_loss(pred["pred_masks"], grnd),
            "Dice": self.compute_dice(pred["pred_masks"], grnd),
        }
        losses = (
            loss_dict["loss_Dice_CE"]
        )
        return losses, loss_dict


class UNetVis(Visualization):
    def __init__(self):
        super(UNetVis, self).__init__()

    def forward(self, inputs, outputs, labels, others, epoch, writer):
        self.save_image(inputs.as_tensor(), "inputs", epoch, writer)
        self.save_image(outputs.float().as_tensor(), "outputs", epoch, writer)
        self.save_image(labels.float().as_tensor(), "labels", epoch, writer)
        for key, value in others.items():
            self.save_image(value.float().as_tensor(), key, epoch, writer)



def build(args):
    model = UNet(args)
    criterion = UNet_Criterion(args)
    visualizer = UNetVis()
    return model, criterion, visualizer
