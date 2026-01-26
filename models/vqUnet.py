import torch
import torch.nn as nn
from .Basic_module import Criterion, Visualization
import torch.nn.functional as F
from .VqVae import VQVAE

class vqUNet(nn.Module):
    def __init__(self, args, base_channels=45, output_channels=2, input_channels=1):
        super(vqUNet, self).__init__()
        self.args = args
        self.num_classes = args.num_classes
        self.base_channels = base_channels
        self.embedding_dim = base_channels*16
        # 编码器部分（包含向量量化层）
        self.encoder1 = self.conv_block(input_channels, base_channels)
        self.encoder2 = self.conv_block(base_channels, base_channels * 2)
        self.encoder3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.encoder4 = self.conv_block(base_channels * 4, base_channels * 8)
        self.bottleneck = self.conv_block(base_channels * 8, base_channels * 16)
        self.vqvae = VQVAE(embedding_dim=self.embedding_dim)
        self.d_model = self.vqvae.d_model
        d_model = self.d_model
        # 定义可学习参数
        self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_o = torch.nn.Linear(d_model, d_model, bias=False)
        
        
        # 解码器部分
        self.upconv4 = nn.ConvTranspose2d(base_channels * 32, base_channels * 8, kernel_size=2, stride=2)
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
            #nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x,ori_samples):
         
        vqout = self.vqvae(ori_samples)
        x_recon =  vqout["pred_masks"]
        vq_loss = vqout["vq_loss"]
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        embedding_weights_constant = self.vqvae.vq_layer.embedding.weight.data.detach()
        batch_size = bottleneck.size(0)
        codebook = embedding_weights_constant.unsqueeze(0).repeat(batch_size,1,1)
        bottleneck = bottleneck.view(batch_size, self.embedding_dim, -1).permute(0, 2, 1)

        # 定义超参数
        batch_size = bottleneck.size(0)
        num_heads = 8  # 多头的数量
        head_dim = self.d_model // num_heads  # 每个头的维度
        # 线性变换获取 Q, K, V
        Q = self.W_q(bottleneck)  # (batch_size, seq_len_bottleneck, d_model)
        K = self.W_k(codebook)    # (batch_size, seq_len_codebook, d_model)
        V = self.W_v(codebook)    # (batch_size, seq_len_codebook, d_model)
        Q = split_heads(Q, batch_size, num_heads, head_dim)  # (batch_size, num_heads, seq_len_bottleneck, head_dim)
        K = split_heads(K, batch_size, num_heads, head_dim)  # (batch_size, num_heads, seq_len_codebook, head_dim)
        V = split_heads(V, batch_size, num_heads, head_dim)  # (batch_size, num_heads, seq_len_codebook, head_dim)
        # 点积注意力
        d_k = head_dim
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))  # (batch_size, num_heads, seq_len_bottleneck, seq_len_codebook)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_len_bottleneck, seq_len_codebook)
        head_output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len_bottleneck, head_dim)
        multi_head_output = combine_heads(head_output, batch_size, num_heads, head_dim)  # (batch_size, seq_len_bottleneck, d_model)
        # 线性变换输出
        output = self.W_o(multi_head_output)  # (batch_size, seq_len_bottleneck, d_model)
        output = output.permute(0, 2, 1).view(batch_size, self.embedding_dim, 12, 12)
        bottleneck = bottleneck.permute(0, 2, 1).view(batch_size, self.embedding_dim, 12, 12)
        
        bottleneck = torch.cat((output, bottleneck), dim=1)

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
            "x_recon":x_recon,
        }

        out = {
            "pred_masks": pred,
            "visualize": visualize,
            "x_recon":x_recon,
            "vq_loss":vq_loss,
        }
        return out
    

class UNet_Criterion(Criterion):
    def __init__(self, args):
        super(UNet_Criterion, self).__init__(args)
        
        self.mse = nn.MSELoss()

    def loss_recon(self, x_recon, x):
        return self.mse(x_recon,x)

    def forward(self, pred, grnd, samples):
        loss_dict = {
            "loss_Dice_CE": self.compute_dice_ce_loss(pred["pred_masks"], grnd),
            "Dice": self.compute_dice(pred["pred_masks"], grnd),
            "loss_recon": self.loss_recon(pred["x_recon"],samples),
            "loss_vq": pred["vq_loss"],
        }
        losses = (
            loss_dict["loss_Dice_CE"]+ loss_dict["loss_recon"] + loss_dict["loss_vq"]
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
    model = vqUNet(args,base_channels=45)
    criterion = UNet_Criterion(args)
    visualizer = UNetVis()
    return model, criterion, visualizer


# 拆分为多头
def split_heads(x, batch_size, num_heads, head_dim):
    """
    分割张量为多头形式
    输入: (batch_size, seq_len, d_model)
    输出: (batch_size, num_heads, seq_len, head_dim)
    """
    return x.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

# 合并多头输出
def combine_heads(x, batch_size, num_heads, head_dim):
    """
    合并多头的输出
    输入: (batch_size, num_heads, seq_len, head_dim)
    输出: (batch_size, seq_len, d_model)
    """
    return x.transpose(1, 2).reshape(batch_size, -1, num_heads * head_dim)

