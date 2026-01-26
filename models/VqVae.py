import torch
import torch.nn as nn
import numpy as np
import random
import torch.backends.cudnn as cudnn
from .Basic_module import Criterion, Visualization
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import os
from glob import glob
from monai.data import CacheDataset, PatchDataset
from data.transform import (
    volume_transform,
    slice_transform_train,
    slice_transform_valid,
    FilterSliced,
)
from monai.transforms import ScaleIntensity


def build_Prostate(image_set, dataset_dir):
    assert os.path.exists(
        dataset_dir
    ), f"provided data path {dataset_dir} does not exist"

    file_paths = glob(os.path.join(dataset_dir, "RUNMC", image_set, "*.nii.gz"))
    if image_set == "test":
        file_paths = glob(os.path.join(dataset_dir, "BIDMC", "*.nii.gz"))

    image_paths, label_paths = [], []
    for path in file_paths:
        if path.split("/")[-1][7:10] in ["seg", "Seg"]:
            label_paths.append(path)
        else:
            image_paths.append(path)

    image_paths, label_paths = sorted(image_paths), sorted(label_paths)
    path_dicts = [
        {"image": image_path, "label": label_path, "ori_image":image_path}
        for image_path, label_path in zip(image_paths, label_paths)
    ]

    # split train and val set
    if image_set == "train":
        slice_transform = slice_transform_train
    elif image_set == "val":
        slice_transform = slice_transform_valid
    elif image_set == "test":
        slice_transform = slice_transform_valid

    dataset = CacheDataset(
        data=path_dicts, transform=volume_transform, cache_rate=1.0, num_workers=4
    )
    slice_sampler = FilterSliced(
        ["image", "label", "ori_image"], source_key="label", samples_per_image=12
    )
    slice_dataset = PatchDataset(dataset, slice_sampler, 12, slice_transform)
    return slice_dataset

# VQ-VAE 中的向量量化层
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # 创建代码本，形状为 [num_embeddings, embedding_dim]
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        # 使用 Xavier 初始化权重
        #nn.init.xavier_uniform_(self.embedding.weight)
        #nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))


    def forward(self, inputs):
        # 1. 将输入展平为 [batch, height, width, embedding_dim]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        # 2. 计算每个输入到所有嵌入的距离并找到最近的嵌入
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True) +
                     torch.sum(self.embedding.weight ** 2, dim=1) -
                     2 * torch.matmul(flat_input, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embedding(encoding_indices).view(input_shape)

        # 3. 损失计算
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # 4. 将量化后的向量的梯度替换为输入的梯度
        quantized = inputs + (quantized - inputs).detach()
        return quantized.permute(0, 3, 1, 2).contiguous(), loss

# VQ-VAE 模型结构
class VQVAE(nn.Module):
    def __init__(self, embedding_dim=720, num_embeddings=512, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.d_model = embedding_dim

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 96x96x32
            #nn.BatchNorm2d(32),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 48x48x64
            #nn.BatchNorm2d(64),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 24x24x128
            #nn.BatchNorm2d(128),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, embedding_dim, kernel_size=4, stride=2, padding=1),  # 12x12xembedding_dim
            #nn.BatchNorm2d(embedding_dim),
            nn.InstanceNorm2d(embedding_dim),
            #nn.ReLU(),
        
        )
        
        # 向量量化层
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 128, kernel_size=4, stride=2, padding=1), # 24
            #nn.BatchNorm2d(128),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 48
            #nn.BatchNorm2d(64),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 96
            #nn.BatchNorm2d(32),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # 192
            #nn.ReLU(),
            nn.Sigmoid()
        )

        # 初始化权重
        #self.initialize_weights()

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
        # 编码
        z = self.encoder(x)
        # 向量量化
        quantized, vq_loss = self.vq_layer(z)
        # 解码
        x_recon = self.decoder(quantized)

        visualize = {
            
        }

        out = {
            "pred_masks": x_recon,
            "vq_loss": vq_loss,
            "visualize": visualize,
        }
        return out


class VQVAE_Criterion(Criterion):
    def __init__(self, args):
        super(VQVAE_Criterion, self).__init__(args)
        self.mse = nn.MSELoss()

    def loss_recon(self, x_recon, x):
        return self.mse(x_recon,x)
    
    def calculate_psnr_torch(self, original, reconstructed, max_pixel=1.0):
        mse = torch.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 10 * torch.log10(max_pixel ** 2 / mse)
        return psnr
    
    def forward(self, pred, grnd):
        loss_dict = {
            "loss_recon": self.loss_recon(pred["pred_masks"], grnd),
            "Dice": self.calculate_psnr_torch(pred["pred_masks"], grnd),
            "loss_vq": pred["vq_loss"],
        }
        losses = (
            loss_dict["loss_recon"] + pred["vq_loss"]
        )
        return losses, loss_dict


class VQVAEVis(Visualization):
    def __init__(self):
        super(VQVAEVis, self).__init__()

    def forward(self, inputs, outputs, labels, others, epoch, writer):
        self.save_image(inputs.as_tensor(), "inputs", epoch, writer)
        self.save_image(outputs.float().as_tensor(), "outputs", epoch, writer)
        self.save_image(labels.float().as_tensor(), "labels", epoch, writer)
        for key, value in others.items():
            self.save_image(value.float().as_tensor(), key, epoch, writer)

def scale(batch_tensor):
    normalized_images = []
    for image in batch_tensor:
        # 逐图片获取最小值和最大值
        xmin = image.min()
        xmax = image.max()
        # 避免除以零
        if xmax - xmin > 0:
            normalized_image = (image - xmin) / (xmax - xmin)
        else:
            normalized_image = image - xmin  # 如果 xmax == xmin，则全设为0
        normalized_images.append(normalized_image)
    return torch.stack(normalized_images)

def build(args):
    model = VQVAE()
    criterion = VQVAE_Criterion(args)
    visualizer = VQVAEVis()
    return model, criterion, visualizer

if __name__ == "__main__":

    # 将训练集和验证集创建为 DataLoader
    batch_size = 32
    dataset_dir = "Processed_data_nii"
    num_workers = 4
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = True
    train_dataset = build_Prostate(image_set="train", dataset_dir=dataset_dir)
    test_dataset = build_Prostate(image_set="test", dataset_dir=dataset_dir)
    val_dataset = build_Prostate(image_set="val", dataset_dir=dataset_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 设置 VQ-VAE 模型和优化器
    device = torch.device("cuda:0")
    model = VQVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    # 训练循环
    writer = SummaryWriter("runs/vqvae_experiment2")
    num_epochs = 500

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_step = len(train_loader)
        train_iterator = iter(train_loader)
        val_iterator = iter(val_loader)
        test_iterator = iter(test_loader)

        for batch_idx in range(total_step):
            data_dict = next(train_iterator)
            data = data_dict["image"]
            data = data.to(device)
            ori_data = data_dict["ori_image"]
            ori_data = ori_data.to(device)
            optimizer.zero_grad()

            # 前向传播
            out = model(data)
            recon_data = out["pred_masks"]
            vq_loss = out["vq_loss"]
            recon_loss = nn.functional.mse_loss(recon_data, ori_data)
            loss = recon_loss + vq_loss

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            # TensorBoard 记录
            if batch_idx % 240 == 0:
                writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + batch_idx)
                writer.add_images("Train/Input", scale(data).cpu(), epoch * len(train_loader) + batch_idx)
                writer.add_images("Train/Recon", scale(recon_data).cpu(), epoch * len(train_loader) + batch_idx)
                writer.add_images("Train/Ori", scale(ori_data).cpu(), epoch * len(train_loader) + batch_idx)
        
        scheduler.step()
        # 打印当前学习率（可选）
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}, Current Learning Rate: {current_lr:.6f}")
        # 验证集损失
        model.eval()
        test_loss = 0.0
        val_loss = 0.0

        with torch.no_grad():
            for step in range(len(test_loader)):
                data_dict = next(test_iterator)
                data = data_dict["image"]
                data = data.to(device)
                out = model(data)
                recon_data = out["pred_masks"]
                vq_loss = out["vq_loss"]
                recon_loss = nn.functional.mse_loss(recon_data, data)
                loss = recon_loss + vq_loss
                test_loss += loss.item()
            for step in range(len(val_loader)):
                data_dict = next(val_iterator)
                data_val = data_dict["image"]
                data_val = data_val.to(device)
                out = model(data_val)
                recon_data_val = out["pred_masks"]
                vq_loss_val = out["vq_loss"]
                recon_loss_val = nn.functional.mse_loss(recon_data_val, data_val)
                loss_val = recon_loss_val + vq_loss_val
                val_loss += loss_val.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        writer.add_scalar("Loss/avg_train_loss", avg_train_loss, epoch + 1)
        writer.add_scalar("Loss/avg_test_loss", avg_test_loss, epoch + 1)
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Loss/avg_val_loss", avg_val_loss, epoch + 1)

        # 每个 epoch 记录验证集的输入和输出
        writer.add_images("Test/Input", scale(data).cpu(), epoch + 1)
        writer.add_images("Test/Output", scale(recon_data).cpu(), epoch + 1)
        writer.add_images("Val/Input", scale(data_val).cpu(), epoch + 1)
        writer.add_images("Val/Output", scale(recon_data_val).cpu(), epoch + 1)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f},  Avg Val Loss: {avg_val_loss:.4f}")
        if epoch>=400 and epoch %50 ==0:
            temp_path = "runs/vqvae_model_"+str(epoch)+".pth"
            torch.save(model.state_dict(), temp_path)
            print(f"模型已保存到 {temp_path}")

    # 保存模型
    model_save_path = "runs/vqvae_model_"+str(num_epochs)+".pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存到 {model_save_path}")

    # 关闭 TensorBoard
    writer.close()