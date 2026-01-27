import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import random
from monai.data import CacheDataset, PatchDataset
from monai.transforms import (
    Compose,
    Resized,
    RandZoomd,
    Rand2DElasticd,
    RandAffined,
    NormalizeIntensityd,
    RandGaussianNoised,
    ScaleIntensityd,
    ToTensord,
)
# Assuming data is a package in the current directory or python path
from data.transform import (
    volume_transform,
    slice_transform_train,
    slice_transform_valid,
    FilterSliced,
)
import config

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)

# Define augmented transform locally
slice_transform_train_aug = Compose(
    [
        Resized(
            keys=["image", "label", "ori_image"],
            spatial_size=[config.INPUT_SIZE, config.INPUT_SIZE],
            mode=("bilinear", "nearest","bilinear"),
        ),
        # Add Random Scaling (Zoom)
        RandZoomd(
            keys=["image", "label", "ori_image"],
            min_zoom=0.5,
            max_zoom=1.5,
            mode=("bilinear", "nearest","bilinear"),
            prob=0.8,
        ),
        # Stronger Elastic Deformation
        Rand2DElasticd(
            keys=["image", "label", "ori_image"],
            spacing=(20, 20),
            magnitude_range=(5, 10),
            prob=0.8,
            padding_mode="zeros",
            mode=("bilinear", "nearest","bilinear"),
        ),
        RandAffined(
            keys=["image", "label", "ori_image"],
            mode=("bilinear", "nearest","bilinear"),
            prob=0.8,
            rotate_range=(3.14 / 2, 3.14 / 2),
            scale_range=(0.3, 0.3),
            translate_range=(20, 20),
        ),
        NormalizeIntensityd(keys=["image"]),
        RandGaussianNoised(keys=["image"], prob=0.5, std=0.5),
        ScaleIntensityd(keys=["ori_image"], minv=0., maxv=1.),
        ToTensord(keys=["image", "label", "ori_image"]),
    ]
)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        label, domain = self.data_list[idx]
        return label, torch.tensor(domain, dtype=torch.long)

def to_labeled_list(monai_dataset, domain_label):
    data_list = []
    # PatchDataset is iterable
    for item in monai_dataset:
        label = item["label"]
        # Detach or clone if necessary to avoid keeping graph? 
        # MONAI transforms return tensors.
        data_list.append((label, domain_label))
    return data_list

class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 192 -> 96
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 96 -> 48
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 48 -> 24
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 24 -> 12
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2) # 2 classes: BIDMC vs RUNMC
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_image_label_paths(base_dir, dataset_name, split):
    search_path = os.path.join(base_dir, dataset_name, split, "*.nii.gz")
    all_files = glob.glob(search_path)
    
    image_paths = []
    label_paths = []
    
    for path in all_files:
        filename = os.path.basename(path)
        # Logic from dataset.py
        # Check if it is a segmentation file
        if len(filename) >= 10 and filename[7:10] in ["seg", "Seg"]:
            label_paths.append(path)
        else:
            image_paths.append(path)
            
    return sorted(image_paths), sorted(label_paths)

def create_monai_dataset(base_dir, dataset_name, split, transform_type="valid"):
    image_paths, label_paths = get_image_label_paths(base_dir, dataset_name, split)
    
    if len(image_paths) != len(label_paths):
        print(f"Warning: Mismatch in {dataset_name}/{split}: {len(image_paths)} images, {len(label_paths)} labels")
        
    path_dicts = [
        {"image": img, "label": lbl, "ori_image": img}
        for img, lbl in zip(image_paths, label_paths)
    ]
    
    # Logic from dataset.py
    if transform_type == "train":
        random.shuffle(path_dicts)
        slice_transform = slice_transform_train_aug
    else:
        slice_transform = slice_transform_valid
        
    # Use CacheDataset as in dataset.py
    # cache_rate=1.0 is fine for small datasets
    dataset = CacheDataset(
        data=path_dicts, transform=volume_transform, cache_rate=1.0, num_workers=0
    )
    
    slice_sampler = FilterSliced(
        ["image", "label", "ori_image"], source_key="label", samples_per_image=12
    )
    
    slice_dataset = PatchDataset(dataset, slice_sampler, 12, slice_transform)
    return slice_dataset

def main():
    base_dir = "/root/SLCL/Processed_data_nii_uda"
    
    print("Preparing datasets using logic from data/dataset.py and data/transform.py...")
    
    # 1. Gather Data based on user instructions
    # Train: BIDMC/val + RUNMC/train
    # NOTE: We use transform_type="train" for training data to ensure augmentations are applied consistently
    train_bidmc_monai = create_monai_dataset(base_dir, "BIDMC", "val", transform_type="train")
    train_runmc_monai = create_monai_dataset(base_dir, "RUNMC", "train", transform_type="train")
    
    # Val: BIDMC/val + RUNMC/val
    val_bidmc_monai = create_monai_dataset(base_dir, "BIDMC", "val", transform_type="train")
    val_runmc_monai = create_monai_dataset(base_dir, "RUNMC", "val", transform_type="train")
    
    # Test: BIDMC/test + RUNMC/test
    test_bidmc_monai = create_monai_dataset(base_dir, "BIDMC", "test", transform_type="train")
    test_runmc_monai = create_monai_dataset(base_dir, "RUNMC", "test", transform_type="train")
    
    print("Converting datasets to memory lists...")
    
    train_list = []
    train_bidmc_list = to_labeled_list(train_bidmc_monai, 0)
    train_runmc_list = to_labeled_list(train_runmc_monai, 1)
    
    # Simple Oversampling for Class Balance
    n_bidmc = len(train_bidmc_list)
    n_runmc = len(train_runmc_list)
    
    if n_bidmc > 0 and n_runmc > 0:
        if n_bidmc < n_runmc:
            # Repeat BIDMC
            factor = n_runmc // n_bidmc
            remainder = n_runmc % n_bidmc
            train_bidmc_list = train_bidmc_list * factor + train_bidmc_list[:remainder]
            print(f"Oversampling BIDMC: {n_bidmc} -> {len(train_bidmc_list)} to match RUNMC ({n_runmc})")
        elif n_runmc < n_bidmc:
             # Repeat RUNMC
            factor = n_bidmc // n_runmc
            remainder = n_bidmc % n_runmc
            train_runmc_list = train_runmc_list * factor + train_runmc_list[:remainder]
            print(f"Oversampling RUNMC: {n_runmc} -> {len(train_runmc_list)} to match BIDMC ({n_bidmc})")
            
    train_list.extend(train_bidmc_list)
    train_list.extend(train_runmc_list)
    train_dataset = SimpleDataset(train_list)
    
    val_list = []
    val_list.extend(to_labeled_list(val_bidmc_monai, 0))
    val_list.extend(to_labeled_list(val_runmc_monai, 1))
    val_dataset = SimpleDataset(val_list)
    
    test_list = []
    test_list.extend(to_labeled_list(test_bidmc_monai, 0))
    test_list.extend(to_labeled_list(test_runmc_monai, 1))
    test_dataset = SimpleDataset(test_list)
    
    print(f"Training set: {len(train_dataset)} slices")
    print(f"Validation set: {len(val_dataset)} slices")
    print(f"Testing set: {len(test_dataset)} slices")
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Calculate Class Weights for Training
    n_bidmc_train = sum(1 for _, d in train_list if d == 0)
    n_runmc_train = sum(1 for _, d in train_list if d == 1)
    
    print(f"Training Class Distribution (Slices): BIDMC={n_bidmc_train}, RUNMC={n_runmc_train}")
    
    if n_bidmc_train > 0 and n_runmc_train > 0:
        # Since we balanced the dataset, we can use equal weights or None
        # But let's keep it 1.0/1.0 to be explicit
        weights = torch.tensor([1.0, 1.0], dtype=torch.float)
        print(f"Using Balanced Weights: BIDMC={weights[0]:.4f}, RUNMC={weights[1]:.4f}")
    else:
        weights = None
        print("Warning: One class missing in training, skipping weights.")
    
    # Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if weights is not None:
        weights = weights.to(device)
    
    model = SimpleDiscriminator().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    epochs = 20
    print("\nStarting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {epoch_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
    print("Training finished.")
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels_list = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())
            
    # Calculate metrics manually
    correct_preds = sum(1 for p, l in zip(all_preds, all_labels_list) if p == l)
    total_samples = len(all_labels_list)
    acc = correct_preds / total_samples
    
    # Confusion Matrix manually
    # 0: BIDMC, 1: RUNMC
    tp = sum(1 for p, l in zip(all_preds, all_labels_list) if p == 1 and l == 1) # RUNMC correctly predicted
    tn = sum(1 for p, l in zip(all_preds, all_labels_list) if p == 0 and l == 0) # BIDMC correctly predicted
    fp = sum(1 for p, l in zip(all_preds, all_labels_list) if p == 1 and l == 0) # BIDMC predicted as RUNMC
    fn = sum(1 for p, l in zip(all_preds, all_labels_list) if p == 0 and l == 1) # RUNMC predicted as BIDMC
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    print("\nDiscriminator Evaluation Results:")
    print("-" * 30)
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"True BIDMC (0) | True RUNMC (1)")
    print(f"{cm}")
    print("\nInterpretation:")
    print(f"Predicted BIDMC: {cm[:, 0]}")
    print(f"Predicted RUNMC: {cm[:, 1]}")
    
    if acc > 0.7:
        print("\nCONCLUSION: The discriminator can easily distinguish between BIDMC and RUNMC labels.")
        print("This confirms a significant distribution shift (Domain Shift) in the labels.")
    else:
        print("\nCONCLUSION: The discriminator struggles to distinguish the domains.")
        print("The label distributions might be similar after transformation.")

if __name__ == "__main__":
    main()
