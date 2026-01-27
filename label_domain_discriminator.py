import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix
import random

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)

class LabelDomainDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels # 0 for BIDMC, 1 for RUNMC
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        domain_label = self.labels[idx]
        
        try:
            img = nib.load(path)
            data = img.get_fdata()
            data = np.round(data).astype(int)
            
            # Apply Mask2To1d transform logic
            data[data == 2] = 1
            
            # Extract middle slice to represent the volume
            # This simplifies 3D to 2D for quick discrimination
            z_center = data.shape[2] // 2
            slice_data = data[..., z_center]
            
            # Resize to 128x128 using simple interpolation (nearest for masks)
            # using torch for resizing
            tensor = torch.from_numpy(slice_data).float().unsqueeze(0) # [C, H, W]
            
            # Resize
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=(128, 128), mode='nearest'
            ).squeeze(0)
            
            # Normalize to 0-1 (it's already 0/1 but ensure float)
            tensor = tensor / (tensor.max() + 1e-8)
            
            return tensor, torch.tensor(domain_label, dtype=torch.long)
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros((1, 128, 128)), torch.tensor(domain_label, dtype=torch.long)

class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 64x64
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32x32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2) # 2 classes: BIDMC vs RUNMC
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_all_label_paths(base_dir, dataset_name):
    # Search in train, val, and test
    paths = []
    for split in ['train', 'val', 'test']:
        search_path = os.path.join(base_dir, dataset_name, split, "*.nii.gz")
        found = glob.glob(search_path)
        for p in found:
            filename = os.path.basename(p)
            if len(filename) > 10 and filename[7:10] in ["seg", "Seg"]:
                paths.append(p)
    return sorted(paths)

def main():
    base_dir = "/root/SLCL/Processed_data_nii_uda"
    
    # 1. Gather Data
    bidmc_paths = get_all_label_paths(base_dir, "BIDMC")
    runmc_paths = get_all_label_paths(base_dir, "RUNMC")
    
    print(f"Total BIDMC labels: {len(bidmc_paths)}")
    print(f"Total RUNMC labels: {len(runmc_paths)}")
    
    # Create labels: 0 for BIDMC, 1 for RUNMC
    all_paths = bidmc_paths + runmc_paths
    all_domain_labels = [0] * len(bidmc_paths) + [1] * len(runmc_paths)
    
    # Split into Train/Test manually since sklearn is missing
    # Shuffle first
    combined = list(zip(all_paths, all_domain_labels))
    random.shuffle(combined)
    all_paths, all_domain_labels = zip(*combined)
    all_paths = list(all_paths)
    all_domain_labels = list(all_domain_labels)
    
    test_size = 0.3
    split_idx = int(len(all_paths) * (1 - test_size))
    
    X_train = all_paths[:split_idx]
    X_test = all_paths[split_idx:]
    y_train = all_domain_labels[:split_idx]
    y_test = all_domain_labels[split_idx:]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # Calculate Class Weights to handle imbalance
    n_bidmc = y_train.count(0)
    n_runmc = y_train.count(1)
    print(f"Training Class Distribution: BIDMC={n_bidmc}, RUNMC={n_runmc}")
    
    if n_bidmc > 0 and n_runmc > 0:
        w_bidmc = 1.0 / n_bidmc
        w_runmc = 1.0 / n_runmc
        # Normalize to sum to 2 roughly or just pass as is
        weights = torch.tensor([w_bidmc, w_runmc], dtype=torch.float)
        weights = weights / weights.sum() * 2.0 # Scale so mean is 1
        print(f"Using Class Weights: BIDMC={weights[0]:.4f}, RUNMC={weights[1]:.4f}")
    else:
        weights = None
        print("Warning: One class missing in training, skipping weights.")
    
    # Datasets and Loaders
    train_dataset = LabelDomainDataset(X_train, y_train)
    test_dataset = LabelDomainDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Model Setup
    # Force CPU to avoid OOM
    device = torch.device("cpu")
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
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {epoch_acc:.2f}%")
        
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
