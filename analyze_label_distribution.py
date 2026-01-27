import os
import glob
import numpy as np
import nibabel as nib
from collections import Counter
import sys

def get_label_paths(base_dir, dataset_name, split):
    search_path = os.path.join(base_dir, dataset_name, split, "*.nii.gz")
    file_paths = glob.glob(search_path)
    label_paths = []
    for path in file_paths:
        filename = os.path.basename(path)
        # Logic from dataset.py: if path.split("/")[-1][7:10] in ["seg", "Seg"]:
        # CaseXX_segmentation.nii.gz -> filename[7:10] is "seg"
        if len(filename) > 10 and filename[7:10] in ["seg", "Seg"]:
            label_paths.append(path)
    return sorted(label_paths)

def analyze_labels(label_paths, dataset_name, apply_transform=True):
    print(f"Analyzing {dataset_name} ({len(label_paths)} files)...")
    total_counts = Counter()
    foreground_ratios = [] # Store foreground ratio per sample
    
    for path in label_paths:
        try:
            img = nib.load(path)
            data = img.get_fdata()
            data = np.round(data).astype(int)
            
            # Apply Mask2To1d equivalent logic
            if apply_transform:
                data[data == 2] = 1
            
            # Calculate per-sample stats
            total_pixels_sample = data.size
            fg_pixels_sample = np.sum(data == 1)
            if total_pixels_sample > 0:
                foreground_ratios.append(fg_pixels_sample / total_pixels_sample)
            
            unique, counts = np.unique(data, return_counts=True)
            for u, c in zip(unique, counts):
                total_counts[u] += c
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
    print(f"Distribution for {dataset_name} (Transformed: {apply_transform}):")
    total_pixels = sum(total_counts.values())
    
    for label, count in sorted(total_counts.items()):
        percentage = (count / total_pixels) * 100
        print(f"  Label {label}: {count} pixels ({percentage:.2f}%)")
        
    # Calculate statistics for foreground ratios
    if foreground_ratios:
        mean_ratio = np.mean(foreground_ratios) * 100
        std_ratio = np.std(foreground_ratios) * 100
        print(f"  Foreground Ratio per Sample: Mean = {mean_ratio:.2f}%, Std = {std_ratio:.2f}%")
        return total_counts, mean_ratio, std_ratio
    
    return total_counts, 0, 0

def main():
    base_dir = "/root/SLCL/Processed_data_nii_uda"
    
    bidmc_labels = get_label_paths(base_dir, "BIDMC", "test")
    runmc_labels = get_label_paths(base_dir, "RUNMC", "test")
    
    print(f"Found {len(bidmc_labels)} labels for BIDMC test")
    print(f"Found {len(runmc_labels)} labels for RUNMC test")
    
    bidmc_res = None
    runmc_res = None

    if bidmc_labels:
        bidmc_res = analyze_labels(bidmc_labels, "BIDMC", apply_transform=True)
        
    if runmc_labels:
        runmc_res = analyze_labels(runmc_labels, "RUNMC", apply_transform=True)

    if bidmc_res and runmc_res:
        bidmc_counts, b_mean, b_std = bidmc_res
        runmc_counts, r_mean, r_std = runmc_res
        
        print("\nComparison after Mask2To1d transform:")
        print(f"{'Metric':<25} {'BIDMC':<20} {'RUNMC':<20}")
        print("-" * 65)
        
        # Calculate global foreground percentage
        b_total = sum(bidmc_counts.values())
        b_fg = bidmc_counts.get(1, 0)
        b_global_pct = (b_fg / b_total * 100) if b_total > 0 else 0
        
        r_total = sum(runmc_counts.values())
        r_fg = runmc_counts.get(1, 0)
        r_global_pct = (r_fg / r_total * 100) if r_total > 0 else 0
        
        print(f"{'Global Foreground %':<25} {b_global_pct:<20.2f} {r_global_pct:<20.2f}")
        print(f"{'Sample Mean FG %':<25} {b_mean:<20.2f} {r_mean:<20.2f}")
        print(f"{'Sample Std FG %':<25} {b_std:<20.2f} {r_std:<20.2f}")


if __name__ == "__main__":
    main()
