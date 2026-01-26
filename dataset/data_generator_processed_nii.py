import os
import random
from glob import glob
from torch.utils.data import DataLoader, Dataset, IterableDataset
from monai.data import CacheDataset, PatchDataset
from data.transform import (
    volume_transform,
    slice_transform_train,
    slice_transform_valid,
    FilterSliced,
)

class DictToTupleWrapper(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __iter__(self):
        for d in self.dataset:
            # Trainer expects: img, label, name
            # Return a dummy name as it is not strictly used in training loop logic of MPSCL
            yield d["image"], d["label"], "unknown"
            
    def __len__(self):
        return len(self.dataset)

def build_dataset_internal(image_set, args):
    assert os.path.exists(
        args.data_dir
    ), f"provided data path {args.data_dir} does not exist"

    is_uda = getattr(args, "uda", False)

    if image_set == "target":
        # Processed_data_nii_uda/BIDMC/train/*.nii.gz
        file_paths = glob(os.path.join(args.data_dir, "BIDMC", "train", "*.nii.gz"))
    elif is_uda and image_set == "val":
        file_paths = glob(os.path.join(args.data_dir, "BIDMC", "val", "*.nii.gz"))
    elif is_uda and image_set == "test":
        file_paths = glob(os.path.join(args.data_dir, "BIDMC", "test", "*.nii.gz"))
    else:
        # Processed_data_nii_uda/RUNMC/[image_set]/*.nii.gz
        file_paths = glob(os.path.join(args.data_dir, "RUNMC", image_set, "*.nii.gz"))

    if not file_paths:
        print(f"Warning: No files found for image_set={image_set} in {args.data_dir}")

    image_paths, label_paths = [], []
    for path in file_paths:
        # Check logic from data/dataset.py: 
        # if path.split("/")[-1][7:10] in ["seg", "Seg"]:
        # But here we should check how the files are named in the new dataset.
        # Assuming standard structure where labels might be distinguished.
        # However, data/dataset.py logic relies on specific naming (7:10).
        # Let's assume the user follows the same convention or the files are separate.
        # Wait, glob returns ALL files. We need to separate images and labels.
        
        # In data/dataset.py:
        # if path.split("/")[-1][7:10] in ["seg", "Seg"]:
        #    label_paths.append(path)
        # else:
        #    image_paths.append(path)
        
        filename = os.path.basename(path)
        # Simple heuristic: if 'seg' or 'label' in filename, it's a label.
        # But let's try to stick to data/dataset.py logic if possible, 
        # or be more robust.
        
        # data/dataset.py logic: path.split("/")[-1][7:10]
        # e.g. "Case00_seg.nii.gz" -> index 7 is 's', 7:10 is 'seg'.
        # e.g. "Case00.nii.gz" -> index 7 is '.', 7:10 is '.ni'.
        
        if "seg" in filename.lower() or "label" in filename.lower():
            label_paths.append(path)
        else:
            image_paths.append(path)

    image_paths, label_paths = sorted(image_paths), sorted(label_paths)
    
    # Verify pairing
    if len(image_paths) != len(label_paths):
        print(f"Warning: Number of images ({len(image_paths)}) and labels ({len(label_paths)}) do not match in {image_set}")
    
    path_dicts = [
        {"image": image_path, "label": label_path, "ori_image": image_path}
        for image_path, label_path in zip(image_paths, label_paths)
    ]

    # split train and val set
    if image_set == "train":
        random.shuffle(path_dicts)
        slice_transform = slice_transform_train
    elif image_set == "val":
        slice_transform = slice_transform_valid
    elif image_set == "test":
        slice_transform = slice_transform_valid    
    elif image_set == "target":
        random.shuffle(path_dicts)
        slice_transform = slice_transform_train
    else:
        # Fallback
        slice_transform = slice_transform_train

    dataset = CacheDataset(
        data=path_dicts, transform=volume_transform, cache_rate=1.0, num_workers=4
    )
    slice_sampler = FilterSliced(
        ["image", "label", "ori_image"], source_key="label", samples_per_image=12
    )
    slice_dataset = PatchDataset(dataset, slice_sampler, 12, slice_transform)
    return slice_dataset

def prepare_dataset(args):
    # content_loader (Source)
    source_dataset = build_dataset_internal("train", args)
    source_wrapper = DictToTupleWrapper(source_dataset)
    
    # style_loader (Target)
    target_dataset = build_dataset_internal("target", args)
    target_wrapper = DictToTupleWrapper(target_dataset)
    
    content_loader = DataLoader(
        source_wrapper, 
        batch_size=args.bs, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    style_loader = DataLoader(
        target_wrapper, 
        batch_size=args.bs, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    # Return scratch dirs as data_dir since we don't use scratch space logic here
    return args.data_dir, args.data_dir, content_loader, style_loader
