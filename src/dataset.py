import os
import shutil
import random
from pathlib import Path

def create_stratified_split(raw_dir, processed_dir, seed=42, train_ratio=0.7, val_ratio=0.15):
    random.seed(seed)
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    
    # If 'processed' exists as a file or folder, clean it up to ensure clean directory creation
    if processed_path.is_file():
        processed_path.unlink()
    processed_path.mkdir(parents=True, exist_ok=True)
    
    classes = [d.name for d in raw_path.iterdir() if d.is_dir()]
    print(f"Found {len(classes)} classes in '{raw_dir}'.")

    for cls in classes:
        cls_images = list((raw_path / cls).glob("*.JPG")) + list((raw_path / cls).glob("*.jpg")) + list((raw_path / cls).glob("*.png"))
        random.shuffle(cls_images)
        
        n_total = len(cls_images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits = {
            'train': cls_images[:n_train],
            'val': cls_images[n_train:n_train + n_val],
            'test': cls_images[n_train + n_val:]
        }
        
        for split_name, files in splits.items():
            split_cls_dir = processed_path / split_name / cls
            split_cls_dir.mkdir(parents=True, exist_ok=True)
            for file_p in files:
                shutil.copy(file_p, split_cls_dir / file_p.name)
                
    print("Dataset split successfully created in 'data/processed/'.")

if __name__ == "__main__":
    RAW_DATA = "data/raw/plantvillage"
    PROCESSED_DATA = "data/processed"
    if os.path.exists(RAW_DATA):
        create_stratified_split(RAW_DATA, PROCESSED_DATA)
    else:
        print(f"Directory '{RAW_DATA}' not found. Ensure raw class folders are placed there.")