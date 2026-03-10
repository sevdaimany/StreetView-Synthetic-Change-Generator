import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

from pathlib import Path

def quick_dataset_summary(output_dir):
    base_out = Path(output_dir)
    splits = ["train", "val", "test"]
    
    print("\n" + "="*40)
    print("📊 QUICK DATASET SUMMARY")
    print("="*40)
    for split in splits:
        json_count = len(list((base_out / split / "meta").glob("*.json")))
        img_count = len(list((base_out / split / "images").glob("*.png"))) + len(list((base_out / split / "images").glob("*.jpg")))
        lbl_count = len(list((base_out / split / "labels").glob("*.txt")))
        mask_count = len(list((base_out / split / "masks").glob("*.png")))
        
        print(f"[{split.upper():>5}] Pairs (JSONs): {json_count}")
        print(f"        -> Images: {img_count} | Labels: {lbl_count} | Masks: {mask_count}\n")

def format_and_split_dataset(data_base_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Shuffles sequences and routes files into a flat train/val/test structure 
    separated by images, labels, and masks.
    """
    data_path = Path(data_base_dir)
    sequences = [d for d in data_path.iterdir() if d.is_dir()]
    
    random.seed(42)
    random.shuffle(sequences)
    
    total_seqs = len(sequences)
    train_end = int(total_seqs * train_ratio)
    val_end = train_end + int(total_seqs * val_ratio)
    
    splits = {
        "train": sequences[:train_end],
        "val": sequences[train_end:val_end],
        "test": sequences[val_end:]
    }
    
    base_out = Path(output_dir)
    categories = ["images", "labels", "masks", "meta"]
    
    for category in categories:
        for split in ["train", "val", "test"]:
            (base_out / split / category).mkdir(parents=True, exist_ok=True)
            
    print(f"Created dataset structure in {output_dir}")
            
    for split_name, seq_folders in splits.items():
        for seq_folder in tqdm(seq_folders):
            seq_id = seq_folder.name
            
            for file_path in seq_folder.iterdir():
                if not file_path.is_file():
                    continue

                filename = file_path.name
                new_filename = f"{seq_id}_{filename}"
                
                if filename.endswith(".txt"):
                    dest = base_out  / split_name / "labels" / new_filename
                    
                elif filename.endswith(".json"):
                    dest = base_out / split_name / "meta" / new_filename
                    
                elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    if "mask" in filename.lower():
                        dest = base_out / split_name / "masks" / new_filename
                    else:
                        if "after" in file_path.name.lower():
                            continue
                        dest = base_out / split_name / "images" / new_filename
                else:
                    continue 
                shutil.copyfile(file_path, dest)
                
    print("\nDataset generation complete! Ready for training.")
    quick_dataset_summary(DESTINATION)

if __name__ == "__main__":
    SOURCE_DATA = "/mnt/stores/store-DAI/pocs/simany/results_pipeline/V2.3/data" 
    DESTINATION = "/mnt/stores/store-DAI/pocs/simany/results_pipeline/training_data_V2.3" 
    
    format_and_split_dataset(SOURCE_DATA, DESTINATION)