import os
import json
import glob
import shutil
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd



def create_output_dirs(cfg):
    """Utility to create output directories if they don't exist."""
    os.makedirs(cfg.output.base, exist_ok=True)
    os.makedirs(os.path.join(cfg.output.base, cfg.output.segmentation_overlay), exist_ok=True)
    os.makedirs(os.path.join(cfg.output.base, cfg.output.inpainting_results), exist_ok=True)
    os.makedirs(os.path.join(cfg.output.base, cfg.output.edge_detection_results), exist_ok=True)
    os.makedirs(os.path.join(cfg.output.base, cfg.output.red_herring_results), exist_ok=True)
    os.makedirs(os.path.join(cfg.output.base, cfg.output.depth_results), exist_ok=True)
    os.makedirs(os.path.join(cfg.output.base, cfg.output.inpaited_only_results), exist_ok=True)
    os.makedirs(os.path.join(cfg.output.base, cfg.output.augmentation), exist_ok=True)
    os.makedirs(os.path.join(cfg.output.base, cfg.output.augmentation_masks), exist_ok=True)
    os.makedirs(os.path.join(cfg.output.base, cfg.output.correspondence_visualization), exist_ok=True)
    os.makedirs(os.path.join(cfg.output.base, cfg.output.production_ready), exist_ok=True)


def load_image(image_path, cfg):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((cfg.input.resize_width, cfg.input.resize_height), Image.Resampling.LANCZOS)
    return img

  
def create_qc_grid(images, labels, font_path,padding=20):
    w, h = images[0].size
    n = len(images)
    total_width = w * n + padding * (n - 1)
    grid = Image.new('RGB', (total_width, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)

    font_size = int(h * 0.05)
    font = ImageFont.truetype(font_path, font_size)

    for i, img in enumerate(images):
        x_offset = i * (w + padding)
        grid.paste(img, (x_offset, 0))

        draw.rectangle(
            [x_offset, h - font_size - 15, x_offset + w, h],
            fill=(0, 0, 0)
        )
        draw.text(
            (x_offset + 10, h - font_size - 10),
            labels[i],
            fill=(255, 255, 255),
            font=font)
    return grid

def get_bbox_from_mask(mask_array):
    """Finds the Pascal VOC bounding box [xmin, ymin, xmax, ymax] of a binary mask."""
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None 

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    return [int(xmin), int(ymin), int(xmax), int(ymax)]


def save_voc_bboxes_and_overlay(image_pil, instances, mask_key, class_name, txt_path, overlay_path):
    """
    Extracts BBoxes, saves them to a VOC .txt file, and saves a visual overlay.
    
    Args:
        image_pil: The original PIL Image.
        instances: List of dictionaries containing the masks.
        mask_key: String key to access the mask (e.g., 'before_mask' or 'after_mask').
        class_name: The string name of the class (e.g., 'dog').
        txt_path: Where to save the .txt file.
        overlay_path: Where to save the visual QC image.
    """
    voc_lines = []
    
    overlay_img = image_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay_img)
    
    for instance in instances:
        mask = np.array(instance[mask_key]) > 0
        bbox = get_bbox_from_mask(mask)
        
        if bbox:
            xmin, ymin, xmax, ymax = bbox
            
            # 1. Format for text file: class_name xmin ymin xmax ymax
            voc_lines.append(f"{class_name} {xmin} {ymin} {xmax} {ymax}")
            
            # 2. Draw on the image (Red rectangle, 3 pixels thick)
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
            
            # Draw the class name just above the box (with a tiny background for readability)
            text_y = max(0, ymin - 15)
            draw.rectangle([xmin, text_y, xmin + len(class_name)*6, text_y + 15], fill="red")
            draw.text((xmin + 2, text_y), class_name, fill="white")
            
    with open(txt_path, "w") as f:
        f.write("\n".join(voc_lines))
        
    overlay_img.save(overlay_path, quality=90)

def check_redundancy(sequence_id, class_name, img_name1, img_name2, cfg):
    base_dir = os.path.join(cfg.output.base, cfg.output.production_ready, "data", sequence_id)
    safe_class = class_name.replace(" ", "_")
    pair_id = f"{img_name1.split('.')[0]}_to_{img_name2.split('.')[0]}_{safe_class}"
    check_path = os.path.join(base_dir, f"{pair_id}_meta.json")
    return os.path.exists(check_path)



def collect_metadata_for_all_pairs(cfg):

    metas = [json.load(open(f)) for f in glob.glob("results/production_ready/**/meta.json", recursive=True)]
    df = pd.DataFrame(metas)
    print(df['class_name'].value_counts())

