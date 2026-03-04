import os
os.environ['TRITON_CACHE_DIR'] = os.path.join(os.getcwd(), '.triton_cache')
import hydra
import logging
import random
import numpy as np
from DatasetGenerator import DatasetGenerator
from dotenv import load_dotenv
from huggingface_hub import login
from omegaconf import OmegaConf, DictConfig
from SAM3Correspondence import SAM3CorrespondencePipeline
import json
from PIL import Image
from tqdm import tqdm
import traceback
from utils import *

load_dotenv()
login(os.getenv("HF_TOKEN"))
logger = logging.getLogger(__name__)




def log_config(cfg):
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    if "FLUX" in cfg.model.inpainting:
        logger.info("Using FLUX for inpainting")
    elif "xl" in cfg.model.inpainting:
        logger.info("Using Stable Diffusion XL")
    else:
        logger.info("Using Stable Diffusion")

def process_and_save_synthetic_change(
    generator, 
    cfg, 
    sequence_id, 
    img1, 
    img2, 
    img_name1, 
    img_name2, 
    selected_instances, 
    prompt_seg, 
    prompt_inpaint):
    """
    Extracts masks, runs the inpainting generator, creates visual overlays, 
    and saves the complete before/after pipeline to disk.
    """
    # 1. EXTRACT AND MERGE BOTH 'BEFORE' AND 'AFTER' MASKS
    H, W = selected_instances[0]["after_mask"].shape
    merged_after = np.zeros((H, W), dtype=bool)
    merged_before = np.zeros((H, W), dtype=bool)
    
    for instance in selected_instances:
        current_after = np.array(instance["after_mask"]) > 0
        merged_after = np.logical_or(merged_after, current_after)
        
        current_before = np.array(instance["before_mask"]) > 0
        merged_before = np.logical_or(merged_before, current_before)
    
    mask_after_np = (merged_after * 255).astype(np.uint8)
    mask_before_np = (merged_before * 255).astype(np.uint8)
    
    mask_after_pil = Image.fromarray(mask_after_np, mode="L")
    mask_before_pil = Image.fromarray(mask_before_np, mode="L")

    # 2. PASS TO INPAINTING
    inpainted_image, _ = generator.inference(
        img=img2, 
        image_name=img_name2, 
        prompt_seg=prompt_seg, 
        prompt_inpaint=prompt_inpaint,
        seg_mask=mask_after_pil, 
        save_all=False 
    )
    # Standardize sizes
    target_size = img1.size
    inpainted_image = inpainted_image.resize(target_size, Image.Resampling.LANCZOS)
    mask_after_pil = mask_after_pil.resize(target_size, Image.Resampling.NEAREST)
    mask_before_pil = mask_before_pil.resize(target_size, Image.Resampling.NEAREST)
    mask_after_np = np.array(mask_after_pil)
    mask_before_np = np.array(mask_before_pil)
    
    # 3. GENERATE OVERLAYS
    overlay_img1 = generator.overlay_mask(img1, mask_before_np)
    overlay_img2 = generator.overlay_mask(img2, mask_after_np)
    overlay_inpainted = generator.overlay_mask(inpainted_image, mask_after_np)

    # 2. DEFINE DIRECTORY STRUCTURE
    # Separate 'raw' data for training and 'viz' for human checking
    base_dir = os.path.join(cfg.input.project_path, cfg.output.base, cfg.output.production_ready)
    viz_dir = os.path.join(base_dir, "visualizations")
    data_dir = os.path.join(base_dir, "data", sequence_id)
    
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    safe_class = prompt_seg.replace(" ", "_")
    pair_id = f"{img_name1.split('.')[0]}_to_{img_name2.split('.')[0]}_{safe_class}"

    # 3. SAVE TRAINING ASSETS
    # We save only the essentials for the model here
    paths = {
        "img_before": os.path.join(data_dir, f"{pair_id}_before.png"),
        "img_after": os.path.join(data_dir, f"{pair_id}_after.png"),
        "mask_before": os.path.join(data_dir, f"{pair_id}_mask_before.png"),
        "mask_after": os.path.join(data_dir, f"{pair_id}_mask_after.png"),
        "inpainted": os.path.join(data_dir, f"{pair_id}_inpainted.png"),
        "txt_before": os.path.join(data_dir, f"{pair_id}_before.txt"),
        "txt_after": os.path.join(data_dir, f"{pair_id}_after.txt"),
    }
    
    img1.save(paths["img_before"])
    img2.save(paths["img_after"])
    mask_before_pil.save(paths["mask_before"])
    mask_after_pil.save(paths["mask_after"])
    inpainted_image.save(paths["inpainted"])
    viz_bbox_before = os.path.join(viz_dir, f"{pair_id}_bbox_before.jpg")
    viz_bbox_after = os.path.join(viz_dir, f"{pair_id}_bbox_after.jpg")
    save_voc_bboxes_and_overlay(image_pil=img1, instances=selected_instances, mask_key="before_mask", 
        class_name=prompt_seg, txt_path=paths["txt_before"], overlay_path=viz_bbox_before)
    save_voc_bboxes_and_overlay(image_pil=img2, instances=selected_instances, mask_key="after_mask", 
        class_name=prompt_seg, txt_path=paths["txt_after"], overlay_path=viz_bbox_after)


    # 4. SAVE METADATA
    metadata = {
        "pair_id": pair_id,
        "sequence_id": sequence_id,
        "class_name": prompt_seg,
        "prompt": prompt_inpaint,
        "num_instances": len(selected_instances),
        "img1_name": img_name1,
        "img2_name": img_name2,
        "paths": paths
    }
    
    json_path = os.path.join(data_dir, f"{pair_id}_meta.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    grid = create_qc_grid([overlay_img1, overlay_img2, inpainted_image], 
                          labels=["Before overlay", "After overlay", "Inpainted"], font_path=cfg.input.font_path)
    grid.save(os.path.join(viz_dir, f"{pair_id}_QC.jpg"), quality=85) # JPG saves space for viz


def filter_cracks_by_roads(target_matches, bg_matches):
    """
    Filters cracks instances to only keep the pixels that overlap 
    with the road instances (background).
    
    Returns: A filtered list of matches where the target is strictly inside the background.
    """
    if not target_matches or not bg_matches:
        return [] 

    # 1. Create a global boolean mask for the background class
    H, W = np.array(target_matches[0]["before_mask"]).shape
    global_bg_before = np.zeros((H, W), dtype=bool)
    global_bg_after = np.zeros((H, W), dtype=bool)
    
    for bg_inst in bg_matches:
        global_bg_before |= (np.array(bg_inst["before_mask"]) > 0)
        global_bg_after |= (np.array(bg_inst["after_mask"]) > 0)

    filtered_matches = []
    
    # 2. Apply the logical AND to every target instance
    for match in target_matches:
        target_b = np.array(match["before_mask"]) > 0
        target_a = np.array(match["after_mask"]) > 0
        
        # The intersection (AND operation)
        filtered_b = np.logical_and(target_b, global_bg_before)
        filtered_a = np.logical_and(target_a, global_bg_after)
        
        # 3. Keep the instance only if some pixels survived the filter
        if np.any(filtered_b) or np.any(filtered_a):
            match["before_mask"] = (filtered_b * 255).astype(np.uint8)
            match["after_mask"] = (filtered_a * 255).astype(np.uint8)
            filtered_matches.append(match)
            
    return filtered_matches

def process_sequence(sequence_id, base_path, classes, class_to_prompt, sam_pipeline, generator, cfg):
    sequence_path = os.path.join(base_path, sequence_id)
    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = sorted([f for f in os.listdir(sequence_path) if f.lower().endswith(valid_extensions)])
    selection_mode = cfg.input.get("mask_selection_mode", "single")
    

    logger.info(f"------------------- Processing sequence {sequence_id}.")
    for i in range(len(image_files) - 1):
        img_name1, img_name2 = image_files[i], image_files[i+1]
        
        try:
            img1 = load_image(os.path.join(sequence_path, img_name1), cfg)
            img2 = load_image(os.path.join(sequence_path, img_name2), cfg)
            sam_pipeline.load_image_pair(img1, img2)
            
            for class_name in classes:
                prompt_seg = class_name
                safe_class = prompt_seg.replace(" ", "")
                prompt_inpaint = class_to_prompt[class_name]
                
                # RESTART LOGIC: Check if work is already done
                if check_redundancy(sequence_id, class_name, img_name1, img_name2, cfg):
                    logger.info(f"Skipping {class_name} for {img_name1} -> {img_name2} (already processed)")
                    continue

                # Core Processing
                matches = sam_pipeline.track_class(prompt_seg)
                if len(matches) == 0:
                    logger.warning(f"No {prompt_seg} matches found. Skipping.")
                    continue
                
                # Apply the Road Filter if this is a crack class
                if "cracks" in prompt_seg.lower():
                    road_matches = sam_pipeline.track_class("road")
                    if len(road_matches) == 0:
                        logger.warning(f"No road found to filter {prompt_seg}. Skipping this pair.")
                        continue
                        
                    matches = filter_cracks_by_roads(matches, road_matches)
                    
                    if len(matches) == 0:
                        logger.warning(f"All {prompt_seg} matches were outside the road. Skipping.")
                        continue

                # save_path = os.path.join(cfg.input.project_path, cfg.output.correspondence_visualization, f"{prompt_seg}_{img_name1.split('.')[0]}_{img_name2.split('.')[0]}.png")
                # sam_pipeline.visualize_correspondence(img1, img2, matches, save_path=save_path)

                # Selection logic...
                if selection_mode == "biggest":
                    selected_instances = [max(matches, key=lambda x: np.sum(x["after_mask"]))]
                else:
                    num_to_select = 1 if selection_mode == "single" else (len(matches) if selection_mode == "all" else random.randint(1, len(matches)))
                    selected_instances = random.sample(matches, k=num_to_select)

                process_and_save_synthetic_change(
                    generator=generator,
                    cfg=cfg,
                    sequence_id=sequence_id,
                    img1=img1,
                    img2=img2,
                    img_name1=img_name1,
                    img_name2=img_name2,
                    selected_instances=selected_instances,
                    prompt_seg=prompt_seg,
                    prompt_inpaint=prompt_inpaint
                )
                logger.info(f"Successfully processed {class_name} | {img_name1}")

        except Exception as e:
            logger.error(f"Error processing pair {img_name1}-{img_name2}: {e}")
            logger.error(traceback.format_exc())
        finally:
            sam_pipeline.clear_current_pair()

@hydra.main(config_path=".", config_name="config")
def run(cfg: DictConfig):
    log_config(cfg)
    create_output_dirs(cfg)
    
    generator = DatasetGenerator(cfg)
    sam_pipeline = SAM3CorrespondencePipeline(device="cuda")
    class_to_prompt = cfg.input.get("prompts_seg", {})   
    classes = list(class_to_prompt.keys())    

    sequence_base_path = os.path.join(cfg.input.project_path, cfg.input.image_folder)
    
    sequences = [d for d in os.listdir(sequence_base_path) if os.path.isdir(os.path.join(sequence_base_path, d))]

    for sequence_id in tqdm(sequences, desc="Sequences"):
        try:
            process_sequence(sequence_id, sequence_base_path, classes, class_to_prompt, sam_pipeline, generator, cfg)
        except Exception as e:
            logger.error(f"CRITICAL FAILURE in sequence {sequence_id}: {e}")
            logger.error(traceback.format_exc())
            # Continue to next sequence despite failure in current one
            continue
            
    sam_pipeline.shutdown()


if __name__ == "__main__":
    run()