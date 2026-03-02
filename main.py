import os
import cv2
import hydra
import torch
import logging
import random
import numpy as np
from PIL import Image
from DatasetGenerator import DatasetGenerator
from dotenv import load_dotenv
from huggingface_hub import login
from omegaconf import OmegaConf, DictConfig
from Panorama import Panorama
from SAM3Correspondence import SAM3CorrespondencePipeline
from ProceduralPromptGenerator import ProceduralPromptGenerator
load_dotenv()
login(os.getenv("HF_TOKEN"))
logg = logging.getLogger(__name__)


def create_output_dirs(cfg):
    """Utility to create output directories if they don't exist."""
    os.makedirs(os.path.join(cfg.input.project_path, cfg.output.segmentation_overlay), exist_ok=True)
    os.makedirs(os.path.join(cfg.input.project_path, cfg.output.inpainting_results), exist_ok=True)
    os.makedirs(os.path.join(cfg.input.project_path, cfg.output.edge_detection_results), exist_ok=True)
    os.makedirs(os.path.join(cfg.input.project_path, cfg.output.red_herring_results), exist_ok=True)
    os.makedirs(os.path.join(cfg.input.project_path, cfg.output.depth_results), exist_ok=True)
    os.makedirs(os.path.join(cfg.input.project_path, cfg.output.inpaited_only_results), exist_ok=True)
    os.makedirs(os.path.join(cfg.input.project_path, cfg.output.augmentation), exist_ok=True)
    os.makedirs(os.path.join(cfg.input.project_path, cfg.output.augmentation_masks), exist_ok=True)
    os.makedirs(os.path.join(cfg.input.project_path, cfg.output.correspondence_visualization), exist_ok=True)
    os.makedirs(os.path.join(cfg.input.project_path, cfg.output.production_ready), exist_ok=True)

def log_config(cfg):
    logg.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    if "FLUX" in cfg.model.inpainting:
        logg.info("Using FLUX for inpainting")
    elif "xl" in cfg.model.inpainting:
        logg.info("Using Stable Diffusion XL + ControlNet for inpainting")
    else:
        logg.info("Using Stable Diffusion + ControlNet for inpainting")


def load_image_test(image_name, cfg):
    image_path = os.path.join(cfg.input.project_path, cfg.input.image_folder, image_name)
    img = Image.open(image_path).convert("RGB")
    if 'panorama' in image_name.lower():
        img = img.resize((cfg.input.resize_width, cfg.input.resize_height), Image.BILINEAR)
    return img
    

@hydra.main(config_path=".", config_name="config")
def augmentation_withoneimage(cfg: DictConfig):
    log_config(cfg)
    create_output_dirs(cfg)
    panorama = Panorama(cfg)
    generator = DatasetGenerator(cfg)

    # input
    image_path = os.path.join(cfg.input.project_path, cfg.input.image_folder, cfg.input.image_name)
    img = Image.open(image_path).convert("RGB")
    resized_img = img.resize((cfg.input.resize_width, cfg.input.resize_height), Image.Resampling.LANCZOS)
    
    # Augmentation
    warped_img, valid_mask, depth_pil = panorama.augment(resized_img, cfg.input.image_name)

    # Refinement
    final_inpainted_img, final_sdxl_mask = generator.refine_novel_view(warped_img, valid_mask, cfg.augmentation.prompt_inpaint)
    
    base_name, _ = os.path.splitext(cfg.input.image_name)
    save_path_mask = os.path.join(cfg.input.project_path, cfg.output.augmentation_masks, f"{base_name}_z{cfg.augmentation.translation_z}_deg{cfg.augmentation.yaw_deg}.png")
    save_path = os.path.join(cfg.input.project_path, cfg.output.augmentation, f"{base_name}_z{cfg.augmentation.translation_z}_deg{cfg.augmentation.yaw_deg}.png")
    
    warped_img.save(os.path.join(cfg.input.project_path, cfg.output.augmentation, f"beforeinpaint_{base_name}_z{cfg.augmentation.translation_z}_deg{cfg.augmentation.yaw_deg}.png"))
    final_sdxl_mask.save(save_path_mask)
    final_inpainted_img.save(save_path)
    print(f"Saved augmented image to {save_path}")

@hydra.main(config_path=".", config_name="config")
def augmentation_withtwoimages(cfg: DictConfig):

    log_config(cfg)
    create_output_dirs(cfg)

    # input
    image_1, image_2 = list(cfg.input.get("image_names", [cfg.input.get("image_name")]))
    img_1 = Image.open(os.path.join(cfg.input.project_path, cfg.input.image_folder, image_1)).convert("RGB")
    img_2 = Image.open(os.path.join(cfg.input.project_path, cfg.input.image_folder, image_2)).convert("RGB")
    resized_img_1 = img_1.resize((cfg.input.resize_width, cfg.input.resize_height), Image.Resampling.LANCZOS)
    resized_img_2 = img_2.resize((cfg.input.resize_width, cfg.input.resize_height), Image.Resampling.LANCZOS)

    # 1. Initialize the class globally or at the top of your script
    sam_pipeline = SAM3CorrespondencePipeline(device="cuda")

    # 2. Load your images into the session
    sam_pipeline.load_image_pair(resized_img_1, resized_img_2)

    # 3. Query different classes (images are already cached in VRAM)
    building_matches = sam_pipeline.track_class("buildings")
    
    print(f"Found {len(building_matches)} buildings.")
    if len(building_matches) > 0:
        save_path = os.path.join(cfg.input.project_path, cfg.output.correspondence_visualization, f"building_{image_1.split('.')[0]}_{image_2.split('.')[0]}.png")
        sam_pipeline.visualize_correspondence(resized_img_1, resized_img_2, building_matches, save_path=save_path)

    sign_matches = sam_pipeline.track_class("traffic signs")
    print(f"Found {len(sign_matches)} traffic signs.")
    if len(sign_matches) > 0:
        save_path = os.path.join(cfg.input.project_path, cfg.output.correspondence_visualization, f"trafficsign_{image_1.split('.')[0]}_{image_2.split('.')[0]}.png")
        sam_pipeline.visualize_correspondence(resized_img_1, resized_img_2, sign_matches, save_path=save_path)

    road_matches = sam_pipeline.track_class("roads")
    print(f"Found {len(road_matches)} roads.")
    if len(road_matches) > 0:
        save_path = os.path.join(cfg.input.project_path, cfg.output.correspondence_visualization, f"road_{image_1.split('.')[0]}_{image_2.split('.')[0]}.png")
        sam_pipeline.visualize_correspondence(resized_img_1, resized_img_2, road_matches, save_path=save_path)

    # 4. Clean up the images when you are ready to move to the next pair
    sam_pipeline.clear_current_pair()
    # 5. Shut it down completely at the very end of your script
    sam_pipeline.shutdown()

    # later for synthetic change
    selected_instance = random.choice(building_matches)
    # 3. Extract the variables for your inpainting function
    instance_id = selected_instance["instance_id"]
    mask_1 = selected_instance["before_mask"]
    mask_2 = selected_instance["after_mask"] # Mask 1 shape: (512, 1024)    
    print(f"Selected instance ID: {instance_id} | Mask 1 shape: {mask_1.shape} | Mask 2 shape: {mask_2.shape}")


@hydra.main(config_path=".", config_name="config")
def test_per_model(cfg: DictConfig):
    """ 
    For evaluating different prompts and models on a single segmentation class.
    """
    log_config(cfg)
    create_output_dirs(cfg)
    generator = DatasetGenerator(cfg)

    # 1. Extract lists from config (fallback to defaults if they don't exist)
    image_names = list(cfg.input.get("image_names", [cfg.input.get("image_name")]))
    prompts = list(cfg.input.get("prompts_inpaint", [""]))
    neg_prompts = list(cfg.input.get("negative_prompts_inpaint", [""]))
    prompt_seg = cfg.input.get("prompt_seg", "buildings")

    # If only one negative prompt is provided but multiple positive prompts, broadcast it
    if len(neg_prompts) == 1 and len(prompts) > 1:
        neg_prompts = neg_prompts * len(prompts)

    run_mode = cfg.input.get("run_mode", "combinatorial")

    # 2. Execute based on run mode
    if run_mode == "pairwise":
        assert len(image_names) == len(prompts), "For pairwise mode, the number of images and prompts must be equal."
        for img_name, prompt, neg_prompt in zip(image_names, prompts, neg_prompts):
            img = load_image_test(img_name, cfg)
            generator.inference(img, img_name, prompt_seg, prompt, negative_prompt_inpaint=neg_prompt)

    elif run_mode == "combinatorial":
        for img_name in image_names:
            
            # Scenario A: Lengths match perfectly -> Treat them as pairs
            if len(prompts) == len(neg_prompts):
                for prompt, neg_prompt in zip(prompts, neg_prompts):
                    logg.info(f"--- Processing: {img_name} | Prompt: '{prompt}' | Negative Prompt: '{neg_prompt}' ---")
                    img = load_image_test(img_name, cfg)
                    generator.inference(img, img_name, prompt_seg, prompt, negative_prompt_inpaint=neg_prompt)
            
            # Scenario B: Lengths differ (or one is length 1) -> Iterate on BOTH (all combinations)
            else:
                for prompt in prompts:
                    for neg_prompt in neg_prompts:
                        logg.info(f"--- Processing: {img_name} | Prompt: '{prompt}' | Negative Prompt: '{neg_prompt}' ---")
                        img = load_image_test(img_name, cfg)
                        generator.inference(img, img_name, prompt_seg, prompt, negative_prompt_inpaint=neg_prompt)   
    else:
        logg.error(f"Unknown run_mode: {run_mode}")

  
@hydra.main(config_path=".", config_name="config")
def run_red_herring(cfg: DictConfig):
    log_config(cfg)
    create_output_dirs(cfg)
    generator = DatasetGenerator(cfg)
    prompt_gen = ProceduralPromptGenerator()
    class_to_prompt = cfg.input.get("prompts_seg", {})   
    classes = class_to_prompt.keys()
    logg.info(f"Classes to process: {classes}")
    logg.info(f"Class to prompt mapping: {class_to_prompt}")
    red_herring_classes = list(cfg.input.get("red_herring_classes", []))

    for img_name in os.listdir(os.path.join(cfg.input.project_path, cfg.input.image_folder)):

        img = load_image_test(img_name, cfg)

        for class_name in classes:
            print(" --------------------------------------------------------- ")

            prompt_seg = class_name
            prompt = class_to_prompt[class_name]

            if isinstance(prompt, str):
                prompt_inpaint = prompt
            else:
                prompt_inpaint = random.choice(prompt)
            logg.info(f"Processing {img_name} for class '{prompt_seg}' with prompt '{prompt_inpaint}'")
            
            inpainted_image, selected_mask = generator.inference(img, img_name, prompt_seg=prompt_seg, prompt_inpaint=prompt_inpaint, save_all=False)

            if cfg.input.red_herring:
                red_herring_class = random.choice(red_herring_classes)
                red_herring_prompt = prompt_gen.get_prompt(red_herring_class)
                logg.info(f"Adding red herring for class '{red_herring_class}' with prompt '{red_herring_prompt}'")
                inpainted_image_red_herring, selected_mask_red_herring  = generator.inference(inpainted_image, img_name, prompt_seg=red_herring_class, prompt_inpaint=red_herring_prompt, save_all=False)
                


                if  selected_mask_red_herring is None:
                    # should save the inpainted and mask for no change samples, but skipping for now
                    logg.warning(f"Skipping red herring overlay for {img_name} due to missing masks.")
                    continue

                if selected_mask:
                    selected_mask = np.array(selected_mask)
                    selected_mask_red_herring = np.array(selected_mask_red_herring)
                    selected_mask = np.squeeze(selected_mask)
                    selected_mask_red_herring = np.squeeze(selected_mask_red_herring)
                    if selected_mask.shape != selected_mask_red_herring.shape:
                        selected_mask_red_herring = cv2.resize(
                            selected_mask_red_herring,
                            (selected_mask.shape[1], selected_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST 
                        )
                    stacked_tensors = np.stack([selected_mask, selected_mask_red_herring], axis=0)

                else: 
                    stacked_tensors = selected_mask_red_herring
                overlay_mask_both = generator.overlay_mask(img, stacked_tensors)
                
                len_red_herring_prompt_toshow = min(60, len(red_herring_prompt))
                save_path = os.path.join(cfg.input.project_path, cfg.output.red_herring_results, f"{img_name.split('.')[0]}_{prompt_seg}_{red_herring_class}_{red_herring_prompt[:len_red_herring_prompt_toshow]}.png")
                generator.save_inpainted_and_mask(inpainted_image_red_herring, overlay_mask_both, save_path=save_path)
                logg.info(f"Saved red herring overlay to {save_path}")



def load_image(image_path, cfg):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((cfg.input.resize_width, cfg.input.resize_height), Image.Resampling.LANCZOS)
    return img
    

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

    # 3. GENERATE OVERLAYS
    overlay_img1 = generator.overlay_mask(img1, mask_before_np)
    overlay_img2 = generator.overlay_mask(img2, mask_after_np)
    overlay_inpainted = generator.overlay_mask(inpainted_image, mask_after_np)

    # 4. SAVE EVERYTHING NEATLY
    save_dir = os.path.join(cfg.input.project_path, cfg.output.production_ready, sequence_id)
    os.makedirs(save_dir, exist_ok=True)
    
    safe_class = prompt_seg.replace(" ", "")
    base_name = f"{img_name1.split('.')[0]}_to_{img_name2.split('.')[0]}_{safe_class}"
    
    img1.save(os.path.join(save_dir, f"{img_name1}.png"))
    mask_before_pil.save(os.path.join(save_dir, f"{base_name}_2_mask_img1.png"))
    overlay_img1.save(os.path.join(save_dir, f"{base_name}_3_overlay_img1.png"))
    
    img2.save(os.path.join(save_dir, f"{img_name2}.png"))
    mask_after_pil.save(os.path.join(save_dir, f"{base_name}_5_mask_img2.png"))
    overlay_img2.save(os.path.join(save_dir, f"{base_name}_6_overlay_img2.png"))
    
    inpainted_image.save(os.path.join(save_dir, f"{base_name}_7_final_inpainted.png"))
    overlay_inpainted.save(os.path.join(save_dir, f"{base_name}_8_overlay_inpainted.png"))
    
    print(f"[{sequence_id}] Saved complete set of 8 images for {prompt_seg} to {save_dir}")


@hydra.main(config_path=".", config_name="config")
def run(cfg: DictConfig):
    log_config(cfg)
    create_output_dirs(cfg)
    
    generator = DatasetGenerator(cfg)
    sam_pipeline = SAM3CorrespondencePipeline(device="cuda")
    class_to_prompt = cfg.input.get("prompts_seg", {})   
    classes = list(class_to_prompt.keys())    
    print(f"Classes to process: {classes}")
    print(f"Class to prompt mapping: {class_to_prompt}")

    sequence_base_path = os.path.join(cfg.input.project_path, cfg.input.image_folder)
    selection_mode = cfg.input.get("mask_selection_mode", "single") 
    
    for sequence_id in os.listdir(sequence_base_path):
        sequence_path = os.path.join(sequence_base_path, sequence_id)
        if not os.path.isdir(sequence_path):
            continue

        valid_extensions = ('.png', '.jpg', '.jpeg')
        image_files = sorted([f for f in os.listdir(sequence_path) if f.lower().endswith(valid_extensions)])

        # Iterate through consecutive pairs (image 1 -> image 2)
        for i in range(len(image_files) - 1):
            img_name1, img_name2 = image_files[i], image_files[i+1]
            img1 = load_image(os.path.join(sequence_path, img_name1), cfg)
            img2 = load_image(os.path.join(sequence_path, img_name2), cfg)
            
            # Load images into SAM
            sam_pipeline.load_image_pair(img1, img2)
            
            for class_name in classes:
                print(" --------------------------------------------------------- ")
                
                # Setup the prompts
                prompt_seg = class_name
                prompt = class_to_prompt[class_name]
                prompt_inpaint = prompt
                
                print(f"[{sequence_id}] Processing '{prompt_seg}' with inpaint prompt: '{prompt_inpaint}'")

                # Track the class
                matches = sam_pipeline.track_class(prompt_seg)
                
                if len(matches) == 0:
                    print(f"No {prompt_seg} matches found. Skipping.")
                    continue

                save_path = os.path.join(cfg.input.project_path, cfg.output.correspondence_visualization, f"{prompt_seg}_{img_name1.split('.')[0]}_{img_name2.split('.')[0]}.png")
                sam_pipeline.visualize_correspondence(img1, img2, matches, save_path=save_path)
                print(f"Found {len(matches)} {prompt_seg} matches between {img_name1} and {img_name2}.")

                # ---------------------------------------------------------
                # 3. RANDOM INSTANCE SELECTION
                # ---------------------------------------------------------
                if selection_mode == "single":
                    num_to_select = 1
                elif selection_mode == "subset":
                    num_to_select = random.randint(1, len(matches))
                else: # "all"
                    num_to_select = len(matches)
                    
                selected_instances = random.sample(matches, k=num_to_select)
                selected_ids = [inst["instance_id"] for inst in selected_instances]
                print(f"Selected {prompt_seg} Instance IDs: {selected_ids}")

                # 2. Inpaint and save
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

            # Clear memory before moving to the next pair
            sam_pipeline.clear_current_pair()
            
    sam_pipeline.shutdown()


if __name__ == "__main__":
    # test_per_model()
    # augmentation_withoneimage()
    # augmentation_withtwoimages()
    # run_red_herring()
    run()