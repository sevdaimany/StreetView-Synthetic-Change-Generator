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

def log_config(cfg):
    logg.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    if "FLUX" in cfg.model.inpainting:
        logg.info("Using FLUX for inpainting")
    elif "xl" in cfg.model.inpainting:
        logg.info("Using Stable Diffusion XL + ControlNet for inpainting")
    else:
        logg.info("Using Stable Diffusion + ControlNet for inpainting")


def load_image(image_name, cfg):
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
def main(cfg: DictConfig):
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
            img = load_image(img_name, cfg)
            generator.inference(img, img_name, prompt_seg, prompt, negative_prompt_inpaint=neg_prompt)

    elif run_mode == "combinatorial":
        for img_name in image_names:
            
            # Scenario A: Lengths match perfectly -> Treat them as pairs
            if len(prompts) == len(neg_prompts):
                for prompt, neg_prompt in zip(prompts, neg_prompts):
                    logg.info(f"--- Processing: {img_name} | Prompt: '{prompt}' | Negative Prompt: '{neg_prompt}' ---")
                    img = load_image(img_name, cfg)
                    generator.inference(img, img_name, prompt_seg, prompt, negative_prompt_inpaint=neg_prompt)
            
            # Scenario B: Lengths differ (or one is length 1) -> Iterate on BOTH (all combinations)
            else:
                for prompt in prompts:
                    for neg_prompt in neg_prompts:
                        logg.info(f"--- Processing: {img_name} | Prompt: '{prompt}' | Negative Prompt: '{neg_prompt}' ---")
                        img = load_image(img_name, cfg)
                        generator.inference(img, img_name, prompt_seg, prompt, negative_prompt_inpaint=neg_prompt)   
    else:
        logg.error(f"Unknown run_mode: {run_mode}")

  
@hydra.main(config_path=".", config_name="config")
def automated_run(cfg: DictConfig):
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

        img = load_image(img_name, cfg)

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







if __name__ == "__main__":
    # main()
    # augmentation_withoneimage()
    # augmentation_withtwoimages()
    automated_run()