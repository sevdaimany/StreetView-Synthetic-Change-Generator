import torch
from DatasetGenerator import DatasetGenerator
import cv2
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv
from huggingface_hub import login
from omegaconf import OmegaConf, DictConfig
import hydra
import logging
# import Panorama
import random

load_dotenv()
login(os.getenv("HF_TOKEN"))
logg = logging.getLogger(__name__)


def create_output_dirs(cfg):
    """Utility to create output directories if they don't exist."""
    os.makedirs(cfg.output.segmentation_overlay, exist_ok=True)
    os.makedirs(cfg.output.inpainting_results, exist_ok=True)
    os.makedirs(cfg.output.edge_detection_results, exist_ok=True)
    os.makedirs(cfg.output.red_herring_results, exist_ok=True)
    os.makedirs(cfg.output.depth_results, exist_ok=True)
    os.makedirs(cfg.output.inpaited_only_results, exist_ok=True)


def log_config(cfg):
    logg.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    if "FLUX" in cfg.model.inpainting:
        logg.info("Using FLUX for inpainting")
    elif "xl" in cfg.model.inpainting:
        logg.info("Using Stable Diffusion XL + ControlNet for inpainting")
    else:
        logg.info("Using Stable Diffusion + ControlNet for inpainting")


def inpaint_output_name(cfg, image_name, mask_index, prompt_inpaint, negative_prompt_inpaint=None):
    model = ""
    if "FLUX" in cfg.model.inpainting:
        model = "flux"
    elif "xl" in cfg.model.inpainting:
        model = "sdxl"
    else:
        model = "sd"

    len_neg_prompt_toshow = min(20, len(negative_prompt_inpaint)) if negative_prompt_inpaint else 0
    len_prompt_toshow = min(20, len(prompt_inpaint)) if prompt_inpaint else 0
    neg_prompt_part = negative_prompt_inpaint[:len_neg_prompt_toshow] if negative_prompt_inpaint else "NoPrompt"
    prompt_part = prompt_inpaint[:len_prompt_toshow] if prompt_inpaint else "PosNoPrompt"
    
    inpainted_name = f"{image_name.split('.')[0]}idx{mask_index}_{cfg.input.prompt_seg}_{prompt_part}_Neg{neg_prompt_part}_{model}"
    if cfg.input.canny:
        inpainted_name += "_canny"
    if cfg.input.depth:
        inpainted_name += "_depth"
    if cfg.input.inpaint:
        inpainted_name += "_inpaint"
    if cfg.input.dilated_mask:
        inpainted_name += "_dilated"
    inpainted_name += ".png"
    return inpainted_name

def process_single_run(cfg, generator, image_name, prompt_inpaint, negative_prompt_inpaint, save_all=True):
    logg.info(f"--- Processing: {image_name} | Prompt: '{prompt_inpaint}' | Negative Prompt: '{negative_prompt_inpaint}' ---")
    
    image_path = os.path.join(cfg.input.project_path, cfg.input.image_folder, image_name)
    img = Image.open(image_path).convert("RGB")
    if 'panorama' in image_name.lower():
        img = img.resize((cfg.input.resize_width, cfg.input.resize_height), Image.BILINEAR)

    # Segmentation and Mask Generation
    prompt_seg = cfg.input.prompt_seg
    mask = generator.segment(img, prompt_seg)
    logg.info(f"Generated mask shape: {mask.shape}")
    if save_all:
        overlay = generator.overlay_mask(img, mask)
        seg_save_path = os.path.join(cfg.input.project_path, cfg.output.segmentation_overlay, f"{os.path.basename(image_path).split('.')[0]}_{prompt_seg}_{cfg.model.segmentation.split('/')[-1]}.png") 
        # generator.save_image(overlay, title="Segmented Image", save_path=seg_save_path)
        overlay.save(seg_save_path)

        logg.info(f"Saved segmentation overlay to {seg_save_path}")

    # Filter masks near borders and select one for inpainting
    filtered_mask = mask
    # filtered_mask = generator.filter_masks(mask)
    logg.info(f"Filtered mask shape: {filtered_mask.shape}")
    mask_index = cfg.input.mask_index
    if mask_index == -1:
        mask_index = random.randint(0, filtered_mask.shape[0] - 1)

    logg.info(f"Selected mask index: {mask_index}")
    selected_mask = filtered_mask[mask_index]
    if cfg.input.dilated_mask:
        selected_mask = generator.dilate_mask(selected_mask, radius=15)

    # generate control images using edge detection and depth estimation for controlnet conditioning
    control_images = generator.generate_control_images(img, selected_mask, image_name, save=save_all)

    # Inpainting
    logg.info("Starting inpainting...")
    inpainted_image = generator.inpainting(img, selected_mask, prompt_inpaint,
                    negative_prompt=negative_prompt_inpaint,
                    control_images=control_images)
    logg.info("Inpainting completed.")

    # Save inpainting results
    overlay = generator.overlay_mask(img, selected_mask)
    inpainted_name = inpaint_output_name(cfg, image_name, mask_index, prompt_inpaint, negative_prompt_inpaint)

    save_path = os.path.join(cfg.input.project_path, cfg.output.inpainting_results, inpainted_name)
    generator.save_inpainted_and_mask(inpainted_image, overlay, save_path=save_path)
    logg.info(f"Saved inpainted image with mask overlay to {save_path}")
    
    save_path = os.path.join(cfg.input.project_path, cfg.output.inpaited_only_results, inpainted_name)
    # generator.save_image(inpainted_image, title="Inpainted Image", save_path=save_path)
    inpainted_image.save(save_path)

    logg.info(f"Saved inpainted image to {save_path}")

    # testing segmentation on inpainted image
    # after_mask = generator.segment(inpainted_image, prompt_seg)
    # overlay = generator.overlay_mask(inpainted_image, after_mask)
    # generator.save_image(overlay, title="Segmented Inpainted Image", save_path=os.path.join(cfg.input.project_path, cfg.output.segmentation_overlay, f"{os.path.basename(image_path).split('.')[0]}index{mask_index}_inpainted_{prompt_seg}_{cfg.model.segmentation.split('/')[-1]}.png")) 
    # overlay.save(os.path.join(cfg.input.project_path, cfg.output.segmentation_overlay, f"{os.path.basename(image_path).split('.')[0]}index{mask_index}_inpainted_{prompt_seg}_{cfg.model.segmentation.split('/')[-1]}.png"))


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    log_config(cfg)
    create_output_dirs(cfg)
    generator = DatasetGenerator(cfg)

    # 1. Extract lists from config (fallback to defaults if they don't exist)
    image_names = list(cfg.input.get("image_names", [cfg.input.get("image_name")]))
    prompts = list(cfg.input.get("prompts_inpaint", [""]))
    neg_prompts = list(cfg.input.get("negative_prompts_inpaint", [""]))

    # If only one negative prompt is provided but multiple positive prompts, broadcast it
    if len(neg_prompts) == 1 and len(prompts) > 1:
        neg_prompts = neg_prompts * len(prompts)

    run_mode = cfg.input.get("run_mode", "combinatorial")

    # 2. Execute based on run mode
    if run_mode == "pairwise":
        assert len(image_names) == len(prompts), "For pairwise mode, the number of images and prompts must be equal."
        for img_name, prompt, neg_prompt in zip(image_names, prompts, neg_prompts):
            process_single_run(cfg, generator, img_name, prompt, neg_prompt)

    elif run_mode == "combinatorial":
        for img_name in image_names:
            # Scenario A: Lengths match perfectly -> Treat them as pairs
            if len(prompts) == len(neg_prompts):
                for prompt, neg_prompt in zip(prompts, neg_prompts):
                    process_single_run(cfg, generator, img_name, prompt, neg_prompt)
            
            # Scenario B: Lengths differ (or one is length 1) -> Iterate on BOTH (all combinations)
            else:
                for prompt in prompts:
                    for neg_prompt in neg_prompts:
                        process_single_run(cfg, generator, img_name, prompt, neg_prompt)   
    else:
        logg.error(f"Unknown run_mode: {run_mode}")

  

# @hydra.main(config_path=".", config_name="config")
# def testing_panorama(cfg: DictConfig):
#     log_config(cfg)
#     create_output_dirs(cfg)
#     cam = Panorama()
    




if __name__ == "__main__":
    main()