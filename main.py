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


def normalize_prompt(p):
    if isinstance(p, (list, tuple)):
        return "".join(p)
    return p    


def inpaint_output_name(cfg, mask_index):
    inpainted_name = f"{cfg.input.image_name.split('.')[0]}idx{mask_index}_{cfg.prompt_seg}_{cfg.prompt_inpaint.split(',')[0]}_{cfg.model.inpainting.split('/')[-1]}"
    if cfg.input.canny:
        inpainted_name += "_canny"
    if cfg.input.depth:
        inpainted_name += "_depth"
    if cfg.input.inpaint:
        inpainted_name += "_inpaint"
    inpainted_name += ".png"
    return inpainted_name
    
@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    log_config(cfg)
    create_output_dirs(cfg)
    generator = DatasetGenerator(cfg)

    # Load image and prompts
    image_path = os.path.join(cfg.input.project_path, cfg.input.image_folder, cfg.input.image_name)
    img = Image.open(image_path).convert("RGB")
    prompt_seg = cfg.input.prompt_seg
    prompt_inpaint = normalize_prompt(list(cfg.input.prompt_inpaint))
    negative_prompt_inpaint = normalize_prompt(list(cfg.input.negative_prompt_inpaint))
    logg.info(f"promt_seg: {prompt_seg}\nprompt_inpaint: {prompt_inpaint}\nnegative_prompt_inpaint: {negative_prompt_inpaint}")

    # Segmentation and Mask Generation
    mask = generator.segment(img, prompt_seg)
    logg.info(f"Generated mask shape: {mask.shape}")
    overlay = generator.overlay_mask(img, mask)
    # generator.save_image(overlay, title="Segmented Image", save_path=os.path.join(cfg.input.project_path, cfg.output.segmentation_overlay, f"{os.path.basename(image_path).split('.')[0]}_{prompt_seg}_{cfg.model.segmentation.split('/')[-1]}.png"))

    # Filter masks near borders and select one for inpainting
    # filtered_mask = generator.filter_masks(mask)
    filtered_mask = mask
    logg.info(f"Filtered mask shape: {filtered_mask.shape}")
    # mask_index = random.randint(0, filtered_mask.shape[0] - 1)
    mask_index = 1
    logg.info(f"Selected mask index: {mask_index}")
    selected_mask = filtered_mask[mask_index] 
    # selected_mask = generator.dilate_mask(selected_mask, radius=11)

    # generate control images using edge detection and depth estimation for controlnet conditioning
    control_images = generator.generate_control_images(img, selected_mask, save=True)

    # Inpainting
    inpainted_image = generator.inpainting(img, selected_mask, prompt_inpaint,
                 negative_prompt=negative_prompt_inpaint,
                 control_images=control_images)

    # Save inpainting results
    overlay = generator.overlay_mask(img, selected_mask)
    inpainted_name = inpaint_output_name(cfg, mask_index)
    save_path = os.path.join(cfg.output.inpainting_results, inpainted_name)
    generator.save_inpainted_and_mask(img, inpainted_image, overlay, save_path=save_path)

    save_path = os.path.join(cfg.output.inpaited_only_results, inpainted_name)
    generator.save_image(inpainted_image, title="Inpainted Image", save_path=save_path)
    logg.info(f"Saved inpainted image to {save_path}")


    # testing segmentation on inpainted image
    # after_mask = generator.segment(inpainted_image, prompt_seg)
    # overlay = generator.overlay_mask(inpainted_image, after_mask)
    # generator.save_image(overlay, title="Segmented Inpainted Image", save_path=os.path.join(cfg.input.project_path, cfg.output.segmentation_overlay, f"{os.path.basename(image_path).split('.')[0]}index{mask_index}_inpainted_{prompt_seg}_{cfg.model.segmentation.split('/')[-1]}.png")) 


# @hydra.main(config_path=".", config_name="config")
# def testing_panorama(cfg: DictConfig):
#     log_config(cfg)
#     create_output_dirs(cfg)
#     cam = Panorama()
    




if __name__ == "__main__":
    main()