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
from SAM3CorrespondenceMultiplex import SAM3CorrespondenceMultiplex
import py360convert
from CubemapsTracker import PanoramaTracker
import matplotlib.pyplot as plt
# from SAM3SequencePipeline import SAM3SequencePipeline

login(os.getenv("HF_TOKEN"))
logg = logging.getLogger(__name__)


def create_output_dirs(cfg):
    """Utility to create output directories if they don't exist."""
    os.makedirs(cfg.output.base, exist_ok=True)
    os.makedirs(os.path.join(cfg.output.base, cfg.output.correspondence_visualization), exist_ok=True)
    os.makedirs(os.path.join(cfg.output.base, cfg.output.segmentation_overlay), exist_ok=True)
    os.makedirs(os.path.join(cfg.output.base, cfg.output.inpainting_results), exist_ok=True)
    os.makedirs(os.path.join(cfg.output.base, cfg.output.edge_detection_results), exist_ok=True)
    os.makedirs(os.path.join(cfg.output.base, cfg.output.red_herring_results), exist_ok=True)
    os.makedirs(os.path.join(cfg.output.base, cfg.output.depth_results), exist_ok=True)
    os.makedirs(os.path.join(cfg.output.base, cfg.output.inpaited_only_results), exist_ok=True)
    os.makedirs(os.path.join(cfg.output.base, cfg.output.augmentation), exist_ok=True)
    os.makedirs(os.path.join(cfg.output.base, cfg.output.augmentation_masks), exist_ok=True)
    os.makedirs(os.path.join(cfg.output.base, cfg.output.production_ready), exist_ok=True)
    os.makedirs(os.path.join(cfg.output.base, cfg.output.cubemap_tracking), exist_ok=True)

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
        img = img.resize((cfg.input.resize_width, cfg.input.resize_height), Image.Resampling.LANCZOS)
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
    save_path_mask = os.path.join(cfg.output.base, cfg.output.augmentation_masks, f"{base_name}_z{cfg.augmentation.translation_z}_deg{cfg.augmentation.yaw_deg}.png")
    save_path = os.path.join(cfg.output.base, cfg.output.augmentation, f"{base_name}_z{cfg.augmentation.translation_z}_deg{cfg.augmentation.yaw_deg}.png")
    
    warped_img.save(os.path.join(cfg.output.base, cfg.output.augmentation, f"beforeinpaint_{base_name}_z{cfg.augmentation.translation_z}_deg{cfg.augmentation.yaw_deg}.png"))
    final_sdxl_mask.save(save_path_mask)
    final_inpainted_img.save(save_path)
    print(f"Saved augmented image to {save_path}")
    
@hydra.main(config_path=".", config_name="config")
def sam3_object_tracking_multiplex(cfg: DictConfig):
    log_config(cfg)
    create_output_dirs(cfg)
    class_to_prompt = cfg.input.get("prompts_seg", {})   
    classes_to_track = list(class_to_prompt.keys())
    # classes_to_track = ["buildings, traffic signs, roads"]
    print(f"Classes to track: {classes_to_track}")

    # input
    image_names = list(cfg.input.get("image_names", [cfg.input.get("image_name")]))
    resized_images = []
    for img_name in image_names:
        img_path = os.path.join(cfg.input.project_path, cfg.input.image_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        resized_img = img.resize(
            (cfg.input.resize_width, cfg.input.resize_height), 
            Image.Resampling.LANCZOS
        )

        resized_images.append(resized_img)

    pipeline = SAM3CorrespondenceMultiplex()
    pipeline.load_image_sequence(resized_images)

    outputs, class_id_mapping = pipeline.track_multiple_classes(classes_to_track)
    save_path = os.path.join(cfg.output.base, cfg.output.correspondence_visualization, f"tracking_{image_names[0].split('.')[0]}_{image_names[1].split('.')[0]}.png")
    pipeline.visualize_sequence(resized_images, outputs, save_path)

    # 1. Find its index in your list (index 1)
    building_ids = class_id_mapping.get("buildings", [])
    print(f"Tracking {len(building_ids)} instances for 'buildings'.")
    building_masks = pipeline.get_masks_for_single_class(outputs, target_obj_ids=building_ids)

    # Visualize ONLY the buildings using the same built-in function
    save_path_buildings = os.path.join(cfg.output.base, cfg.output.correspondence_visualization, f"building_tracking_{image_names[0].split('.')[0]}_{image_names[1].split('.')[0]}.png")
    pipeline.visualize_sequence(resized_images, building_masks, save_path_buildings)


    pipeline.shutdown()

@hydra.main(config_path=".", config_name="config")
def sam3_object_tracking(cfg: DictConfig):

    log_config(cfg)
    create_output_dirs(cfg)
    print(f"image folder: {cfg.input.image_folder}")
    print(f"image names: {cfg.input.image_names}")
    
    # input
    # image_1, image_2 = list(cfg.input.get("image_names", [cfg.input.get("image_name")]))

    # seq_id = "0b1aefa9-2a60-4ae7-a208-f6a934065086"
    # image_1 = "buffer_01.jpg"
    # image_2 = "04_next_1.jpg"
    # seq_id = "01d7040b-82f7-4088-a02d-56dd76a594f3"
    # image_1 = "buffer_01.jpg"
    # image_2 = "01_prev_2.jpg"

    # seq_id = "0555c731-9dfb-4c23-8440-283d2fa20f69"
    # image_1 = "buffer_04.jpg"
    # image_2 = "02_prev_1.jpg"

    seq_id = "02b00ac0-83fd-4446-97d6-111d31699fa4"
    image_1 = "03_center.jpg"
    image_2 = "02_prev_1.jpg"
    # image_1 = "buffer_04.jpg"


    # seq_id = "fake"
    # image_1 = "panorama3_original.png"
    # image_2 = "panorama3_same street, heavy snow, _canny_depth_segmented_sd_strength0.8.png"

    # seq_id = "argentina_835-Calle-57-La-Plata_11-2024"
    # image_1 = "1.jpg"
    # image_2 = "2.jpg"

    img_1 = Image.open(os.path.join(cfg.input.project_path, cfg.input.image_folder, seq_id, image_1)).convert("RGB")
    img_2 = Image.open(os.path.join(cfg.input.project_path, cfg.input.image_folder, seq_id, image_2)).convert("RGB")
    resized_img_1 = img_1.resize((cfg.input.resize_width, cfg.input.resize_height), Image.Resampling.LANCZOS)
    resized_img_2 = img_2.resize((cfg.input.resize_width, cfg.input.resize_height), Image.Resampling.LANCZOS)

    # 1. Initialize the class globally or at the top of your script
    use_sam3 = cfg.input.get("use_sam3", False)
    sam_pipeline = SAM3CorrespondencePipeline(use_sam3=use_sam3, device="cuda")
    

    # 2. Load your images into the session
    sam_pipeline.load_image_pair(resized_img_1, resized_img_2)

    # 3. Query different classes (images are already cached in VRAM)
    building_matches = sam_pipeline.track_class("buildings")
    
    print(f"Found {len(building_matches)} buildings.")
    sam_version = "sam3" if use_sam3 else "sam3.1"
    if len(building_matches) > 0:
        save_path = os.path.join(cfg.output.base, cfg.output.correspondence_visualization, f"building_{seq_id}_{image_1.split('.')[0]}_{image_2.split('.')[0]}_{sam_version}.png")
        sam_pipeline.visualize_correspondence(resized_img_1, resized_img_2, building_matches, save_path=save_path)

    sign_matches = sam_pipeline.track_class("traffic signs")
    print(f"Found {len(sign_matches)} traffic signs.")
    if len(sign_matches) > 0:
        save_path = os.path.join(cfg.output.base, cfg.output.correspondence_visualization, f"trafficsign_{seq_id}_{image_1.split('.')[0]}_{image_2.split('.')[0]}_{sam_version}.png")
        sam_pipeline.visualize_correspondence(resized_img_1, resized_img_2, sign_matches, save_path=save_path)

    crack_matches = sam_pipeline.track_class("cracks")
    print(f"Found {len(crack_matches)} cracks.")
    if len(crack_matches) > 0:
        save_path = os.path.join(cfg.output.base, cfg.output.correspondence_visualization, f"crack_{seq_id}_{image_1.split('.')[0]}_{image_2.split('.')[0]}_{sam_version}.png")
        sam_pipeline.visualize_correspondence(resized_img_1, resized_img_2, crack_matches, save_path=save_path)

    trash_matches = sam_pipeline.track_class("trash bins")
    print(f"Found {len(trash_matches)} trash bins.")
    if len(trash_matches) > 0:
        save_path = os.path.join(cfg.output.base, cfg.output.correspondence_visualization, f"trash_{seq_id}_{image_1.split('.')[0]}_{image_2.split('.')[0]}_{sam_version}.png")
        sam_pipeline.visualize_correspondence(resized_img_1, resized_img_2, trash_matches, save_path=save_path)

    crosswalk_matches = sam_pipeline.track_class("cross walks")
    print(f"Found {len(crosswalk_matches)} cross walks.")
    if len(crosswalk_matches) > 0:
        save_path = os.path.join(cfg.output.base, cfg.output.correspondence_visualization, f"crosswalk_{seq_id}_{image_1.split('.')[0]}_{image_2.split('.')[0]}_{sam_version}.png")
        sam_pipeline.visualize_correspondence(resized_img_1, resized_img_2, crosswalk_matches, save_path=save_path)
    
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
            generator.inference(img, img_name, prompt_seg, prompt, negative_prompt_inpaint=neg_prompt, save_all=True)

    elif run_mode == "combinatorial":
        for img_name in image_names:
            
            # Scenario A: Lengths match perfectly -> Treat them as pairs
            if len(prompts) == len(neg_prompts):
                for prompt, neg_prompt in zip(prompts, neg_prompts):
                    logg.info(f"--- Processing: {img_name} | Prompt: '{prompt}' | Negative Prompt: '{neg_prompt}' ---")
                    img = load_image(img_name, cfg)
                    # generator.inference(img, img_name, prompt_seg, prompt, negative_prompt_inpaint=neg_prompt, save_all=True)
            
                    # if you want changing the weather
                    generator.inference_change_style(img, img_name, prompt_seg, prompt, negative_prompt_inpaint=neg_prompt, save_all=True)
            
            # Scenario B: Lengths differ (or one is length 1) -> Iterate on BOTH (all combinations)
            else:
                for prompt in prompts:
                    for neg_prompt in neg_prompts:
                        logg.info(f"--- Processing: {img_name} | Prompt: '{prompt}' | Negative Prompt: '{neg_prompt}' ---")
                        img = load_image(img_name, cfg)
                        # generator.inference(img, img_name, prompt_seg, prompt, negative_prompt_inpaint=neg_prompt, save_all=True)   

                        # changing the weather
                        generator.inference_change_style(img, img_name, prompt_seg, prompt, negative_prompt_inpaint=neg_prompt, save_all=True)   
    else:
        logg.error(f"Unknown run_mode: {run_mode}")


@hydra.main(config_path=".", config_name="config")
def automated_run_folder(cfg: DictConfig):
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
            
            inpainted_image, selected_mask = generator.inference(img, img_name, prompt_seg=prompt_seg, prompt_inpaint=prompt_inpaint, save_all=True)

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
                save_path = os.path.join(cfg.output.base, cfg.output.red_herring_results, f"{img_name.split('.')[0]}_{prompt_seg}_{red_herring_class}_{red_herring_prompt[:len_red_herring_prompt_toshow]}.png")
                generator.save_inpainted_and_mask(inpainted_image_red_herring, overlay_mask_both, save_path=save_path)
                logg.info(f"Saved red herring overlay to {save_path}")


@hydra.main(config_path=".", config_name="config")
def sam3_object_tracking_sequence(cfg: DictConfig):
    log_config(cfg)
    create_output_dirs(cfg)
    
    print(f"image folder: {cfg.input.image_folder}")
    
    seq_id = "02b00ac0-83fd-4446-97d6-111d31699fa4"
    # image_names =  ["01_prev_2.jpg", "02_prev_1.jpg", "03_center.jpg", "04_next_1.jpg", "05_next_2.jpg"]
    image_names =  ["01_prev_2.jpg", "02_prev_1.jpg", "03_center.jpg", "04_next_1.jpg", "05_next_2.jpg"][::-1]  # Reverse the order to test robustness of tracking in both directions

    
    resized_images = []
    for img_name in image_names:
        img_path = os.path.join(cfg.input.project_path, cfg.input.image_folder, seq_id, img_name)
        img = Image.open(img_path).convert("RGB")
        resized_img = img.resize((cfg.input.resize_width, cfg.input.resize_height), Image.Resampling.LANCZOS)
        resized_images.append(resized_img)

    use_sam3 = cfg.input.get("use_sam3", False)
    sam_pipeline = SAM3CorrespondencePipeline(use_sam3=use_sam3, device="cuda")
    
    # 1. Load entire sequence
    sam_pipeline.load_image_sequence(resized_images)
    sam_version = "sam3" if use_sam3 else "sam3.1"
    
    classes_to_track = ["buildings", "traffic signs", "cracks", "trash bins", "cross walks"]
    seq_str = f"{image_names[0].split('.')[0]}_to_{image_names[-1].split('.')[0]}"
    
    # NEW: Create a dictionary to store all our tracking results so we don't have to re-run SAM
    all_tracked_data = {}
    
    # 2. Iterate dynamically over classes to track and save 
    for class_name in classes_to_track:
        matches = sam_pipeline.track_class_sequence(class_name)
        
        # Save the matches into our dictionary for later use
        all_tracked_data[class_name] = matches
        
        print(f"Found {len(matches)} {class_name}.")
        
        if len(matches) > 0:
            clean_name = class_name.replace(" ", "")
            save_path = os.path.join(
                cfg.output.base, 
                cfg.output.correspondence_visualization, 
                f"{clean_name}_{seq_id}_{seq_str}_{sam_version}_ALLframe_index.png"
            )
            sam_pipeline.visualize_sequence_correspondence(resized_images, matches, save_path=save_path)


    # 3. Extract data for synthetic change later (using the dictionary we just populated!)
    building_matches = all_tracked_data.get("buildings", [])
    
    if building_matches:
        selected_instance = random.choice(building_matches)
        instance_id = selected_instance["instance_id"]
        
        mask_first_frame = selected_instance["masks"][0]
        mask_last_frame = selected_instance["masks"][-1]
        
        print(f"\n--- Synthetic Change Prep ---")
        print(f"Selected instance ID: {instance_id}")
        print(f"Mask First Frame shape: {mask_first_frame.shape if mask_first_frame is not None else 'None'}")
        print(f"Mask Last Frame shape: {mask_last_frame.shape if mask_last_frame is not None else 'None'}")
        print("-----------------------------\n")

    # 4. Clean up SAFELY at the very end of the script
    sam_pipeline.clear_current_pair()
    sam_pipeline.shutdown()


if __name__ == "__main__":
    # main()
    # augmentation_withoneimage()
    # sam3_object_tracking()
    # automated_run_folder()
    # sam3_object_tracking_multiplex()
    # sam3_object_tracking_cubemaps()
    sam3_object_tracking_sequence()