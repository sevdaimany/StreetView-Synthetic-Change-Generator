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
import torch.multiprocessing as mp
import fcntl
import torch
import gc

load_dotenv()
login(os.getenv("HF_TOKEN"))


def process_and_save_synthetic_change(
    generator, 
    cfg, 
    sequence_id, 
    city_name,
    img1, 
    img2, 
    img_name1, 
    img_name2, 
    selected_instances, 
    prompt_seg, 
    prompt_inpaint,
    logger
    ):
    """
    Gets the selected masks from SAM, applies the inpainting generator to create synthetic changes,
    and saves the complete before/after pipeline to disk.
    """

    current_pair = (img_name1, img_name2)
    viz_dir, data_dir = create_folders(cfg, city_name, sequence_id)

    # EXTRACT AND MERGE BOTH 'BEFORE' AND 'AFTER' MASKS FOR ALL SELECTED INSTANCES
    after_masks = [np.array(inst["after_mask"]) > 0 for inst in selected_instances]
    before_masks = [np.array(inst["before_mask"]) > 0 for inst in selected_instances]

    # np.any evaluates True if any mask in the stack has a True pixel at that coordinate
    merged_after = np.any(after_masks, axis=0)
    merged_before = np.any(before_masks, axis=0)
    
    mask_after_np = (merged_after * 255).astype(np.uint8)
    mask_before_np = (merged_before * 255).astype(np.uint8)
    
    mask_after_pil = Image.fromarray(mask_after_np, mode="L")
    mask_before_pil = Image.fromarray(mask_before_np, mode="L")

    # 1) change season style
    apply_weather = random.random() < 0.5
    prompts_weather = cfg.input.get("prompts_weather", [])
    if apply_weather:
        # randomly choose between image1 and image2 for the weather change
        if random.random() < 0.5:
            img_for_weather = img1
            weather_image_name = img_name1
            # randomly select a weather prompt
            prompt_weather = random.choice(prompts_weather) 
            img1 = generator.inference_change_style(img_for_weather, weather_image_name, prompt_weather, save_path=viz_dir, save_all=True)

        else:
            img_for_weather = img2
            weather_image_name = img_name2
            prompt_weather = random.choice(prompts_weather)
            img2 = generator.inference_change_style(img_for_weather, weather_image_name, prompt_weather, save_path=viz_dir, save_all=True)
        logger.info(f"[{city_name} / {sequence_id} / {weather_image_name} / {prompt_seg}] Applied weather change: '{prompt_weather}' to image '{weather_image_name}'")
        logger.info(f"[{city_name} / {sequence_id} / {current_pair} / {prompt_seg}] After generator.inference_change_style: img1 size: {img1.size}, img2 size: {img2.size}")


    # 2) apply synthetic change
    inpainted_image, selected_mask = generator.inference(
        img=img2, 
        image_name=img_name2, 
        prompt_seg=prompt_seg, 
        prompt_inpaint=prompt_inpaint,
        seg_mask=mask_after_pil, 
    )
    # Standardize sizes
    logger.info(f"[{city_name} / {sequence_id} / {current_pair} / {prompt_seg}] After generator.inference: img1 size: {img1.size}, img2 size: {img2.size},  inpainted size: {inpainted_image.size}, mask_after size: {mask_after_pil.size}")
                    
    target_size = img1.size
    if inpainted_image.size != target_size or mask_after_pil.size != target_size or mask_before_pil.size != target_size:
        print(f"⚠️ Size mismatch detected. Resizing all to {target_size}.")
        logger.warning(f"[{city_name} / {sequence_id} / {current_pair} / {prompt_seg}] Size mismatch detected. Resizing all to {target_size}.")
        inpainted_image = inpainted_image.resize(target_size, Image.Resampling.LANCZOS)
        mask_after_pil = mask_after_pil.resize(target_size, Image.Resampling.NEAREST)
        mask_before_pil = mask_before_pil.resize(target_size, Image.Resampling.NEAREST)
        logger.info( f"[{city_name} / {sequence_id} / {current_pair} / {prompt_seg}] After Size Standardization - img1: {img1.size}, img2: {img2.size}, inpainted: {inpainted_image.size}, mask_before: {mask_before_pil.size}, mask_after: {mask_after_pil.size}")
    mask_after_np = np.array(mask_after_pil)
    mask_before_np = np.array(mask_before_pil)
    
    # VERIFY BUILDING REMOVAL / REPLACEMENT
    verification_status = "Not Checked"
    
    if "buildings" == prompt_seg.lower():
        verification_status = generator.verify_building_removal(inpainted_image, prompt_seg, mask_after_np)
        logger.info(f"[{city_name} / {sequence_id} / {current_pair} / {prompt_seg}] SAM Verification: Building was {verification_status.upper()}.")



    safe_class = prompt_seg.replace(" ", "_")
    pair_id = f"{img_name1.split('.')[0]}_to_{img_name2.split('.')[0]}_{safe_class}"

    # SAVE TRAINING ASSETS
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

    # SAVE VISUALIZATION ASSETS
    viz_bbox_before = os.path.join(viz_dir, f"{pair_id}_bbox_before.jpg")
    viz_bbox_after = os.path.join(viz_dir, f"{pair_id}_bbox_after.jpg")
    save_voc_bboxes_and_overlay(image_pil=img1, instances=selected_instances, mask_key="before_mask", 
        class_name=prompt_seg, txt_path=paths["txt_before"], overlay_path=viz_bbox_before)
    save_voc_bboxes_and_overlay(image_pil=img2, instances=selected_instances, mask_key="after_mask", 
        class_name=prompt_seg, txt_path=paths["txt_after"], overlay_path=viz_bbox_after)
    
    overlay_img1 = generator.overlay_mask(img1, mask_before_np)
    overlay_img2 = generator.overlay_mask(img2, mask_after_np)
    grid = create_qc_grid([overlay_img1, overlay_img2, inpainted_image], 
                          labels=["Before overlay", "After overlay", "Inpainted"], font_path=cfg.input.font_path)
    grid.save(os.path.join(viz_dir, f"{pair_id}_QC.jpg"), quality=85) # JPG saves space for viz

    # Save inpainting results
    overlay = generator.overlay_mask(img2, selected_mask)
    inpainted_name = pair_id + ".png"
    generator.save_inpainted_and_mask(inpainted_image, overlay, save_path=os.path.join(viz_dir, "inpainting", inpainted_name))

    # SAVE METADATA
    metadata = {
        "pair_id": pair_id,
        "city_name": city_name,
        "sequence_id": sequence_id,
        "class_name": prompt_seg,
        "prompt": prompt_inpaint,
        "num_instances": len(selected_instances),
        "verification_status": verification_status,
        "img1_name": img_name1,
        "img2_name": img_name2,
        "applied_weather_change": apply_weather,
        "weather_prompt": prompt_weather if apply_weather else None,
        "weather_image_name": weather_image_name if apply_weather else None,
        "paths": paths
    }
    
    json_path = os.path.join(data_dir, f"{pair_id}_meta.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)


def create_folders(cfg, city_name, sequence_id):
    # DEFINE DIRECTORY STRUCTURE
    # Separate 'raw' data for training and 'viz' for human checking
    base_dir = cfg.output.dir_root
    viz_dir = os.path.join(base_dir, "pipeline_visualization", city_name, sequence_id)
    data_dir = os.path.join(base_dir, "pipeline_data", city_name, sequence_id)
    
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    os.makedirs(os.path.join(viz_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(viz_dir, "edge_detection"), exist_ok=True)
    os.makedirs(os.path.join(viz_dir, "weather"), exist_ok=True)
    os.makedirs(os.path.join(viz_dir, "inpainting"), exist_ok=True)
    

    return viz_dir, data_dir


def process_sequence(sequence_id, base_path, classes, class_to_prompt, sam_pipeline, generator, cfg, logger):
    # get city name from path
    sequence_path = os.path.join(base_path, sequence_id)
    city_name = os.path.basename(base_path)
    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = sorted([f for f in os.listdir(sequence_path) if f.lower().endswith(valid_extensions)])
    image_files = image_files[1:4]
    selection_mode = cfg.input.get("mask_selection_mode", "single")
    logger.info(f"\n[{city_name} / {sequence_id}] Initializing sequence...")


    # Generate the adjacent pairs
    adjacent_pairs = []
    adjacent_pairs.append((image_files[1], image_files[1])) # center-center
    adjacent_pairs.append((image_files[0], image_files[1])) # prev-center
    adjacent_pairs.append((image_files[1], image_files[2])) # center-next
    adjacent_pairs.append((image_files[0], image_files[2])) # prev-next


    adjacent_classes = ["traffic signs", "traffic lights", "trash cans"] # classes that require adjacent pairing logic
    
    for img_name1, img_name2 in adjacent_pairs:
        current_pair = (img_name1, img_name2)
        try:
            img1 = load_image(os.path.join(sequence_path, img_name1), cfg)
            img2 = load_image(os.path.join(sequence_path, img_name2), cfg)
            sam_pipeline.load_image_pair(img1, img2)
            
            for class_name in classes:
                try:
                    if class_name in adjacent_classes and current_pair == (image_files[0], image_files[2]):
                        continue  # Skip should only process adjacent pairs for these classes
                    

                    prompt_seg = class_name
                    prompt_inpaint = class_to_prompt[class_name]
                    
                    # # RESTART LOGIC: Check if work is already done
                    if check_redundancy(city_name, sequence_id, class_name, img_name1, img_name2, cfg):
                        logger.info(f"[{city_name} / {sequence_id} / {current_pair} / {prompt_seg}] Skipping {class_name} for {img_name1} -> {img_name2} (already processed)")
                        continue

                    # Core Processing
                    matches = sam_pipeline.track_class(prompt_seg)
                    if len(matches) == 0:
                        logger.warning(f"[{city_name} / {sequence_id} / {current_pair} / {prompt_seg}] No {prompt_seg} matches found. Skipping.")
                        continue
                

                    # comment if you dont want to save correspondences
                    # save_path = os.path.join(cfg.output.base, cfg.output.production_ready, cfg.output.correspondence_visualization, f"{prompt_seg}_{sequence_id}_{img_name1.split('.')[0]}_{img_name2.split('.')[0]}.jpg")
                    # sam_pipeline.visualize_correspondence(img1, img2, matches, save_path=save_path)
                    # continue # Skip the rest if you only want to save correspondences and not generate synthetic changes


                    # Selection logic...
                    areas = [np.sum(np.array(m["after_mask"]) > 0) for m in matches]
                    average_area = np.mean(areas)
                    selected_matches = [m for m, area in zip(matches, areas) if area >= average_area]

                    logger.info(f"[{city_name} / {sequence_id} / {current_pair} / {prompt_seg}]  {len(selected_matches)} matches selected after area filtering (average area: {average_area:.2f}), out of {len(matches)} total matches.")
                    
                    if selection_mode == "biggest":
                        selected_instances = [max(selected_matches, key=lambda x: np.sum(x["after_mask"]))]
                    else:
                        if selection_mode == "subset": # 75% chance to select only one, 25% chance to select all
                                num_to_select = 1 if random.random() < 0.75 else min(len(selected_matches), 2)
                        elif selection_mode == "single":
                            num_to_select = 1
                        else: # "all"
                            num_to_select = len(selected_matches)
                        selected_instances = random.sample(selected_matches, k=num_to_select)

                    logger.info(f"[{city_name} / {sequence_id} / {current_pair} / {prompt_seg}] Before processing synthetic change: img1 size: {img1.size}, img2 size: {img2.size}, segmentation size: {selected_instances[0]['after_mask'].shape}")
                    process_and_save_synthetic_change(
                        generator=generator,
                        cfg=cfg,
                        sequence_id=sequence_id,
                        city_name=city_name,
                        img1=img1,
                        img2=img2,
                        img_name1=img_name1,
                        img_name2=img_name2,
                        selected_instances=selected_instances,
                        prompt_seg=prompt_seg,
                        prompt_inpaint=prompt_inpaint,
                        logger=logger
                    )
                    logger.info(f"[{city_name} / {sequence_id} / {current_pair} / {prompt_seg}] Successfully processed {class_name} | {img_name1} \n")

                except Exception as e:
                    logger.error(f"[{city_name} / {sequence_id} / {current_pair} / {prompt_seg}] Error processing class '{class_name}' for pair {img_name1}-{img_name2}: {e}")
                    logger.error(traceback.format_exc())

        except Exception as e:
            logger.error(f"[{city_name} / {sequence_id} / {current_pair}] Error processing pair {img_name1}-{img_name2}: {e}")
            logger.error(traceback.format_exc())
        finally:
            sam_pipeline.clear_current_pair()


def load_models(cfg, device, logger):
    generator = DatasetGenerator(cfg, device=device)
    sam_pipeline = SAM3CorrespondencePipeline(device=device)
    return generator, sam_pipeline



def process_city_worker(args):
    city_name, cfg, gpu_queue, completed_file = args

    # Atomic File Locking
    city_output_path = os.path.join(cfg.output.dir_root, 'pipeline_data', city_name)
    os.makedirs(city_output_path, exist_ok=True)
    lock_file = os.path.join(city_output_path, '.processing_lock')
    try:
        fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        print(f"City {city_name} is already being processed by another worker. Skipping.")
        return
    
    # Dynamically assign GPU from the queue
    gpu_id = gpu_queue.get()
    import torch
    device = torch.device(f"cuda:{gpu_id}")

    # Initialize logger for the city
    log_dir_root = os.path.join(os.path.dirname(__file__), "logs_pipeline")
    logger = setup_logger(log_dir_root, city_name)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    logger.info(f"[{city_name}] Started processing city. Claimed GPU: {device}")
    print(f"[{city_name}] Started processing city. Claimed GPU: {device}")

    try:
        city_path = os.path.join(cfg.input.dir_root, city_name)
        sequence_folders = sorted([ d for d in os.listdir(city_path) if os.path.isdir(os.path.join(city_path, d))])
        
        if not sequence_folders:
            logger.warning(f"No sequence folders found in {city_path}. Skipping city.")
            return
        
        # load models once per city
        generator, sam_pipeline = load_models(cfg, device, logger)
        class_to_prompt = cfg.input.get("prompts_seg", {})   
        classes = list(class_to_prompt.keys())
        for sequence_id in tqdm(sequence_folders, desc=f"{city_name} Sequences"):
            try:
                process_sequence(sequence_id, city_path, classes, class_to_prompt, sam_pipeline, generator, cfg, logger)
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"FAILURE in sequence {sequence_id}: {e}")
                logger.error(traceback.format_exc())
                continue # Continue to next sequence despite failure in current one
        logger.info(f"[{city_name}] Completed processing city.")
        print(f"[{city_name}] Completed processing city.")

        with open(completed_file, 'a') as f:
            # Request exclusive lock. If another script is writing, this will pause and wait.
            fcntl.flock(f, fcntl.LOCK_EX) 
            f.write(city_name + '\n')
            f.flush() # Force OS to write to disk immediately
            fcntl.flock(f, fcntl.LOCK_UN) # Release the lock
    except Exception as e:
        logger.error(f"[{city_name}] City-level error: {e}")
        logger.error(traceback.format_exc())
        
    finally:
        # Clean up and release the GPU for the next city
        try: del generator, sam_pipeline
        except: pass
        
        gc.collect()
        torch.cuda.empty_cache()
        sam_pipeline.shutdown()
        
        # Return the GPU ID to the queue so another city can use it
        gpu_queue.put(gpu_id)
        logger.info(f"[{city_name}] VRAM cleared. Released GPU {device} back to queue.")


@hydra.main(config_path=".", config_name="config_pipeline")
def run(cfg: DictConfig):

    mp.set_start_method('spawn', force=True)  # For safe multiprocessing with PyTorch and CUDA
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs found. This pipeline requires at least one GPU.")
    
    print(f"\n Detected {num_gpus} GPUs. Initializing city-level multiprocessing pool...\n")

    # create a queue and populate it with available GPU IDs
    manager = mp.Manager()
    gpu_queue = manager.Queue()
    for i in range(num_gpus):
        gpu_queue.put(i)

    # Load already-completed cities from the tracking file
    completed_file = os.path.join(os.path.dirname(__file__), "completed_cities.txt")
    completed = set()
    if os.path.exists(completed_file):
        with open(completed_file, 'r') as f:
            fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file for reading
            completed = {line.strip() for line in f if line.strip()}
            fcntl.flock(f, fcntl.LOCK_UN)  # Unlock after reading
    
    # Gather all tasks(cities), skipping already completed ones
    city_folders = sorted([
        d for d in os.listdir(cfg.input.dir_root)
        if os.path.isdir(os.path.join(cfg.input.dir_root, d)) and d not in completed
    ])

    if completed:
        print(f" Skipping {len(completed)} already completed cities.")
        print(f"Skipping already completed cities: {', '.join(completed)}")
    print(f" Total cities to process: {len(city_folders)}")

    # build arguments for the pool
    worker_args = []
    for city in city_folders:
        worker_args.append((city, cfg, gpu_queue, completed_file))
    
    #  Run the pool - As soon as a worker finishes a city, it automatically grabs the next one
    with mp.Pool(processes=num_gpus) as pool:
        pool.map(process_city_worker, worker_args)
    print("\n All cities processed. Pipeline complete.")


if __name__ == "__main__":
    run()