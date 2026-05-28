import os
import gc
import cv2
import yaml
import random
import math
import torch
import numpy as np
import matplotlib 
from PIL import Image
from transformers import Sam3Model, Sam3Processor
import matplotlib.pyplot as plt
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, StableDiffusionXLInpaintPipeline, AutoencoderKL
import DAP.test.infer as DAP_infer 
from DAP.test.infer import load_model as DAP_load_model
from scipy.ndimage import gaussian_filter

class DatasetGenerator:
    def __init__(self, cfg, device):
        '''  
        Initializes the dataset generator by loading the necessary models and setting up the environment.
        '''
        self.device  = device
        self.segmodel = Sam3Model.from_pretrained(cfg.model.segmentation)
        self.segprocessor = Sam3Processor.from_pretrained(cfg.model.segmentation)
        self.cfg = cfg
        self.num_controlnets = int(cfg.input.depth) + int(cfg.input.canny) + int(cfg.input.inpaint)

        # SD XL
        vae = AutoencoderKL.from_pretrained(cfg.model.vae, torch_dtype=torch.float16)
        self.inpaint_pipeline_xl = StableDiffusionXLInpaintPipeline.from_pretrained(
            cfg.model.inpainting_xl,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16")
        self.inpaint_pipeline_xl.enable_model_cpu_offload(device=self.device) 

        # SD 1.5
        controllers = self.controlnet_models()
        self.inpaint_pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                cfg.model.inpainting,
                controlnet=controllers,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16")
        self.inpaint_pipeline.enable_model_cpu_offload(device=self.device)

        self.depth_predictor = self.DAP_depth_estimation() 


    def controlnet_models(self):
        models = []
        if self.cfg.input.depth:
            controlnet_depth = ControlNetModel.from_pretrained(self.cfg.model.controlnet_depth, torch_dtype=torch.float16, use_safetensors=True)
            models.append(controlnet_depth)
        if self.cfg.input.canny:
            controlnet_canny = ControlNetModel.from_pretrained(self.cfg.model.controlnet_canny, torch_dtype=torch.float16, use_safetensors=True)
            models.append(controlnet_canny)
        if self.cfg.input.inpaint:
            controlnet_mask = ControlNetModel.from_pretrained(self.cfg.model.controlnet, torch_dtype=torch.float16, use_safetensors=True)
            models.append(controlnet_mask)
        return models


    def inpainting(self, img, mask, prompt, control_images=None, negative_prompt="", num_controlnets=0):
        """Inpaints the masked region of an image using the FluxFillPipeline."""
        if num_controlnets == 2: # both depth and canny
                controlnet_conditioning_scale = [0.85, 0.65]  # [depth, canny] tune
                control_guidance_start = [0.0, 0.0]
                control_guidance_end   = [1.0, 1.0]
        if num_controlnets == 3:
                controlnet_conditioning_scale = [0.55, 0.55, 0.85]  # [depth, canny, inpaint_mask] tune
                control_guidance_start = [0.0, 0.0, 0.0]
                control_guidance_end   = [1.0, 1.0, 1.0]
        else: #default for single controlnet or no controlnet
                controlnet_conditioning_scale = 0.5
                control_guidance_start = 0.0
                control_guidance_end   = 1.0


        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().detach().cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            mask = Image.fromarray(mask).convert("L")

        width, height = img.size
        if num_controlnets == 0: # SDXL without ControlNet conditioning
            image = self.inpaint_pipeline_xl(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img,
            mask_image=mask,
            width=width,    
            height=height,
            generator=torch.Generator(self.device).manual_seed(0),
            ).images[0]

        else:   # SD 1.5 with ControlNet conditioning
            image = self.inpaint_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img,
                mask_image=mask,
                width=width,    
                height=height,
                strength=self.cfg.model.strength if hasattr(self.cfg.model, 'strength') else 1.0, # default 1.0
                control_image=control_images,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
                generator=torch.Generator(self.device).manual_seed(0),
            ).images[0]

        self.flush()
        return image

    def generate_control_images(self, img, mask, image_name, save_path, save=False):
        control_images = []
        if self.cfg.input.depth:
            control_image = self.make_depth_control(img)
            if save:
                depth_array = np.array(control_image)                
                if len(depth_array.shape) == 3:
                    depth_array = depth_array[:, :, 0]
                plt.figure(figsize=(10, 5))
                plt.imshow(depth_array, cmap='magma_r')
                plt.colorbar(label='Depth')
                plt.title('Depth Map')
                plt.axis('off')
                plt.savefig(os.path.join(save_path, "depth", f"{image_name.split('.')[0]}.png"))
            control_images.append(control_image)

        if self.cfg.input.canny:
            control_image = self.make_canny_control(img)
            if save:
                control_image.save(os.path.join(save_path, "edge_detection", f"{image_name.split('.')[0]}.png"))
            control_images.append(control_image)

        if self.cfg.input.inpaint:
            control_image = self.make_inpaint_condition(img, mask)
            control_images.append(control_image)
        return control_images   

    def make_inpaint_condition(self, init_image, mask_image):
        if isinstance(mask_image, torch.Tensor):
                mask_image = mask_image.squeeze().detach().cpu().numpy()
                mask_image = (mask_image * 255).astype(np.uint8)
                mask_image = Image.fromarray(mask_image).convert("L")

        init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
        mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0

        assert init_image.shape[0:1] == mask_image.shape[0:1], "image and image_mask must have the same image size"
        init_image[mask_image > 0.5] = -1.0  # set as masked pixel
        init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
        init_image = torch.from_numpy(init_image)
        return init_image

    def make_canny_control(self, img):
        # control image for stable diffusion inpainting, save the result in the edge_detection_results folder   
        image_np = np.array(img)
        control_image = cv2.Canny(image_np, 100, 200)
        edges = np.stack([control_image] * 3, axis=-1)
        control_image = Image.fromarray(edges)
        return control_image

    def normalize_depth_to_8bit(self, depth):
        """depth: float32 HxW or HxWx1 -> 0..255"""
        d = depth.astype(np.float32)
        d = d - np.nanmin(d)
        denom = np.nanmax(d) - np.nanmin(d)
        if denom < 1e-6:
            denom = 1.0
        d = d / denom
        d8 = (d * 255.0).clip(0, 255).astype(np.uint8)
        if d8.ndim == 2:
            d8 = np.stack([d8, d8, d8], axis=-1)
        return Image.fromarray(d8)

    def DAP_depth_estimation(self):
        config_path = self.cfg.model.dap_config_path
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            print("DAP Config loaded.")

        config["load_weights_dir"] = self.cfg.model.dap_load_weights_dir
        model, _ = DAP_load_model(config)

        return model
        
    def make_depth_control(self, rgb_pil):
        depth = torch.from_numpy(DAP_infer.infer_raw(self.depth_predictor, self.device, np.array(rgb_pil))).cpu().numpy()
        depth = self.normalize_depth_to_8bit(depth)
        return depth

    
    def dilate_mask(self, mask, radius=8):
        """Often improves seams: dilate mask a bit so model repaints edges cleanly."""
        if isinstance(mask, torch.Tensor):
            # Convert tensor to numpy and scale to 0-255
            m = (mask.cpu().numpy() * 255).astype(np.uint8)
        else:
            # Assume it's already a numpy array
            m = np.array(mask).astype(np.uint8)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
        m = cv2.dilate(m, k, iterations=1)
        return Image.fromarray(m, mode="L")


    def flush(self):
        gc.collect()
        torch.cuda.empty_cache()

    def overlay_mask(self, image, masks):
        """Utility to visualize masks."""

        image = image.convert("RGBA")
        if isinstance(masks, torch.Tensor):
            masks = 255 * masks.cpu().numpy()
        
        if isinstance(masks, Image.Image):
            masks = np.array(masks)
        
        if masks.ndim == 2:
            masks = masks[None, ...]

        masks = masks.astype(np.uint8)

        n_masks = masks.shape[0]
        cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
        colors = [
            tuple(int(c * 255) for c in cmap(i)[:3])
            for i in range(n_masks)
        ]

        for mask, color in zip(masks, colors):
            mask = Image.fromarray(mask)
            overlay = Image.new("RGBA", image.size, color + (0,))
            alpha = mask.point(lambda v: int(v * 0.5))
            overlay.putalpha(alpha)
            image = Image.alpha_composite(image, overlay)
        return image

    def save_inpainted_and_mask(self, inpainted_image, mask, save_path):
        """Utility to save inpainted image and corresponding mask. in one plot with captions."""
        fig, axes = plt.subplots(1, 2, figsize=(20, 12))
    
        axes[0].imshow(mask, cmap="gray")
        axes[0].set_title("Overlay Mask")
        axes[0].axis("off")

        axes[1].imshow(inpainted_image)
        axes[1].set_title("Inpainted Image")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()


    def _to_numpy(self, tensor_or_array):
        """Safely convert tensor (any device) or array to numpy."""
        if isinstance(tensor_or_array, torch.Tensor):
            return tensor_or_array.detach().cpu().numpy()
        return np.array(tensor_or_array)
    
    def sam_weather_mask(self, img, image_name, save_path=None):
        """
        Generates a semantic-aware inpainting mask for weather/lighting transfer.
        Uses SAM3 with text prompts per semantic region.
        
        Returns:
            PIL Image (mode "L"): mask where
                255 = regenerate (sky, road, vegetation)
                80  = subtle change (buildings)
                0   = preserve (vehicles, people, signs)
        """
        self.segmodel.to(self.device)

        img_size = img.size  # (W, H)
        mask_array = np.full((img_size[1], img_size[0]), -1, dtype=np.float32)

        # Define semantic groups and their mask intensities
        region_config = [
            (["buildings", "walls", "Billboards", "vehicles"], 80),
            (["traffic signs", "pole", "traffic lights"], 0),
            (["trash cans"], 0)
            ]

        for prompts, mask_value in region_config:
            class_mask = np.zeros((img_size[1], img_size[0]), dtype=bool)
            
            # Query each specific noun phrase individually
            for prompt in prompts:
                masks = self._query_sam(img, prompt)
                if masks is None or len(masks) == 0:
                    continue
                
                # Merge all instance masks for this prompt
                for m in masks:
                    m_np = self._to_numpy(m)
                    
                    # FIX 3: Safe PIL resizing for boolean arrays (Safety net)
                    if m_np.shape != (img_size[1], img_size[0]):
                        print(f"Resizing mask for prompt '{prompt}' from {m_np.shape} to {img_size[::-1]}")
                        m_img = Image.fromarray((m_np * 255).astype(np.uint8))
                        m_resized = np.array(m_img.resize(img_size, Image.NEAREST)) > 127
                    else:
                        m_resized = m_np.astype(bool)
                        
                    class_mask |= m_resized

            # Priority logic: Lower mask_value wins (preserve > change).
            # Update only if the pixel is unset (-1) or the new value is more conservative.
            update_region = class_mask & (
                (mask_array == -1) | (mask_value < mask_array)
            )
            mask_array[update_region] = mask_value

        self.segmodel.to("cpu")

        mask_array[mask_array == -1] = 255

        # Smooth mask edges to avoid hard seams
        mask_smoothed = gaussian_filter(mask_array, sigma=3)
        mask_smoothed = np.clip(mask_smoothed, 0, 255).astype(np.uint8)

        result_mask = Image.fromarray(mask_smoothed, mode="L")

        if save_path:
            result_mask.save(os.path.join(save_path, "weather" , f"{image_name.split('.')[0]}_mask.png"))

        return result_mask
    
    def _query_sam(self, img, prompt):
        """
        Single SAM3 query for one text prompt.
        Returns list of binary mask tensors, or None.
        """
        inputs = self.segprocessor(
            images=img,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.segmodel(**inputs)

        target_sizes = [list(img.size[::-1])]  # PIL (W, H) -> Target (H, W)

        results = self.segprocessor.post_process_instance_segmentation(
            outputs,
            threshold=0.35,  
            mask_threshold=0.4,
            target_sizes=target_sizes
        )

        if not results or len(results[0]["masks"]) == 0:
            return None

        return results[0]["masks"]
    
    def segment(self, img, prompt):
        """Generates a segmentation mask for the given image and prompt using the SAM model."""

        self.segmodel.to(self.device)

        inputs = self.segprocessor(images=img, text=prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.segmodel(**inputs)

        results = self.segprocessor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.5,
                    mask_threshold=0.5,
                    target_sizes=inputs.get("original_sizes").tolist()
                )[0]

        self.segmodel.to("cpu")
        self.flush()

        return results["masks"]
    
    def verify_building_removal(self, inpainted_image, prompt_seg, mask_after_np):
        new_building_instances = self.segment(inpainted_image, prompt_seg)
        verification_status = None

        if len(new_building_instances) == 0:
            verification_status = "Removed"
        else:
            is_replaced = False
            painted_area = np.sum(mask_after_np > 0)
            
            # Check if any new building overlaps with the exact area we painted
            for new_inst in new_building_instances:
                # Assuming the single-image mask is stored under a key like 'mask'
                new_mask = np.array(new_inst.cpu().numpy()) > 0 
                
                # Find the intersecting pixels (logical AND)
                overlap = np.logical_and(new_mask, mask_after_np > 0)
                overlap_pixels = np.sum(overlap)
                
                # If a newly found building covers more than 30% of the area we just painted,
                # SDXL generated a new building instead of removing it.
                if overlap_pixels > (0.30 * painted_area):
                    is_replaced = True
                    break # No need to check other buildings
                    
            verification_status = "Replaced" if is_replaced else "Removed"
        return verification_status
        
    def inference_change_style(self, img, image_name, prompt_inpaint, save_path=None, save_all=False):


        full_mask = self.sam_weather_mask(img, image_name, save_path=save_path)
        
        control_images = self.generate_control_images(img, full_mask, image_name, save_path, save=save_all)

        inpainted_image = self.inpainting(img, full_mask, prompt_inpaint,
                        control_images=control_images, num_controlnets=self.num_controlnets)
        # print(f"After Weather inpainting: img {img.size}, mask {full_mask.size}, inpainted {inpainted_image.size}")

        # Save inpainting results
        len_prompt_toshow = min(25, len(prompt_inpaint))
        inpainted_name = f"{image_name.split('.')[0]}_{prompt_inpaint[:len_prompt_toshow]}.png"

        if save_all:
            save_path = os.path.join(save_path, "weather", inpainted_name)
            fig, axes = plt.subplots(1, 2, figsize=(20, 12))
        
            axes[0].imshow(img)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(inpainted_image)
            axes[1].set_title("Inpainted Image")
            axes[1].axis("off")
            plt.suptitle(f"Prompt: {prompt_inpaint}", fontsize=16)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.show()
        return inpainted_image



    def inference(self, img, image_name, prompt_seg, prompt_inpaint, seg_mask, negative_prompt_inpaint=None):
        
        selected_mask = seg_mask
        selected_mask = self.dilate_mask(selected_mask, radius=15)

        # Inpainting
        inpainted_image = self.inpainting(img, selected_mask, prompt_inpaint,
                        negative_prompt=negative_prompt_inpaint)

    
        return inpainted_image, selected_mask

    def inpaint_output_name(self, image_name, prompt_seg):

        inpainted_name = f"{image_name.split('.')[0]}_{prompt_seg}"
        inpainted_name += ".png"
        return inpainted_name
