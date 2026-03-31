import cv2
import numpy as np
import random
import math
# import Panorama
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor
import matplotlib.pyplot as plt
import matplotlib 
from diffusers import FluxFillPipeline
import gc
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, StableDiffusionInpaintPipeline, StableDiffusionXLControlNetInpaintPipeline, StableDiffusionXLInpaintPipeline, AutoencoderKL
from diffusers.utils import  make_image_grid
from transformers import DPTForDepthEstimation, DPTImageProcessor
from diffusers import FluxControlInpaintPipeline, FluxMultiControlNetModel, FluxControlNetModel, FluxControlPipeline, DiffusionPipeline
import os
from PIL import ImageFilter
import DAP.test.infer as DAP_infer 
from DAP.test.infer import load_model as DAP_load_model
import yaml
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from scipy.ndimage import gaussian_filter


class DatasetGenerator:
    def __init__(self, cfg):
        '''  
        Initializes the dataset generator by loading the necessary models and setting up the environment.
        '''
        # self.cam = Panorama()
        self.device  = cfg.device
        self.segmodel = Sam3Model.from_pretrained(cfg.model.segmentation)
        self.segprocessor = Sam3Processor.from_pretrained(cfg.model.segmentation)
        self.use_flux = "FLUX" in cfg.model.inpainting
        self.use_xl = "xl" in cfg.model.inpainting
        self.cfg = cfg
        self.num_controlnets = int(cfg.input.depth) + int(cfg.input.canny) + int(cfg.input.inpaint)

        # self.segformerextractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b4-finetuned-cityscapes-1024-1024")
        # self.segformermodel = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-cityscapes-1024-1024", use_safetensors=True)

        if self.use_flux:
            if self.num_controlnets == 0:
                self.inpaint_pipeline = FluxFillPipeline.from_pretrained(cfg.model.inpainting, torch_dtype=torch.bfloat16).to(self.device)
            else:
                # controlnet_union = FluxControlNetModel.from_pretrained(cfg.model.controlnet_union, torch_dtype=torch.bfloat16)
                # controlnet = FluxMultiControlNetModel([controlnet_union])
                # self.inpaint_pipeline = FluxControlInpaintPipeline.from_pretrained(cfg.model.inpainting,
                #         controlnet=controlnet,
                #         torch_dtype=torch.bfloat16).to(self.device)

                controlnet_depth = FluxControlPipeline.from_pretrained(cfg.model.controlnet_depth, torch_dtype=torch.bfloat16)  
                controlnet_canny = FluxControlPipeline.from_pretrained(cfg.model.controlnet_canny, torch_dtype=torch.bfloat16)
                self.inpaint_pipeline = FluxControlInpaintPipeline.from_pretrained(cfg.model.inpainting,
                        controlnet=[controlnet_depth, controlnet_canny],
                        torch_dtype=torch.bfloat16).to(self.device)

                
            # self.inpaint_pipeline.enable_sequential_cpu_offload(device=self.device) # Very slow
            # self.inpaint_pipeline.enable_model_cpu_offload(device=self.device) 
    
        elif self.use_xl:
            if self.num_controlnets == 0:
                vae = AutoencoderKL.from_pretrained(cfg.model.vae, torch_dtype=torch.float16)
                self.inpaint_pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                    cfg.model.inpainting,
                    vae=vae,
                    torch_dtype=torch.float16,
                    variant="fp16")
                if cfg.augmentation.refiner:
                    self.refiner = DiffusionPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-refiner-1.0",
                        text_encoder_2=self.inpaint_pipeline.text_encoder_2,
                        vae=self.inpaint_pipeline.vae,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        variant="fp16",
                    )
                    self.refiner.enable_model_cpu_offload(device=self.device) 
                self.inpaint_pipeline.enable_model_cpu_offload(device=self.device) 
            else:
                controllers = self.controlnet_models()
                vae = AutoencoderKL.from_pretrained(cfg.model.vae, torch_dtype=torch.float16)
                self.inpaint_pipeline = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                        cfg.model.inpainting,
                        controlnet=controllers,
                        vae=vae,
                        torch_dtype=torch.float16,
                        variant="fp16")
                self.inpaint_pipeline.enable_model_cpu_offload(device=self.device) 

        else:
            if self.num_controlnets == 0:
                self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    cfg.model.inpainting,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16")
                self.inpaint_pipeline.enable_model_cpu_offload(device=self.device) 
            else:
                controllers = self.controlnet_models()
                self.inpaint_pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                        cfg.model.inpainting,
                        controlnet=controllers,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        variant="fp16")
                self.inpaint_pipeline.enable_model_cpu_offload(device=self.device) 

        if cfg.input.depth:
            # self.depth_processor = DPTImageProcessor.from_pretrained(cfg.model.depth_estimation)
            # self.depth_model = DPTForDepthEstimation.from_pretrained(cfg.model.depth_estimation,
            #     use_safetensors=True).to(self.device)
            self.depth_predictor = self.DAP_depth_estimation()

    def refine_novel_view(self, warped_img, valid_mask, prompt):
        """
        Takes the mathematically warped image and uses SDXL 
        to synthesize a photorealistic, artifact-free novel view.
        """
        warped_cv = np.array(warped_img)
        valid_mask_cv = np.array(valid_mask) # 255 = valid, 0 = hole
        raw_holes_mask = cv2.bitwise_not(valid_mask_cv) # (Now: 255 = holes/noise, 0 = valid image)

        # B. Clean the image: Fix the image's micro-holes using fast classical inpainting (Telea).
        cleaned_warped_cv = cv2.inpaint(warped_cv, raw_holes_mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)

        # C. Clean the mask: Morphological OPENING
        # This completely ERASES the tiny scattered noise, leaving only the large camera-movement holes.
        kernel_small = np.ones((3, 3), np.uint8)
        clean_sdxl_mask = cv2.morphologyEx(raw_holes_mask, cv2.MORPH_OPEN, kernel_small)

        # D. Dilate the surviving large holes so SDXL can blend them smoothly
        kernel_large = np.ones((7, 7), np.uint8) 
        dilated_sdxl_mask = cv2.dilate(clean_sdxl_mask, kernel_large, iterations=1)

        # Convert back to PIL for SDXL
        final_warped_img = Image.fromarray(cleaned_warped_cv)
        final_sdxl_mask = Image.fromarray(dilated_sdxl_mask)
        final_inpainted_img = self.inpainting(final_warped_img, final_sdxl_mask, prompt)
        
        return final_inpainted_img, final_sdxl_mask

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
    
    def control_modes(self):
        modes = []
        if self.cfg.input.depth:
            modes.append(2)
        if self.cfg.input.canny:
            modes.append(0)
        return modes

    def inpainting(self, img, mask, prompt, control_images=None, negative_prompt=""):
        """Inpaints the masked region of an image using the FluxFillPipeline."""
        if self.num_controlnets == 2: # both depth and canny
                controlnet_conditioning_scale = [0.85, 0.65]  # [depth, canny] tune
                control_guidance_start = [0.0, 0.0]
                control_guidance_end   = [1.0, 1.0]
        if self.num_controlnets == 3:
                controlnet_conditioning_scale = [0.55, 0.55, 0.85]  # [depth, canny, inpaint_mask] tune
                control_guidance_start = [0.0, 0.0, 0.0]
                control_guidance_end   = [1.0, 1.0, 1.0]
        else: #default for single controlnet or no controlnet
                controlnet_conditioning_scale = 0.5
                control_guidance_start = 0.0
                control_guidance_end   = 1.0


        if self.use_flux:
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().float()

            if self.num_controlnets == 0:
                image = self.inpaint_pipeline(
                image=img,
                prompt=prompt,
                mask_image=mask,
                generator=torch.manual_seed(0)
            ).images[0]
            else:
                image = self.inpaint_pipeline(
                    image=img,
                    prompt=prompt,
                    mask_image=mask,
                    control_image=control_images,
                    # negative_prompt=negative_prompt,
                    # control_mode=self.control_modes(),
                    # generator=torch.Generator(self.device).manual_seed(0)
                    generator=torch.manual_seed(0)
                ).images[0]

        else:
            if isinstance(mask, torch.Tensor):
                mask = mask.squeeze().detach().cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                mask = Image.fromarray(mask).convert("L")
            
            if self.num_controlnets == 0:
                image = self.inpaint_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img,
                mask_image=mask,
                # width=width,    
                # height=height,
                generator=torch.Generator(self.device).manual_seed(0),
                # output_type="latent" #for augmentation
                ).images[0]

                if self.cfg.augmentation.refiner:
                    refined_image = self.refiner(prompt=prompt, image=image[None, :]).images[0]
                    # Blur the mask slightly so the transition between original and generated is seamless
                    blend_mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
                    
                    # Image.composite(foreground, background, mask), Where the mask is white, it uses the refined_image. 
                    # Where the mask is black, it strictly uses the original_warped_img!
                    image = Image.composite(refined_image, img, blend_mask)
            else:   
                image = self.inpaint_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=img,
                    mask_image=mask,
                    control_image=control_images,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    control_guidance_start=control_guidance_start,
                    control_guidance_end=control_guidance_end,
                    generator=torch.Generator(self.device).manual_seed(0),
                ).images[0]

        self.flush()
        return image

    def generate_control_images(self, img, mask, image_name, save=False):
        control_images = []
        if self.cfg.input.depth:
            control_image = self.make_depth_control(img)
            if save:
                save_path = os.path.join(self.cfg.output.base, self.cfg.output.depth_results, f"{image_name.split('.')[0]}_depth_dap.png")
                plt.figure(figsize=(10, 5))
                plt.imshow(control_image, cmap='magma_r')
                plt.colorbar(label='Depth')
                plt.title('Depth Map Image 1')
                plt.axis('off')
                plt.savefig(save_path)
                # control_image.save(save_path)
            control_images.append(control_image)

        if self.cfg.input.canny:
            control_image = self.make_canny_control(img)
            if save:
                control_image.save(os.path.join(self.cfg.output.base, self.cfg.output.edge_detection_results, f"{image_name.split('.')[0]}_canny.png"))
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
        
    # def make_depth_control(self, rgb_pil):
        
    #     self.depth_model.eval()
    #     img = rgb_pil.convert("RGB")
    #     inputs = self.depth_processor(images=img, return_tensors="pt").to(self.device)

    #     with torch.no_grad():
    #         outputs = self.depth_model(**inputs)
    #         predicted_depth = outputs.predicted_depth  # [1, H, W]

    #     # Resize depth to original size
    #     depth = torch.nn.functional.interpolate(
    #         predicted_depth.unsqueeze(1),
    #         size=img.size[::-1],  # (H, W)
    #         mode="bicubic",
    #         align_corners=False,
    #     ).squeeze().cpu().numpy()

    #     return self.normalize_depth_to_8bit(depth)

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
        m = np.array(mask.cpu().numpy().astype(np.uint8) * 255)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
        m = cv2.dilate(m, k, iterations=1)
        return Image.fromarray(m, mode="L")
    
    def select_largest_mask(self, masks):
        """
        Select the largest mask (by pixel area) from a stack of masks.
        """
        if masks.numel() == 0:
            return None

        areas = masks.flatten(1).sum(dim=1)
        largest_idx = torch.argmax(areas)

        return largest_idx

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

    def filter_masks(self, masks, remove_borders=True, border_margin=10):
            """
            Filters masks based on criteria.
            Args:
                masks: Tensor [N, H, W]
                remove_borders: If True, removes masks that touch the image edges.
            """
            if masks is None or masks.shape[0] == 0:
                return []

            valid_indices = []
            h, w = masks.shape[1], masks.shape[2]

            for i, mask in enumerate(masks):
                mask_np = mask.cpu().numpy().astype(np.uint8)
                
                # Criterion 1: Border Check
                if remove_borders:
                    # Check if any pixel in the border margin is True
                    top = np.any(mask_np[:border_margin, :])
                    bottom = np.any(mask_np[h-border_margin:, :])
                    left = np.any(mask_np[:, :border_margin])
                    right = np.any(mask_np[:, w-border_margin:])
                    
                    if top or bottom or left or right:
                        continue # Skip this mask

                # # Criterion 2: Minimum Area (avoid single pixel noise)
                # if np.sum(mask_np) < 100: 
                #     continue

                valid_indices.append(i)

            if not valid_indices:
                return None
                
            return masks[valid_indices]

    def get_pothole_sub_mask(self, road_mask, img_size):
        width, height = img_size
        
        # road_mask was 0.0-1.0 floats or booleans. -> scaled to 0-255
        road_arr = road_mask.cpu().numpy()
        mask_np = (road_arr > 0).astype(np.uint8) * 255
        
        coords = np.argwhere(mask_np > 0)
        if len(coords) == 0:
            return road_mask
        
        # Pick a random point on the road
        center_idx = np.random.randint(len(coords))
        center_y, center_x = coords[center_idx]
        
        # pothole_w = int(width * 0.06) 
        # pothole_h = int(height * 0.02)
        
        pothole_w = int(width * 0.08) 
        pothole_h = int(height * 0.04)

        pothole_mask = np.zeros_like(mask_np)
        
        # (center_x, center_y), (axes_width, axes_height), angle, startAngle, endAngle, color, thickness
        cv2.ellipse(pothole_mask, (center_x, center_y), (pothole_w, pothole_h), 0, 0, 360, 255, -1)
        
        # Both masks are now 0-255
        final_mask = cv2.bitwise_and(pothole_mask, mask_np)
        
        # Convert back to 0.0 - 1.0 float scale for the PyTorch Diffusers pipeline
        final_mask = (final_mask / 255.0).astype(np.float32)
    
        return torch.from_numpy(final_mask).to(self.device)

    def generate_weather_mask(self, img, softness=True):
        """
        Generate a semantic-aware inpainting mask for weather transfer.
        White (255) = change, Black (0) = preserve.
        """
        # Cityscapes label IDs
        SKY_ID        = 10
        VEGETATION_ID = 8
        ROAD_ID       = 0
        SIDEWALK_ID   = 1
        CAR = 13
        TERRAIN = 11

        BUILDING_ID   = 2  # preserve
        TRAFFIC_LIGHT = 6
        TRAFFIC_SIGN = 7

        inputs = self.segformerextractor(images=img, return_tensors="pt")
        with torch.no_grad():
            logits = self.segformermodel(**inputs).logits  # (1, num_classes, H/4, W/4)

        seg_map = torch.argmax(logits, dim=1).squeeze().numpy()
        seg_map = Image.fromarray(seg_map.astype(np.uint8)).resize(
            img.size, resample=Image.NEAREST
        )
        seg_map = np.array(seg_map)

        # Build mask
        mask = np.zeros(seg_map.shape, dtype=np.uint8)

        REGENERATE_IDS = [SKY_ID, VEGETATION_ID, ROAD_ID, SIDEWALK_ID, CAR, TERRAIN]
        PARTIAL_IDS    = [BUILDING_ID]  # allow lighting but resist geometry change

        for label_id in REGENERATE_IDS:
            mask[seg_map == label_id] = 255

        for label_id in PARTIAL_IDS:
            mask[seg_map == label_id] = 40  # subtle — lets lighting bleed through

        if softness:
            # Blur mask edges to avoid hard seams at boundaries
            mask = gaussian_filter(mask.astype(float), sigma=3).astype(np.uint8)

        return Image.fromarray(mask, mode="L")
    
    def _to_numpy(self, tensor_or_array):
        """Safely convert tensor (any device) or array to numpy."""
        if isinstance(tensor_or_array, torch.Tensor):
            return tensor_or_array.detach().cpu().numpy()
        return np.array(tensor_or_array)
    
    def sam_weather_mask(self, img, save_path=None):
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
            # (["sky", "clouds", "sun"], 255),
            # (["roads", "asphalt", "streets"], 255),
            # (["sidewalks", "pavement"], 255),
            # (["vegetation", "trees", "grass"], 255),
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
            result_mask.save(save_path)
            print(f"Saved weather mask to {save_path}")

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
        
    def inference_change_style(self, img, image_name, prompt_seg, prompt_inpaint, seg_mask=None, negative_prompt_inpaint=None, save_all=False):
        # generate control images using edge detection and depth estimation for controlnet conditioning
        control_images = self.generate_control_images(img, None, image_name, save=save_all)
        # Inpainting
        # full_mask = Image.new("L", img.size, 255)
        # full_mask = self.generate_weather_mask(img, softness=True)

        # save mask
        # if save_all:
        #     save_path = os.path.join(self.cfg.output.base, self.cfg.output.segmentation_overlay, f"{image_name.split('.')[0]}_weather_mask.png")
        #     full_mask.save(save_path)
        #     print(f"Saved weather mask to {save_path}")

        mask_save_path = os.path.join(
            self.cfg.output.base, self.cfg.output.segmentation_overlay, f"{image_name.split('.')[0]}_weather_mask_sam_2.png"
        ) if save_all else None
        full_mask = self.sam_weather_mask(img, save_path=mask_save_path)
        
        inpainted_image = self.inpainting(img, full_mask, prompt_inpaint,
                        negative_prompt=negative_prompt_inpaint,
                        control_images=control_images)

        # Save inpainting results
        len_prompt_toshow = min(25, len(prompt_inpaint))
        inpainted_name = f"{image_name.split('.')[0]}_{prompt_inpaint[:len_prompt_toshow]}"
        if self.cfg.input.canny:
            inpainted_name += "_canny"
        if self.cfg.input.depth:
            inpainted_name += "_depth"

        inpainted_name += "_segmented"
        
        if self.use_xl:
            inpainted_name += "_sdxl"
        inpainted_name += ".png"
        
        save_path = os.path.join(self.cfg.output.base, self.cfg.output.inpainting_results, inpainted_name)
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
        print(f"Saved inpainted image to {save_path}")




    def inference(self, img, image_name, prompt_seg, prompt_inpaint, seg_mask=None, negative_prompt_inpaint=None, save_all=False):
        
        # Segmentation and Mask Generation
        if seg_mask is not None:
            print("Using provided segmentation mask.")
            mask = seg_mask
        else:
            mask = self.segment(img, prompt_seg)
            if mask.shape[0] == 0:
                print(f"No valid masks found for prompt '{prompt_seg}' in image '{image_name}'. Skipping inpainting.")
                return img, None
            print(f"Generated mask shape: {mask.shape}")
            if save_all:
                overlay = self.overlay_mask(img, mask)
                seg_save_path = os.path.join(self.cfg.output.base, self.cfg.output.segmentation_overlay, f"{os.path.basename(image_name).split('.')[0]}_{prompt_seg}_{self.cfg.model.segmentation.split('/')[-1]}.png") 
                overlay.save(seg_save_path)
                print(f"Saved segmentation overlay to {seg_save_path}")

        # Filter masks near borders and select one for inpainting
        filtered_mask = mask
        # filtered_mask = self.filter_masks(mask)
        mask_index = self.cfg.input.mask_index
        if mask_index == -1:
            mask_index = random.randint(0, filtered_mask.shape[0] - 1)
        if mask_index == -2:
            mask_index = self.select_largest_mask(filtered_mask)
        print(f"Selected mask index: {mask_index}")
        selected_mask = filtered_mask[mask_index]

        if self.cfg.input.add_pothole and prompt_seg == "roads":
            print("Adding pothole to the mask...")
            selected_mask = self.get_pothole_sub_mask(selected_mask, img.size)
        
        
        if self.cfg.input.dilated_mask:
            selected_mask = self.dilate_mask(selected_mask, radius=15)

        # generate control images using edge detection and depth estimation for controlnet conditioning
        control_images = self.generate_control_images(img, selected_mask, image_name, save=save_all)

        # Inpainting
        inpainted_image = self.inpainting(img, selected_mask, prompt_inpaint,
                        negative_prompt=negative_prompt_inpaint,
                        control_images=control_images)

        # Save inpainting results
        if save_all:
            overlay = self.overlay_mask(img, selected_mask)
            inpainted_name = self.inpaint_output_name(self.cfg, image_name, mask_index, prompt_seg, prompt_inpaint, negative_prompt_inpaint)

            save_path = os.path.join(self.cfg.output.base, self.cfg.output.inpainting_results, inpainted_name)
            self.save_inpainted_and_mask(inpainted_image, overlay, save_path=save_path)
            
            save_path = os.path.join(self.cfg.output.base, self.cfg.output.inpaited_only_results, inpainted_name)
            inpainted_image.save(save_path)

            print(f"Saved inpainted image to {save_path}")

        # testing segmentation on inpainted image
        # after_mask = self.segment(inpainted_image, prompt_seg)
        # overlay = self.overlay_mask(inpainted_image, after_mask)
        # inpainted_image.save(os.path.join(self.cfg.output.base, self.cfg.output.segmentation_overlay, f"{os.path.basename(image_path).split('.')[0]}index{mask_index}_inpainted_{prompt_seg}_{self.cfg.model.segmentation.split('/')[-1]}.png"))
        # overlay.save(os.path.join(self.cfg.output.base, self.cfg.output.segmentation_overlay, f"{os.path.basename(image_path).split('.')[0]}index{mask_index}_inpainted_{prompt_seg}_{self.cfg.model.segmentation.split('/')[-1]}.png"))

        return inpainted_image, selected_mask

    def inpaint_output_name(self, cfg, image_name, mask_index, prompt_seg, prompt_inpaint, negative_prompt_inpaint=None):
        model = ""
        if self.use_flux:
            model = "flux"
        elif self.use_xl:
            model = "sdxl"
        else:
            model = "sd"

        len_neg_prompt_toshow = min(20, len(negative_prompt_inpaint)) if negative_prompt_inpaint else 0
        len_prompt_toshow = min(20, len(prompt_inpaint)) if prompt_inpaint else 0
        neg_prompt_part = negative_prompt_inpaint[:len_neg_prompt_toshow] if negative_prompt_inpaint else "NoPrompt"
        prompt_part = prompt_inpaint[:len_prompt_toshow] if prompt_inpaint else "PosNoPrompt"
        
        inpainted_name = f"{image_name.split('.')[0]}idx{mask_index}_{prompt_seg}_{prompt_part}_Neg{neg_prompt_part}_{model}"
        if self.cfg.input.canny:
            inpainted_name += "_canny"
        if self.cfg.input.depth:
            inpainted_name += "_depth"
        if self.cfg.input.inpaint:
            inpainted_name += "_inpaint"
        if self.cfg.input.dilated_mask:
            inpainted_name += "_dilated"
        inpainted_name += ".png"
        return inpainted_name
