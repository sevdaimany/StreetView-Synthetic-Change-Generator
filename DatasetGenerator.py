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
from diffusers import FluxControlInpaintPipeline, FluxMultiControlNetModel, FluxControlNetModel, FluxControlPipeline
import os

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
            self.depth_processor = DPTImageProcessor.from_pretrained(cfg.model.depth_estimation)
            self.depth_model = DPTForDepthEstimation.from_pretrained(cfg.model.depth_estimation,
                use_safetensors=True).to(self.device)


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

    def inpainting(self, img, mask, prompt, control_images, negative_prompt=""):
        """Inpaints the masked region of an image using the FluxFillPipeline."""
        if len(control_images) == 2: # both depth and canny
                controlnet_conditioning_scale = [0.85, 0.65]  # [depth, canny] tune
                control_guidance_start = [0.0, 0.0]
                control_guidance_end   = [1.0, 1.0]
        if len(control_images) == 3:
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
                generator=torch.Generator(self.device).manual_seed(0),
            ).images[0]
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
                control_image.save(os.path.join(self.cfg.input.project_path, self.cfg.output.depth_results, f"{image_name.split('.')[0]}_depth.png"))
            control_images.append(control_image)

        if self.cfg.input.canny:
            control_image = self.make_canny_control(img)
            if save:
                control_image.save(os.path.join(self.cfg.input.project_path, self.cfg.output.edge_detection_results, f"{image_name.split('.')[0]}_canny.png"))
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
        
    def make_depth_control(self, rgb_pil):
        
        self.depth_model.eval()
        img = rgb_pil.convert("RGB")
        inputs = self.depth_processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth  # [1, H, W]

        # Resize depth to original size
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=img.size[::-1],  # (H, W)
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        return self.normalize_depth_to_8bit(depth)

    def dilate_mask(self, mask, radius=8):
        """Often improves seams: dilate mask a bit so model repaints edges cleanly."""
        m = np.array(mask.cpu().numpy().astype(np.uint8) * 255)
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
        

    def save_image(self, img, title="Image", save_path=None):
        """Utility to display and optionally save images."""
        # plt.figure(figsize=(10, 10))
        # plt.imshow(img)
        # plt.title(title)
        # plt.axis("off")
        # if save_path:
        #     plt.savefig(save_path)
        # plt.show()
        img.save(save_path)


    def generate_red_herring_pair(self, img, target_class, change_prompt):
            """
            Generates a change that is NOT structural (e.g., changing a window curtain).
            """
            # 1. Segment the target (e.g., 'window')
            print(f"Finding red herring candidate: {target_class}...")
            masks = self.segment(img, target_class)
            
            # Filter: Don't pick borders, we want clean central objects for red herrings
            masks = self.filter_masks(masks, remove_borders=True)
            
            if masks is None:
                print(f"No valid {target_class} found for red herring.")
                return None, None

            # Pick a random instance
            idx = random.randint(0, len(masks) - 1)
            mask = masks[idx]

            # 2. Inpaint with the distraction prompt (e.g., 'window with red flower pot')
            # We assume red herrings are generated by the 'strongest' model (Flux/SD) regardless of MAT config
            original_model_type = self.model_type
            if self.model_type == "MAT": self.model_type = "SD" # MAT can't do prompt-based red herrings
            
            inpainted_img = self.inpainting(img, mask, change_prompt)
            
            self.model_type = original_model_type # Restore
            
            return inpainted_img, mask


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

    def process_object(self, pano_img, class_name):
            """
            Finds, Masks, and Removes an object from the panorama.
            Returns:
                - Modified Panorama
                - Global Change Mask (Equirectangular)
            """
            h, w = pano_img.shape[:2]
            global_mask = np.zeros((h, w), dtype=np.uint8)
            modified_pano = pano_img.copy()
            
            # 1. Look around to find the object (Scan 4 directions)
            # In a real pipeline, you'd probably use a lightweight detector to find WHICH crop to process.
            # Here we just blindly process the "Front" view (Yaw=0) for demonstration.
            yaw, pitch = 0, 0 
            
            # 2. Extract Perspective Crop
            crop, map_x, map_y = self.cam.get_perspective_crop(pano_img, fov_deg=90, yaw=yaw, pitch=pitch)
            
            # 3. Get Mask (SAM)
            # Note: Pass the class name to SAM/GroundingDINO here
            crop_mask = self.fake_sam_inference(crop, class_name)
            
            # Only proceed if object found
            if np.sum(crop_mask) > 0:
                # 4. Inpaint (MAT) - Remove the object
                # We assume crop_mask is 1 where object IS. MAT removes it.
                inpainted_crop = self.fake_mat_inference(crop, crop_mask)
                
                # 5. Stitch Back
                # A. Stitch the modified pixels
                self.cam.stitch_crop_back(modified_pano, inpainted_crop, map_x, map_y)
                
                # B. Stitch the mask (so we have a ground truth label)
                # We need to reshape mask to have a channel dim for the stitch function
                crop_mask_3ch = np.stack([crop_mask]*3, axis=-1) 
                # We temporarily use a 3-channel mask container for the panorama
                global_mask_3ch = np.zeros((h, w, 3), dtype=np.uint8)
                
                self.cam.stitch_crop_back(global_mask_3ch, crop_mask_3ch, map_x, map_y)
                global_mask = global_mask_3ch[:,:,0] # Flatten back to 1 channel

            return modified_pano, global_mask



