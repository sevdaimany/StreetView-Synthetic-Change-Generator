import os
import shutil
import tempfile
import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_video_predictor, build_sam3_multiplex_video_predictor
from sam3.visualization_utils import prepare_masks_for_visualization
import matplotlib.pyplot as plt
import random

class SAM3CorrespondencePipeline:
    def __init__(self, use_sam3=True, device="cuda"):
        print("Initializing SAM 3 Predictor (Loading weights to VRAM)...")
        if use_sam3:
            self.predictor = build_sam3_video_predictor()
        else:
            self.predictor =  build_sam3_multiplex_video_predictor(use_fa3=False)
        self.device = device
        self.current_session_id = None
        self.temp_dir = None
        self.num_frames = 0

    def load_image_pair(self, img_1, img_2):
        """Saves images to a temp folder and starts a SAM 3 session."""
        # Clean up any previously open session just in case
        self.clear_current_pair()
        self.num_frames = 2
        
        self.temp_dir = tempfile.mkdtemp()
        img_1.save(os.path.join(self.temp_dir, "00000.jpg"))
        img_2.save(os.path.join(self.temp_dir, "00001.jpg"))
        
        response = self.predictor.handle_request(
            request=dict(type="start_session", resource_path=self.temp_dir)
        )
        self.current_session_id = response["session_id"]
        print("Image pair loaded into SAM 3 memory.")
    
    def load_image_sequence(self, images):
        """Saves an arbitrary list of images to a temp folder and starts a SAM 3 session."""
        self.clear_current_pair()
        self.num_frames = len(images)
        
        self.temp_dir = tempfile.mkdtemp()
        
        # Save dynamically based on sequence length
        for i, img in enumerate(images):
            filename = f"{i:05d}.jpg"
            img.save(os.path.join(self.temp_dir, filename))
        
        response = self.predictor.handle_request(
            request=dict(type="start_session", resource_path=self.temp_dir)
        )
        self.current_session_id = response["session_id"]
        print(f"Sequence of {self.num_frames} images loaded into SAM 3 memory.")

    def track_class(self, class_name):
        """
        Tracks a specific class separately. 
        Resets memory first so classes don't mix!
        """
        if not self.current_session_id:
            raise RuntimeError("No images loaded. Call load_image_pair() first.")
            
        # Reset the session memory so the new prompt doesn't merge with the old one
        self.predictor.handle_request(
            request=dict(type="reset_session", session_id=self.current_session_id)
        )

        # Add the text prompt
        self.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=self.current_session_id,
                frame_index=0,
                text=class_name,
            )
        )
        # detect on both images
        self.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=self.current_session_id,
                frame_index=1,
                text=class_name,
            )
        )
        
        # Propagate to Image B
        outputs_per_frame = {}
        for stream_res in self.predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=self.current_session_id)
        ):
            outputs_per_frame[stream_res["frame_index"]] = stream_res["outputs"]
            
        # Clean the outputs using Meta's official utility
        clean_outputs = prepare_masks_for_visualization(outputs_per_frame)
        
        # Extract the matched pairs
        return self._extract_matches(clean_outputs)

    def track_class_sequence(self, class_name):
        """Tracks a specific class across all loaded frames."""
        if not self.current_session_id:
            raise RuntimeError("No images loaded. Call load_image_sequence() first.")
            
        # Reset memory for fresh tracking
        self.predictor.handle_request(
            request=dict(type="reset_session", session_id=self.current_session_id)
        )

        # Add the prompt to EVERY frame so the Detector registers new appearances
        for i in range(self.num_frames):
            self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=self.current_session_id,
                    frame_index=i,
                    text=class_name,
                )
            )
        
        # Propagate through the sequence
        outputs_per_frame = {}
        for stream_res in self.predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=self.current_session_id)
        ):
            outputs_per_frame[stream_res["frame_index"]] = stream_res["outputs"]
            
        clean_outputs = prepare_masks_for_visualization(outputs_per_frame)
        return self._extract_sequence_matches(clean_outputs)

    def clear_current_pair(self):
        """Closes the active session and deletes the temp images."""
        if self.current_session_id:
            self.predictor.handle_request(
                request=dict(type="close_session", session_id=self.current_session_id)
            )
            self.current_session_id = None
            
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
        self.num_frames = 0
        

    def shutdown(self):
        """Completely shuts down the predictor and frees GPU workers."""
        self.clear_current_pair()
        self.predictor.shutdown()
        print("SAM 3 Predictor shut down safely.")

    def _extract_matches(self, clean_outputs):
        """Internal helper to parse the matched arrays"""
        matched_pairs = []
        if 0 not in clean_outputs or 1 not in clean_outputs:
            return matched_pairs
            
        outputs_0, outputs_1 = clean_outputs[0], clean_outputs[1]
        
        for obj_id, mask_a in outputs_0.items():
            if obj_id in outputs_1:
                mask_b = outputs_1[obj_id]
                
                if mask_a is None or mask_b is None:
                    continue
                    
                if mask_b.sum() > 50: 
                    matched_pairs.append({
                        "instance_id": obj_id,
                        "before_mask": mask_a,
                        "after_mask": mask_b
                    })
        return matched_pairs
    
    def _extract_sequence_matches(self, clean_outputs):
        """Internal helper to parse masks across N frames, regardless of when they appear."""
        matched_instances = []
        
        # 1. Gather EVERY unique object ID generated across ALL frames
        all_obj_ids = set()
        for frame_idx, frame_data in clean_outputs.items():
            all_obj_ids.update(frame_data.keys())
            
        # If SAM found absolutely nothing in any frame, return empty
        if not all_obj_ids:
            return matched_instances
            
        # 2. Extract masks for each unique object across the timeline
        for obj_id in all_obj_ids:
            masks = []
            valid_sequence = False
            
            for i in range(self.num_frames):
                # Check if this frame exists and contains our specific object ID
                if i in clean_outputs and obj_id in clean_outputs[i]:
                    mask = clean_outputs[i][obj_id]
                    
                    # Ensure the mask isn't empty/noise (threshold of 50 pixels)
                    if mask is not None and mask.sum() > 50:
                        masks.append(mask)
                        valid_sequence = True
                    else:
                        # Object is occluded or disappeared in this specific frame
                        masks.append(None)
                else:
                    # Object hasn't appeared yet, or is gone
                    masks.append(None)
            
            # 3. Keep the instance if it had a valid mask in at least ONE frame
            if valid_sequence:
                matched_instances.append({
                    "instance_id": obj_id,
                    "masks": masks
                })
                
        return matched_instances

        

    def visualize_correspondence(self, img_a, img_b, matched_pairs, save_path, alpha=0.5):
        """
        Visualizes matched instance masks side-by-side with consistent colors and IDs.
        
        img_a, img_b: Original images (PIL Images or numpy arrays).
        matched_pairs: The output list from our SAM 3 tracking function.
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        vis_a = np.array(img_a).astype(np.float32)
        vis_b = np.array(img_b).astype(np.float32)
        
        # Generate a distinct color palette (e.g., up to 100 instances)
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(100, 3))

        for pair in matched_pairs:
            obj_id = pair["instance_id"]
            mask_a = pair["before_mask"]
            mask_b = pair["after_mask"]

            # Safely get a unique color
            color_index = hash(str(obj_id)) % 100
            color = colors[color_index]
            
            # 1. Overlay Mask on Image A 
            if mask_a is not None and mask_a.sum() > 0:
                vis_a[mask_a > 0] = vis_a[mask_a > 0] * (1 - alpha) + color * alpha

                # Find centroid to place the text label
                y_coords, x_coords = np.where(mask_a > 0)
                cy, cx = int(y_coords.mean()), int(x_coords.mean())
                axes[0].text(cx, cy, f"ID: {obj_id}", color='white', 
                            fontsize=12, fontweight='bold', ha='center',
                            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

                
            # 2. Overlay Mask on Image B
            if mask_b is not None and mask_b.sum() > 0:
                vis_b[mask_b > 0] = vis_b[mask_b > 0] * (1 - alpha) + color * alpha
            
                # Find centroid to place the text label
                y_coords, x_coords = np.where(mask_b > 0)
                cy, cx = int(y_coords.mean()), int(x_coords.mean())
                axes[1].text(cx, cy, f"ID: {obj_id}", color='white', 
                            fontsize=12, fontweight='bold', ha='center',
                            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

        # Display Image A
        axes[0].imshow(vis_a.astype(np.uint8))
        axes[0].set_title(f"Image A (Before) - {len(matched_pairs)} Instances", fontsize=16)
        axes[0].axis("off")
        
        # Display Image B
        axes[1].imshow(vis_b.astype(np.uint8))
        axes[1].set_title(f"Image B (After) - {len(matched_pairs)} Instances", fontsize=16)
        axes[1].axis("off")
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved correspondence visualization to {save_path}")
    
    def visualize_sequence_correspondence(self, images, matched_instances, save_path, alpha=0.5):
        """Dynamically creates subplots to visualize tracking across N images."""
        num_images = len(images)
        fig, axes = plt.subplots(1, num_images, figsize=(10 * num_images, 10))
        
        # Handle case where num_images == 1 for consistency
        if num_images == 1:
            axes = [axes]
            
        vis_images = [np.array(img).astype(np.float32) for img in images]
        
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(100, 3))

        for inst in matched_instances:
            obj_id = inst["instance_id"]
            color = colors[hash(str(obj_id)) % 100]
            
            for i, mask in enumerate(inst["masks"]):
                if mask is not None and mask.sum() > 0:
                    vis_images[i][mask > 0] = vis_images[i][mask > 0] * (1 - alpha) + color * alpha

                    y_coords, x_coords = np.where(mask > 0)
                    cy, cx = int(y_coords.mean()), int(x_coords.mean())
                    axes[i].text(cx, cy, f"ID: {obj_id}", color='white', 
                                fontsize=12, fontweight='bold', ha='center',
                                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

        for i in range(num_images):
            axes[i].imshow(vis_images[i].astype(np.uint8))
            axes[i].set_title(f"Frame {i} - {len(matched_instances)} Instances", fontsize=16)
            axes[i].axis("off")
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory
        print(f"Saved sequence visualization to {save_path}")