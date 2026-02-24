
import cv2
import numpy as np
from transformers import pipeline
import torch
import math
import torchvision.transforms.functional as TF
import os
from PIL import Image
import torch.nn.functional as F

class Panorama:
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.depth_pipe = pipeline(task="depth-estimation", model=cfg.augmentation.depth_estimation)
        self.device = cfg.device

    def augment(self, img, save_path):

        depth_pil, raw_depth_tensor = self.depth_estimation(img)
    
        T = torch.tensor([0.0, 0.0, self.cfg.augmentation.translation_z], dtype=torch.float32, device=self.device)
        R = self._get_rotation_matrix()

        img_tensor = TF.to_tensor(img).to(self.device) 
        _, H, W = img_tensor.shape
        
        raw_depth_tensor = raw_depth_tensor.to(self.device)
        # 2. Add batch and channel dimensions for interpolation: shape becomes [1, 1, Hd, Wd]
        raw_depth_tensor = raw_depth_tensor.unsqueeze(0).unsqueeze(0)
        depth_tensor = torch.nn.functional.interpolate(raw_depth_tensor, size=(H, W), mode="nearest")
        depth_tensor = depth_tensor.squeeze(0)

        warped_img_tensor, valid_mask_tensor = self.simulate_panorama_ego_motion(
            img_tensor, depth_tensor, R, T
        )
        print(f"[{save_path}] Warped shape: {warped_img_tensor.shape} | Holes: {(valid_mask_tensor == 0).sum().item()}")
        
        warped_img = TF.to_pil_image(warped_img_tensor.cpu())
        valid_mask = TF.to_pil_image(valid_mask_tensor.cpu())

        return warped_img, valid_mask, depth_pil
        

    def depth_estimation(self, images):
        """Handles both single images and batches for the HF pipeline."""
        results = self.depth_pipe(images)
        
        if isinstance(images, list):
            # Return both lists: the PIL images for saving, and raw tensors for math
            return [res["depth"] for res in results], [res["predicted_depth"] for res in results]

        return results["depth"], results["predicted_depth"]


    def simulate_panorama_ego_motion(self, image, depth, R, T, edge_threshold=1.5):
        """
        3D panorama warping with Depth Smoothing and 2x2 Splatting.
        """
        C, H, W = image.shape
        device = image.device
        
        # ---------------------------------------------------------
        # 1. DEPTH SMOOTHING (Fixes wavy roads and distorted walls)
        # ---------------------------------------------------------
        # Apply a gentle average pool to flatten the depth map's micro-bumps
        D = depth.unsqueeze(0) # [1, 1, H, W]
        D = F.avg_pool2d(D, kernel_size=5, stride=1, padding=2).squeeze() # [H, W]
        
        # ---------------------------------------------------------
        # 2. SPHERICAL COORDINATES & EDGE MASKING
        # ---------------------------------------------------------
        v_idx, u_idx = torch.meshgrid(torch.arange(H, device=device), 
                                    torch.arange(W, device=device), 
                                    indexing='ij')
        
        theta = ((u_idx.float() / (W - 1)) - 0.5) * 2 * math.pi
        phi = (0.5 - (v_idx.float() / (H - 1))) * math.pi
        
        # Edge mask to prevent foreground smearing (Flying Pixels)
        dy = torch.abs(D[1:, :] - D[:-1, :])
        dx = torch.abs(D[:, 1:] - D[:, :-1])
        dy = F.pad(dy, (0, 0, 0, 1))
        dx = F.pad(dx, (0, 1, 0, 0))
        edge_mask = (dx < edge_threshold) & (dy < edge_threshold)
        
        # ---------------------------------------------------------
        # 3. 3D PROJECTION & EGO-MOTION
        # ---------------------------------------------------------
        X = D * torch.cos(phi) * torch.sin(theta)
        Y = D * torch.sin(phi)
        Z = D * torch.cos(phi) * torch.cos(theta)
        
        points_3d = torch.stack([X, Y, Z], dim=-1).view(-1, 3)
        
        # Apply Rotation and Translation
        points_transformed = (R @ points_3d.T).T + T.view(1, 3)
        
        X_t = points_transformed[:, 0]
        Y_t = points_transformed[:, 1]
        Z_t = points_transformed[:, 2]
        
        # ---------------------------------------------------------
        # 4. REPROJECT TO 2D
        # ---------------------------------------------------------
        D_t = torch.sqrt(X_t**2 + Y_t**2 + Z_t**2)
        theta_t = torch.atan2(X_t, Z_t)
        phi_t = torch.asin(torch.clamp(Y_t / torch.clamp(D_t, min=1e-6), min=-1.0, max=1.0))
        
        u_dest_float = ((theta_t / (2 * math.pi) + 0.5) * (W - 1))
        v_dest_float = ((0.5 - phi_t / math.pi) * (H - 1))
        
        # Base integer coordinates
        u_dest = u_dest_float.round().long()
        v_dest = v_dest_float.round().long()
        
        # ---------------------------------------------------------
        # 5. 2x2 SPLATTING (Eliminates the Screen-Door blur!)
        # ---------------------------------------------------------
        # Instead of plotting 1 point, we plot 4 points (a 2x2 square) for every pixel.
        # This physically fills the magnification gaps without algorithmic blurring.
        
        offsets = [(0,0), (1,0), (0,1), (1,1)]
        
        out_image_flat = torch.zeros((C, H * W), device=device)
        mask_flat = torch.zeros((1, H * W), device=device)
        
        pixels_flat = image.view(C, -1)


        # Create an empty canvas for the warped depth map
        out_depth_flat = torch.zeros((1, H * W), device=device)

        for du, dv in offsets:
            u_splat = u_dest + du
            v_splat = v_dest + dv
            
            # Valid boundaries + Wrap around for Panoramas (Longitude wraps!)
            u_splat = u_splat % W 
            valid = (v_splat >= 0) & (v_splat < H) & edge_mask.view(-1)
            
            u_v = u_splat[valid]
            v_v = v_splat[valid]
            D_v = D_t[valid]
            p_v = pixels_flat[:, valid]
            
            idx_1d = v_v * W + u_v
            
            # Sort descending by depth so foreground overwrites background
            sort_idx = torch.argsort(D_v, descending=True)
            idx_1d_sorted = idx_1d[sort_idx]
            p_v_sorted = p_v[:, sort_idx]
            
            # Scatter this specific offset
            out_image_flat.scatter_(1, idx_1d_sorted.unsqueeze(0).expand(C, -1), p_v_sorted)
            mask_flat.scatter_(1, idx_1d_sorted.unsqueeze(0), torch.ones_like(idx_1d_sorted, dtype=torch.float32).unsqueeze(0))

            # Scatter the image
            out_image_flat.scatter_(1, idx_1d_sorted.unsqueeze(0).expand(C, -1), p_v_sorted)
            # Scatter the mask
            mask_flat.scatter_(1, idx_1d_sorted.unsqueeze(0), torch.ones_like(idx_1d_sorted, dtype=torch.float32).unsqueeze(0))

        warped_image = out_image_flat.view(C, H, W)
        mask = mask_flat.view(1, H, W)
        
        return warped_image, mask

  
    def _get_rotation_matrix(self):
        angle = math.radians(self.cfg.augmentation.yaw_deg)
        return torch.tensor([
            [math.cos(angle),  0.0, math.sin(angle)],
            [0.0,              1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)]
        ], dtype=torch.float32, device=self.device)
        

