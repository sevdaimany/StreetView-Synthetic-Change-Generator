import numpy as np
from PIL import Image
import py360convert

class PanoramaTracker:
    def __init__(self, sam_pipeline, face_resolution=512):
        self.pipeline = sam_pipeline
        self.face_res = face_resolution

    def track_360_overlap(self, pano_A, pano_B, target_class):
        h, w, _ = pano_A.shape
        
        # Initialize TWO master masks
        master_mask_A = np.zeros((h, w), dtype=np.uint8)
        master_mask_B = np.zeros((h, w), dtype=np.uint8)
        
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        
        for yaw in angles:
            print(f"\n--- Processing Perspective View: {yaw}° ---")
            
            front_A_arr = py360convert.e2p(
                pano_A, fov_deg=(90, 90), u_deg=yaw, v_deg=0.0, 
                out_hw=(self.face_res, self.face_res), in_rot_deg=0, mode='bilinear'
            )
            
            front_B_arr = py360convert.e2p(
                pano_B, fov_deg=(90, 90), u_deg=yaw, v_deg=0.0, 
                out_hw=(self.face_res, self.face_res), in_rot_deg=0, mode='bilinear'
            )
            
            front_A = Image.fromarray(front_A_arr.astype('uint8'))
            front_B = Image.fromarray(front_B_arr.astype('uint8'))
            
            self.pipeline.load_image_pair(front_A, front_B)
            matched_pairs = self.pipeline.track_class(target_class)
            
            if not matched_pairs:
                self.pipeline.clear_current_pair()
                continue
                
            shift = int(w * (yaw / 360.0))
            
            # --- LOOP THROUGH EVERY OBJECT FOUND IN THIS VIEW ---
            for pair in matched_pairs:
                obj_id = pair["instance_id"]
                mask_A = pair["before_mask"]
                mask_B = pair["after_mask"]
                
                # --- Project Mask A back to Equirectangular ---
                if mask_A is not None and mask_A.sum() > 0:
                    blank_A = {k: np.zeros((self.face_res, self.face_res, 3), dtype=np.uint8) for k in ['F', 'R', 'B', 'L', 'U', 'D']}
                    # Ensure binary mask is scaled to 255 for projection
                    blank_A['F'] = np.stack([((mask_A > 0).astype(np.uint8) * 255)]*3, axis=-1)
                    
                    rolled_A = py360convert.c2e(blank_A, h=h, w=w, cube_format='dict')[:, :, 0]
                    true_A = np.roll(rolled_A, shift=-shift, axis=1)
                    
                    # Stamp the precise obj_id onto the master mask (Thresholding removes blur)
                    master_mask_A[true_A > 128] = obj_id

                # --- Project Mask B back to Equirectangular ---
                if mask_B is not None and mask_B.sum() > 0:
                    blank_B = {k: np.zeros((self.face_res, self.face_res, 3), dtype=np.uint8) for k in ['F', 'R', 'B', 'L', 'U', 'D']}
                    blank_B['F'] = np.stack([((mask_B > 0).astype(np.uint8) * 255)]*3, axis=-1)
                    
                    rolled_B = py360convert.c2e(blank_B, h=h, w=w, cube_format='dict')[:, :, 0]
                    true_B = np.roll(rolled_B, shift=-shift, axis=1)
                    
                    # Stamp the precise obj_id
                    master_mask_B[true_B > 128] = obj_id
                    
            print(f"Tracked {len(matched_pairs)} '{target_class}' instance(s) at {yaw}°!")
            self.pipeline.clear_current_pair()

        return master_mask_A, master_mask_B