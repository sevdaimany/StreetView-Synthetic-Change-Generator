# StreetView Synthetic Change Generator

A pipeline for generating synthetic training data for change detection models. It combines **geometric augmentation** from sequential panoramic street-view images with **semantic changes** synthesized via diffusion-based inpainting, producing realistic before/after image sequences annotated at the pixel and bounding-box level.


## Overview

Training change detection models requires paired images where the same scene appears with and without a meaningful change. Collecting such data in the real world is expensive and slow. This pipeline automates the process:

1. **Sequence-based Geometric Augmentation** — A sequence of 5 temporally adjacent images from the same street-view sequence (Panoramax API) is used as context. The viewpoint shift between captures creates a natural geometric augmentation that simulates real-world camera motion.

2. **Center-Anchored Semantic Change Synthesis** — SAM3 video tracking segments target objects (buildings, traffic signs, trash cans, etc.) across **all frames** of the sequence, yielding pixel-accurate masks with cross-frame correspondence. The synthetic change (via Stable Diffusion inpainting) is applied **only to the center image** (frame index 2 in a 5-frame sequence). All other frames are kept as context, unchanged reference views of the same scene.

3. **Per-frame Weather Augmentation** — Each frame independently has a 50% chance of receiving a randomly chosen weather style change (night, snow, autumn), simulating temporal appearance variation across the sequence.

The result is a labeled dataset with:
- Full-sequence images (one changed center + unchanged context frames)
- Binary segmentation masks for every frame where the object appears
- Bounding-box annotations in Pascal VOC format per frame
- Per-sample metadata (class, inpaint prompt, weather applied, verification status, center image name)


## Pipeline Stages

```
Panoramax sequence (5 images)
         │
         ▼
 ┌─────────────────────────────────────────────────────┐
 │  1. Load Sequence                                    │
 │     Load all frames; identify center frame (idx 2)  │
 └─────────────────────┬───────────────────────────────┘
                       │
                       ▼
 ┌─────────────────────────────────────────────────────┐
 │  2. Track & Detect  (SAM3)                          │
 │     Text-prompted segmentation on center frame       │
 │     Temporal propagation to ALL frames in sequence   │
 │     Output: per-instance mask list (one per frame)  │
 └─────────────────────┬───────────────────────────────┘
                       │
                       ▼
 ┌─────────────────────────────────────────────────────┐
 │  3. Rank & Select Instances                          │
 │     Object must appear in center frame (required)   │
 │     Rank by temporal coverage:                      │
 │       ±1 frame from center  → +3 pts each           │
 │       ±2 frames from center → +1 pt each            │
 │     Area used as tie-breaker                        │
 │     Pick top-ranked instance(s) per selection mode  │
 └─────────────────────┬───────────────────────────────┘
                       │
                       ▼
 ┌─────────────────────────────────────────────────────┐
 │  4. Weather Augmentation (per-frame, optional)      │
 │     Each frame independently: 50% chance of         │
 │     random weather style (night / snow / autumn)    │
 └─────────────────────┬───────────────────────────────┘
                       │
                       ▼
 ┌─────────────────────────────────────────────────────┐
 │  5. Synthesize Change  (Stable Diffusion)    │
 │     Inpaint the merged center mask in center image  │
 │     All other frames saved unchanged                │
 └─────────────────────┬───────────────────────────────┘
                       │
                       ▼
 ┌─────────────────────────────────────────────────────┐
 │  6. Verify (buildings only)                         │
 │     Re-run SAM3 to confirm object was removed       │
 │     or replaced — labels sample accordingly         │
 └─────────────────────┬───────────────────────────────┘
                       │
                       ▼
 ┌─────────────────────────────────────────────────────┐
 │  7. Save                                            │
 │     pipeline_data/ → images, masks, bboxes,         │
 │                       metadata.json                 │
 │     pipeline_visualization/ → inpainting grids,    │
 │                               bbox overlays         │
 └─────────────────────────────────────────────────────┘
```


## Output Format

For each processed sequence + class the pipeline writes one event folder:

```
<output_dir>/
├── pipeline_data/{city_name}/{sequence_id}/{class_name}/
│   ├── {frame_0}.png                  # frame 0 (weather-augmented or original)
│   ├── {frame_1}.png                  # frame 1
│   ├── {center_frame}.png             # center frame (unchanged, weather may apply)
│   ├── {center_frame}_changed.png     # center frame with synthetic change applied
│   ├── {frame_3}.png                  # frame 3
│   ├── {frame_4}.png                  # frame 4
│   ├── {frame_N}_mask.png             # binary mask (only for frames where object appears)
│   ├── {frame_N}_bbox.txt             # Pascal VOC bounding boxes (same frames as masks)
│   └── metadata.json                  # class, prompt, weather log, verification status,
│                                      # center image name, frames_with_object, all paths
│
└── pipeline_visualization/{city_name}/{sequence_id}/
    ├── inpainting/{class}_{center_frame}.jpg   # side-by-side mask overlay + inpainted result
    ├── bbox/{class}_{frame_N}.jpg              # per-frame bbox overlays
    ├── weather/{frame_N}_weather.jpg           # weather-augmented frames (when applied)
    ├── depth/                                  # depth maps (when enabled)
    └── edge_detection/                         # Canny edges (when enabled)
```

### metadata.json fields

| Field | Description |
|---|---|
| `city_name` | City identifier |
| `sequence_id` | Sequence folder name |
| `class_name` | Object class (e.g. `buildings`) |
| `inpaint_prompt` | Text prompt used for inpainting |
| `center_image_name` | Filename of the center frame |
| `num_instances` | Number of tracked instances used |
| `rank_score` | Temporal coverage rank per instance (max 8) |
| `track_confidences` | SAM3 tracking confidence per instance |
| `verification_status` | `"removed"` / `"replaced"` / `"Not Checked"` |
| `weather_applied` | List of `{image_name, weather_prompt}` for augmented frames |
| `frames_with_object` | Frame filenames where the tracked object is visible |
| `image_paths` | Absolute paths to all saved sequence frames |
| `mask_paths` | Absolute paths to saved per-frame masks |
| `bbox_paths` | Absolute paths to saved per-frame bbox `.txt` files |
| `changed_center_image_path` | Absolute path to the inpainted center frame |


## Supported Change Classes

Classes and prompts are fully configurable in [config_pipeline.yaml](config_pipeline.yaml).

| Class | Inpainting prompt style |
|---|---|
| `buildings` | sky / trees matching surrounding context (removal) |
| `buildings under reconstruction` | completed building (completion) |
| `traffic signs` | removal |
| `traffic lights` | removal |
| `trash cans` | removal |

Weather augmentation prompts (applied randomly per-frame):

| Prompt |
|---|
| `same street, nighttime` |
| `same street, heavy snow, overcast sky` |
| `same street, autumn, fallen leaves on the ground` |


## Results

Each row shows the **center frame** before (left), the center frame with mask overlay (center), and the **inpainted** result (right).

### Buildings — removal / facade replacement

![](results/01_to_03_buildings_QC.jpg)
![](results/02_to_03_buildings_QC.jpg)

### Cracks / Road damage

![](results/05_to_06_cracks_QC.jpg)

### Traffic lights — removal

![](results/02_to_03_traffic_signs_QC.jpg)


## Installation

```bash
conda create -n pipeline3 python=3.10 -y
conda activate pipeline3

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

pip install -e .

huggingface-cli download Insta360-Research/DAP-weights --local-dir ./vendor/DAP/weights
```

> **Note:** CUDA 12.1 is assumed. Adjust the PyTorch index URL if your CUDA version differs.


The pipeline automatically:
- Detects all available GPUs and assigns one per city worker
- Skips cities already listed in `completed_cities.txt`
- Skips sequences/classes already written to disk (restart-safe)




## Dependencies

- [SAM3](https://github.com/facebookresearch/sam3) — video object segmentation and cross-frame tracking
- [Diffusers](https://github.com/huggingface/diffusers) — Stable Diffusion XL / FLUX inpainting
- [DAP (Depth Any Panorama)](https://github.com/Insta360-Research/DAP) — monocular depth estimation (optional ControlNet guidance)
- [Panoramax API](https://panoramax.fr) — source of sequential street-view image sequences
