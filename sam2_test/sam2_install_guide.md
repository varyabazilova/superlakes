# SAM 2 Installation Guide

## Step 1: Install SAM 2

```bash
# Activate your environment
conda activate superlakes

# Install SAM 2 dependencies
pip install torch torchvision
pip install opencv-python matplotlib
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Alternative if above fails:
# git clone https://github.com/facebookresearch/segment-anything-2.git
# cd segment-anything-2
# pip install -e .
```

## Step 2: Download Model Weights

SAM 2 needs pretrained weights. Run this in Python:

```python
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# This will automatically download the model weights (first time only)
checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

# If automatic download doesn't work, manually download from:
# https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

## Step 3: Test Installation

```python
# Quick test
import sam2
print("SAM 2 installed successfully!")
```

## Troubleshooting

If you get errors:

1. **CUDA issues**: SAM 2 works on CPU too, just slower
2. **Download issues**: Manually download weights from Facebook's repo
3. **Import errors**: Try `pip install segment-anything-2` instead

## Next Steps

Once installed, we'll create the NDWI + SAM 2 hybrid notebook!