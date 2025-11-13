# Fixed DINOv3 model architecture
import torch
import torch.nn as nn
from transformers import AutoModel

class UNetDecoder(nn.Module):
    """U-Net decoder that converts DINOv3 features to segmentation masks"""
    
    def __init__(self, feature_dim=768, target_size=224):
        super(UNetDecoder, self).__init__()
        self.target_size = target_size
        
        # Progressive upsampling layers
        self.conv1 = nn.Conv2d(feature_dim, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Apply convolutional layers with ReLU activation
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.final_conv(x)
        
        # Ensure output is exactly the target size
        x = nn.functional.interpolate(
            x, size=(self.target_size, self.target_size), 
            mode='bilinear', align_corners=False
        )
        
        # Apply sigmoid for probability output
        x = self.sigmoid(x)
        return x


class DINOv3LakeDetector(nn.Module):
    """FIXED: Complete model with proper DINOv3 feature handling"""
    
    def __init__(self, dinov3_model_name, feature_dim=768):
        super(DINOv3LakeDetector, self).__init__()
        
        # Load pre-trained DINOv3 model (frozen for feature extraction)
        self.dinov3 = AutoModel.from_pretrained(dinov3_model_name)
        
        # Freeze DINOv3 parameters (use as feature extractor only)
        for param in self.dinov3.parameters():
            param.requires_grad = False
        
        # Trainable U-Net decoder
        self.decoder = UNetDecoder(feature_dim=feature_dim, target_size=224)
        
        print(f"✅ FIXED Model created: DINOv3 (frozen) + U-Net decoder (trainable)")
    
    def forward(self, x):
        # Extract features with DINOv3 (no gradients)
        with torch.no_grad():
            features = self.dinov3(x).last_hidden_state
            
            # Remove CLS token
            patch_features = features[:, 1:]  # Remove first token (CLS)
            batch_size, num_patches, feature_dim = patch_features.shape
            
            # FIXED: Calculate correct spatial dimensions
            # DINOv3 with 224x224 input typically gives 16x16 = 256 patches (for patch size 14)
            h = w = int(num_patches ** 0.5)
            
            # Handle case where num_patches is not a perfect square
            if h * h != num_patches:
                # Find the closest factorization
                factors = []
                for i in range(1, int(num_patches**0.5) + 1):
                    if num_patches % i == 0:
                        factors.append((i, num_patches // i))
                
                if factors:
                    # Choose the factorization closest to square
                    h, w = min(factors, key=lambda x: abs(x[0] - x[1]))
                else:
                    # Fallback: pad to make it square
                    h = w = int(num_patches ** 0.5) + 1
                    needed_patches = h * w
                    
                    if needed_patches > num_patches:
                        # Pad with zeros
                        padding = torch.zeros(batch_size, needed_patches - num_patches, feature_dim, 
                                            device=patch_features.device)
                        patch_features = torch.cat([patch_features, padding], dim=1)
                    else:
                        # Truncate
                        patch_features = patch_features[:, :needed_patches]
            
            print(f"DEBUG: Reshaping {num_patches} patches to {h}x{w} = {h*w}")
            
            # Reshape to 2D feature map
            feature_map = patch_features.reshape(batch_size, h, w, feature_dim)
            feature_map = feature_map.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Generate segmentation mask with trainable decoder
        mask = self.decoder(feature_map)
        return mask

print("✅ FIXED model architecture saved to fixed_model.py")