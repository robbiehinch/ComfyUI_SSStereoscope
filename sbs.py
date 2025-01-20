import torch
from PIL import Image
import numpy as np
import tqdm
import torch.nn.functional as F

from comfy.utils import ProgressBar

class SideBySide:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "depth_scale": ("INT", {"default": 30}),
                "mode": (["Parallel", "Cross-eyed"], {}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "SideBySide"
    CATEGORY = "👀 SamSeen"

    def SideBySide(self, base_image, depth_map, depth_scale, mode="Cross-eyed"):

        """
        Create a side-by-side (SBS) stereoscopic image from a standard image and a depth map.

        Parameters:
        - base_image: numpy array representing the base image.
        - depth_map: numpy array representing the depth map.
        - depth_scale: integer representing the scaling factor for depth.
        - modes: 
        "Parallel" = the right view angle is on the right side 
        "Cross-eyed" = flipped

        Returns:
        - sbs_image: the stereoscopic image.
        """

            
        # Determine the device from base_image; assume depth_map is on the same device
        device = base_image.device

        # Validate mode
        if mode not in ["Parallel", "Cross-eyed"]:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'Parallel' or 'Cross-eyed'.")

        # Preprocess base_image
        if base_image.ndim != 4:
            raise ValueError(
                f"Base image must have 4 dimensions (Batch, Channels, H, W) or (Batch, H, W, Channels). "
                f"Found shape {base_image.shape}"
            )

        # Determine if channels are first or last
        if base_image.shape[1] == 3:
            # Channels first: (Batch, 3, H, W) -> (H, W, C)
            image = base_image.squeeze(0).permute(1, 2, 0)
        elif base_image.shape[3] == 3:
            # Channels last: (Batch, H, W, 3) -> (H, W, C)
            image = base_image.squeeze(0)
        else:
            raise ValueError(
                f"Base image must have 3 channels. Found {base_image.shape[1]} or {base_image.shape[3]} channels."
            )

        # Ensure image is in [0, 255] and uint8
        image_uint8 = torch.clamp(image * 255.0, 0, 255).to(torch.uint8).to(device)  # (H, W, C)

        # Preprocess depth_map
        if depth_map.ndim != 4:
            raise ValueError(
                f"Depth map must have 4 dimensions (Batch, Channels, H, W) or (Batch, H, W, Channels). "
                f"Found shape {depth_map.shape}"
            )

        if depth_map.shape[1] == 1:
            # Shape: (Batch, 1, H, W) -> (H, W)
            depth = depth_map.squeeze(0).squeeze(0)
        elif depth_map.shape[3] == 1:
            # Shape: (Batch, H, W, 1) -> (H, W)
            depth = depth_map.squeeze(0).squeeze(3)
        elif depth_map.shape[1] > 1:
            # Shape: (Batch, C, H, W), C > 1 -> average across channels
            depth = depth_map.squeeze(0).mean(dim=0)
        elif depth_map.shape[3] > 1:
            # Shape: (Batch, H, W, C), C > 1 -> average across channels
            depth = depth_map.squeeze(0).mean(dim=-1)
        else:
            raise ValueError(
                f"Depth map must have 1 channel. Found {depth_map.shape[1]} or {depth_map.shape[3]} channels."
            )

        # Normalize depth_map to [0,1]
        depth_normalized = torch.clamp(depth, 0.0, 1.0).to(device)

        # Resize depth_map to match base_image dimensions if necessary
        H, W, _ = image_uint8.shape
        if depth_normalized.shape != (H, W):
            # Resize using nearest neighbor interpolation
            depth_normalized = F.interpolate(
                depth_normalized.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
                size=(H, W),
                mode='nearest'
            ).squeeze(0).squeeze(0)  # Remove batch and channel dimensions

        # Ensure depth_normalized is on the correct device
        depth_normalized = depth_normalized.to(device)

        # Compute shift amounts
        shift_map = (depth_normalized * depth_scale).long().to(device)  # (H, W)

        # Generate grid of indices
        y_indices, x_indices = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )  # Both (H, W)

        # Compute shifted indices based on mode
        if mode == "Parallel":
            # Left view: shift right
            left_shifted_x = x_indices + shift_map
            # Right view: shift left
            right_shifted_x = x_indices - shift_map
        else:  # Cross-eyed
            # Left view: shift left
            left_shifted_x = x_indices - shift_map
            # Right view: shift right
            right_shifted_x = x_indices + shift_map

        # Clamp shifted indices to valid range [0, W-1]
        left_shifted_x = torch.clamp(left_shifted_x, 0, W - 1)
        right_shifted_x = torch.clamp(right_shifted_x, 0, W - 1)

        # Ensure indices are of type Long for indexing
        left_shifted_x = left_shifted_x.long()
        right_shifted_x = right_shifted_x.long()

        # Create left and right views by indexing
        # image_uint8: (H, W, C), indices: (H, W)
        left_view = image_uint8[y_indices, left_shifted_x, :]   # (H, W, C)
        right_view = image_uint8[y_indices, right_shifted_x, :] # (H, W, C)

        # Concatenate left and right views side-by-side
        sbs_image = torch.cat((left_view, right_view), dim=1)  # (H, W*2, C)

        # Convert to float tensor and normalize to [0,1]
        sbs_image_tensor = sbs_image.float() / 255.0  # (H, W*2, C)

        # Permute to (C, H, W*2) and add batch dimension
        sbs_image_tensor = sbs_image_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W*2, C)

        return sbs_image_tensor.to(self.device)