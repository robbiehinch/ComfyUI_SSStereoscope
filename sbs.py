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

        # Ensure base_image and depth_map are on CPU and convert to NumPy
        image_np = base_image.squeeze(0).cpu().numpy()  # H x W x C
        # image_np = base_image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # H x W x C
        depth_map_np = depth_map.squeeze(0).squeeze(0).cpu().numpy().mean(2)    # H x W

        # Normalize images if necessary
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8)
        if depth_map_np.dtype != np.uint8:
            depth_map_np = (depth_map_np * 255).astype(np.uint8)

        height, width, _ = image_np.shape

        # Resize depth map to match base image using NumPy (nearest-neighbor)
        depth_map_resized = np.array(Image.fromarray(depth_map_np).resize((width, height), Image.NEAREST))

        # Determine flip offset based on mode
        flip_offset = width if mode == "Cross-eyed" else 0

        # Initialize SBS image by duplicating the base image side by side
        sbs_image = np.tile(image_np, (1, 2, 1))  # H x (2W) x C

        # Calculate pixel shifts
        depth_scaling = depth_scale / width
        pixel_shift = (depth_map_resized * depth_scaling).astype(np.int32)  # H x W

        # Create coordinate grids
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

        # Calculate new x coordinates with shift
        new_x = x_coords + pixel_shift

        # Clamp new_x to [0, width-1]
        new_x = np.clip(new_x, 0, width - 1)

        # Flatten arrays for efficient processing
        flat_y = y_coords.flatten()
        flat_x = x_coords.flatten()
        flat_new_x = new_x.flatten()

        # Calculate target positions in SBS image
        if mode == "Parallel":
            target_x = flat_new_x
        else:
            target_x = flat_new_x + width  # Shift to the other half

        # Ensure target_x does not exceed SBS image boundaries
        target_x = np.clip(target_x, 0, 2 * width - 1)

        # Assign colors to the SBS image at shifted positions
        sbs_image[flat_y, target_x] = image_np[flat_y, flat_x]

        # Convert back to torch tensor
        sbs_image_tensor = torch.from_numpy(sbs_image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)  # 1 x C x H x (2W)

        return sbs_image_tensor
    
class ShiftedImage:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "depth_scale": ("INT", {"default": 50}),
                "mode": (["Parallel", "Cross-eyed"], {}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "ShiftedImage"
    CATEGORY = "👀 SamSeen"

    def ShiftedImage(self, base_image, depth_map, depth_scale, mode="Left"):
        """
        Shift the base image using a depth map and return the shifted image.

        Parameters:
        - base_image: PyTorch tensor representing the base image.
        - depth_map: PyTorch tensor representing the depth map.
        - depth_scale: Integer representing the scaling factor for depth.
        - mode: "Left" or "Right" to select the shifted image direction.

        Returns:
        - shifted_image_tensor: The shifted image as a PyTorch tensor.
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
        image_uint8 = torch.clamp(image * 255.0, 0, 255).to(torch.uint8).to(device)

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
            # depth = depth_map.squeeze(0).mean(dim=0)
            depth = depth_map.squeeze(0).mean(2)
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
            depth_normalized = F.interpolate(
                depth_normalized.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='nearest'
            ).squeeze(0).squeeze(0)

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
            shifted_x = x_indices + shift_map
            # Right view: shift left
        else:  # Cross-eyed
            # Left view: shift left
            shifted_x = x_indices - shift_map
            # Right view: shift right

        # Clamp shifted indices to valid range [0, W-1]
        shifted_x = torch.clamp(shifted_x, 0, W - 1)

        # Ensure indices are of type Long for indexing
        shifted_x = shifted_x.long()

        # Create the shifted view by indexing
        shifted_view = image_uint8[y_indices, shifted_x, :]  # (H, W, C)

        # Convert to float tensor and normalize to [0,1]
        shifted_image_tensor = shifted_view.float() / 255.0  # (H, W, C)

        # Permute to (C, H, W) and add batch dimension
        shifted_image_tensor = shifted_image_tensor.unsqueeze(0).unsqueeze(0)  # (1, C, H, W)

        return shifted_image_tensor


class PairImages:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left_image": ("IMAGE",),
                "right_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "Pair"
    CATEGORY = "👀 SamSeen"

    def Pair(self, left_image, right_image):
        """
        Create a side-by-side (SBS) image from two images.

        Parameters:
        - left_image: PyTorch tensor representing the left image.
        - right_image: PyTorch tensor representing the right image.

        Returns:
        - sbs_image: The side-by-side image as a PyTorch tensor.
        """

        # Ensure both images have the same dimensions
        if left_image.shape != right_image.shape:
            raise ValueError(
                f"Left and right images must have the same shape. "
                f"Found left: {left_image.shape}, right: {right_image.shape}."
            )

        # Ensure images are in the expected format (Batch, Channels, H, W)
        if left_image.ndim != 4 or right_image.ndim != 4:
            raise ValueError(
                f"Images must have 4 dimensions (Batch, Channels, H, W). "
                f"Found left: {left_image.ndim}D, right: {right_image.ndim}D."
            )

        # Check if both images are on the same device
        if left_image.device != right_image.device:
            raise ValueError(
                f"Left and right images must be on the same device. "
                f"Found left: {left_image.device}, right: {right_image.device}."
            )

        # Concatenate the images side by side along the width dimension
        # (Batch, Channels, H, W) -> Concatenate along W
        # lshaped = self._reshape(left_image)
        # rshaped = self._reshape(right_image)
        # sbs_image = torch.cat((lshaped, rshaped), dim=0)  # Concatenate along width (W)
        sbs_image = torch.cat((left_image, right_image), dim=0).unsqueeze(0)  # Concatenate along width (W)

        return sbs_image

        # Convert to float tensor and normalize to [0,1]
        # sbs_image = sbs_image.float() / 255.0  # (H, W*2, C)

        # Permute to (C, H, W*2) and add batch dimension
        sbs_image = sbs_image.permute(1, 0, 2).unsqueeze(0)  # (1, 1, H, W*2, C)
        # sbs_image = sbs_image.permute(1, 0, 2).unsqueeze(0)  # (1, 1, H, W*2, C)

        return sbs_image
    
    def _reshape(self, img):
        
        # Determine if channels are first or last
        if img.shape[1] == 3:
            # Channels first: (Batch, 3, H, W) -> (H, W, C)
            return img.squeeze(0).permute(1, 2, 0)
        elif img.shape[3] == 3:
            # Channels last: (Batch, H, W, 3) -> (H, W, C)
            return img.squeeze(0)
        else:
            raise ValueError(
                f"Base image must have 3 channels. Found {base_image.shape[1]} or {base_image.shape[3]} channels."
            )

