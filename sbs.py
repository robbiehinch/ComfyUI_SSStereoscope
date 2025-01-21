import torch
from PIL import Image
import numpy as np
import tqdm
import torch.nn.functional as F
from scipy.ndimage import zoom,gaussian_filter

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

        batch_size, height, width, channels = base_image.shape
        out_tensors = base_image.clone()
        for i, (base_image, depth_map) in enumerate(zip(base_image, depth_map)):
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

            # Calculate the zoom factors for each axis
            zoom_factors = (
                width / depth_map.shape[0],  # Height scaling factor
                height / depth_map.shape[1],  # Width scaling factor
            )
            
            # Use zoom with order=0 for nearest-neighbor interpolation
            # Resize depth map to match base image using NumPy (nearest-neighbor)
            # depth_map_resized = np.array(Image.fromarray(depth_map_np).resize((width, height), Image.NEAREST))
            depth_map_resized = zoom(depth_map, zoom_factors, order=0)

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
            out_tensors[i, :, :, :] = sbs_image_tensor

        return out_tensors.unsqueeze(0)
    
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

    # def ShiftedImageTensorTry(self, base_image, depth_map, depth_scale, mode="Left"):
    #     """
    #     Shift the base image using a depth map and return the shifted image.

    #     Parameters:
    #     - base_image: PyTorch tensor of shape (1, C, H, W)
    #     - depth_map: PyTorch tensor of shape (1, 1, H, W)
    #     - depth_scale: Integer representing the scaling factor for depth.
    #     - mode: "Left", "Right", or "Cross-eyed" to select the shifted image direction.

    #     Returns:
    #     - shifted_image_tensor: The shifted image as a PyTorch tensor of shape (1, C, H, 2W)
    #     """
    #     # Remove batch dimensions
    #     # image = base_image.squeeze(0)          # Shape: (C, H, W)
    #     # depth = depth_map.squeeze(0).squeeze(0)  # Shape: (H, W)

    #     batch_size, height, width, channels = base_image.shape
    #     out_tensors = base_image.clone()

    #     # Resize depth map to match base image using NumPy (nearest-neighbor)
    #     # depth_map_resized = np.array(Image.fromarray(depth_map_np).resize((width, height), Image.NEAREST))

    #     B,H,W, C = base_image.shape

    #     resized_depth_map = F.interpolate(depth_map.permute(0, 3, 1, 2), size=(height, width), mode="nearest-exact").permute(0, 2, 3, 1)
    #     depth_map_mean = resized_depth_map.mean(dim=3)

    #     # Determine flip offset based on mode
    #     flip_offset = width if mode == "Cross-eyed" else 0

    #     # Calculate pixel shifts
    #     depth_scaling = depth_scale / width
    #     pixel_shift = (depth_map_mean * depth_scaling).type(torch.int64)  # H x W

    #     # Create coordinate grids
    #     # y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    #     y_coords, x_coords = torch.meshgrid(
    #         torch.arange(height, device=base_image.device),
    #         torch.arange(width, device=base_image.device),
    #         indexing='ij'
    #     )

    #     # Calculate new x coordinates with shift
    #     new_x = x_coords + pixel_shift

    #     # Clamp new_x to [0, width-1]
    #     new_x = torch.clamp(new_x, 0, width - 1)
    #     new_x = new_x.unsqueeze(1).expand(-1, 3, -1, -1)

    #     bp = base_image.permute(0, 3, 1, 2)
    #     shifted_image = bp.clone()
    #     # flat_y = y_coords.flatten()
    #     flat_x = x_coords * width + y_coords
    #     flat_new_x = new_x * width + y_coords
    #     # flat_new_x = new_x.flatten()

    #     bp_flat = bp.view(B,C, -1)
    #     new_indices = flat_new_x.view(B,C,-1)
    #     size = H*W
    #     new_flat = flat_new_x.view(B,C,H,W)#.flatten()
    #     bp_flat = bp.view(B,C,H,W)#.flatten()
    #     # gathered = torch.gather(bp_flat, 2, new_indices)
    #     # new_vals = gathered.view(B,C,H,W)

    #     # shifted_image[:, flat_y, flat_new_x, :] = base_image[:, flat_y, flat_x, :]

    #     # bp.flatten()
    #     shifted_image.flatten().scatter_(0, new_flat.flatten(), bp_flat.flatten())
    #     p = shifted_image.permute(0, 2, 3, 1)
    #     return p.unsqueeze(0)

    def ShiftedImage(self, base_image, depth_map, depth_scale, mode="Cross-eyed"):

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
        device = base_image.device
        out_tensors = torch.zeros_like(base_image, device=device)
        for batch, (base_image, depth_map) in enumerate(zip(base_image, depth_map)):

            # Convert tensor to numpy array and then to PIL Image
            # image_np = base_image.cpu().numpy().flatten()  # Convert from CxHxW to HxWxC
            image_np = base_image.cpu().numpy().flatten()  # Convert from CxHxW to HxWxC
            # image_np *= 255
            # image_np = image_np.astype(np.uint8)
            # image = Image.fromarray((image_np * 255).astype(np.uint8))  # Convert float [0,1] tensor to uint8 image

            depth_map_np = depth_map.cpu().numpy()  # Convert from CxHxW to HxWxC
            # depth_map_np = depth_map.squeeze(0).mean(2).cpu().numpy()  # Convert from CxHxW to HxWxC
            # depth_map_np = depth_map.mean(2).cpu().numpy()  # Convert from CxHxW to HxWxC
            # depth_map_np = depth_map.mean(2).cpu().numpy()  # Convert from CxHxW to HxWxC
            # depth_map_np *= 255
            # depth_map_img = Image.fromarray((depth_map_np * 255).astype(np.uint8))  # Convert float [0,1] tensor to uint8 image

            # Get dimensions and resize depth map to match base image
            # _, width, height, _ = base_image.shape
            width, height, _ = base_image.shape
            # depth_map_img = depth_map_img.resize((width, height), Image.NEAREST)

            # Calculate the zoom factors for each axis
            zoom_factors = (
                width / depth_map.shape[0],  # Height scaling factor
                height / depth_map.shape[1],  # Width scaling factor
                1,  # colour scaling factor
            )

            depth_scaling = 255 * depth_scale / width 
            
            # Use zoom with order=0 for nearest-neighbor interpolation
            # Resize depth map to match base image using NumPy (nearest-neighbor)
            # depth_map_resized = np.array(Image.fromarray(depth_map_np).resize((width, height), Image.NEAREST)).flatten()
            depth_map_resized = zoom(depth_map_np, zoom_factors, order=0).flatten()
            depth_map_resized = gaussian_filter(depth_map_resized, sigma=1)
            depth_map_scaled = depth_map_resized * depth_scaling
            depth_map_scaled = depth_map_scaled.astype(int)
            depth_map_scaled *= 3

            # shifted_image = np.zeros((height, width, 3), dtype=np.uint8)
            shifted_image = np.zeros(image_np.size)

            # Create an empty image for the side-by-side result

            pbar = ProgressBar(height)

            # Fill the base images 
            # for y in range(height):
            #     for x in range(width):
            #         color = image.getpixel((x, y))
            #         shifted_image[y, x] = color

            dpeth_max = depth_map_resized.max()

            # generating the shifted image
            full_width = width * 3
            for row_start in tqdm.tqdm(range(0, height*full_width, full_width)):
                pbar.update(1)
                next_row_start = row_start + full_width
                for pixel_index in range(row_start, next_row_start, 3):
                    # depth_value = depth_map_scaled[pixel_index]
                    # pixel_shift = int(depth_value * depth_scaling) * 3
                    # pixel_shift = int(depth_value)
                    pixel_shift = depth_map_scaled[pixel_index]
                    # depth_value = depth_map_resized[int( pixel_index / 3)]
                    # pixel_shift = depth_map_scaled[pixel_index]
                    
                    new_x = pixel_index + pixel_shift
                    rval = image_np[pixel_index]
                    gval = image_np[pixel_index+1]
                    bval = image_np[pixel_index+2]
                    redraw_end = new_x + pixel_shift + 10
                    row_end = min(redraw_end, len(shifted_image)-3)
                    # row_end = min(redraw_end, next_row_start, len(shifted_image)-3)

                    # update_array = image_np[pixel_index:pixel_index+3]
                    # for i in range(new_x, row_end-2, 3):
                    #     shifted_image[i:i+3] = update_array#image_np[pixel_index:pixel_index+3]
                    # shifted_image[new_x:row_end-1] = rval
                    # shifted_image[new_x+1:row_end] = gval
                    # shifted_image[new_x+2:row_end+1] = bval

                    for i in range(new_x, row_end, 3):
                        # if new_x + i  >= width or new_x  < 0:
                        #     break
                        # new_coords = (y, new_x + i)
                        shifted_image[i] = rval
                        shifted_image[i+1] = gval
                        shifted_image[i+2] = bval

            shifted_image_tensor = torch.tensor(shifted_image, device=device).view(width, height, 3)
            unsqueezed = shifted_image_tensor.unsqueeze(0).unsqueeze(0)
            # return shifted_image_tensor.unsqueeze(0).unsqueeze(0)
            out_tensors[batch, :, :, :] = unsqueezed

        return out_tensors.unsqueeze(0)


        # return torch.tensor(shifted_image).view(width, height, 3).unsqueeze(0).unsqueeze(0)
        # si32 = shifted_image.astype(np.float32)
        # Convert back to tensor if needed
        # shifted_image_tensor = torch.tensor(si32 / 255.0).view(1, width, height, 3).unsqueeze(0)#.unsqueeze(0)  # Convert back to CxHxW
        # shifted_image_tensor = torch.tensor(si32 / 255.0).view(width, height, 3)#.unsqueeze(0)#.unsqueeze(0)  # Convert back to CxHxW
        shifted_image_tensor = torch.tensor(shifted_image)#.view(width, height, 3)#.unsqueeze(0)#.unsqueeze(0)  # Convert back to CxHxW
        out_tensors[i, :, :, :] = shifted_image_tensor.unsqueeze(0)
        return out_tensors.unsqueeze(0)
    
    # def ShiftedImage(self, base_image, depth_map, depth_scale, mode="Left"):
    #     """
    #     Shift the base image using a depth map and return the shifted image.

    #     Parameters:
    #     - base_image: PyTorch tensor of shape (1, C, H, W)
    #     - depth_map: PyTorch tensor of shape (1, 1, H, W)
    #     - depth_scale: Integer representing the scaling factor for depth.
    #     - mode: "Left", "Right", or "Cross-eyed" to select the shifted image direction.

    #     Returns:
    #     - shifted_image_tensor: The shifted image as a PyTorch tensor of shape (1, C, H, 2W)
    #     """
    #     # Remove batch dimensions
    #     # image = base_image.squeeze(0)          # Shape: (C, H, W)
    #     # depth = depth_map.squeeze(0).squeeze(0)  # Shape: (H, W)

    #     batch_size, height, width, channels = base_image.shape
    #     out_tensors = base_image.clone()
    #     for i, (base_image, depth_map) in enumerate(zip(base_image, depth_map)):
    #         # Ensure base_image and depth_map are on CPU and convert to NumPy
    #         image_np = base_image.cpu().numpy()  # H x W x C
    #         # image_np = base_image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # H x W x C
    #         depth_map_np = depth_map.squeeze(0).cpu().numpy().mean(2)    # H x W

    #         # Normalize images if necessary
    #         if image_np.dtype != np.uint8:
    #             image_np = (image_np * 255).astype(np.uint8)
    #         if depth_map_np.dtype != np.uint8:
    #             depth_map_np = (depth_map_np * 255).astype(np.uint8)

    #         height, width, _ = image_np.shape

    #         # Resize depth map to match base image using NumPy (nearest-neighbor)
    #         depth_map_resized = np.array(Image.fromarray(depth_map_np).resize((width, height), Image.NEAREST))

    #         # Determine flip offset based on mode
    #         flip_offset = width if mode == "Cross-eyed" else 0

    #         # Calculate pixel shifts
    #         depth_scaling = depth_scale / width
    #         pixel_shift = (depth_map_resized * depth_scaling).astype(np.int32)  # H x W

    #         # Create coordinate grids
    #         y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    #         # Calculate new x coordinates with shift
    #         new_x = x_coords + pixel_shift

    #         # Clamp new_x to [0, width-1]
    #         new_x = np.clip(new_x, 0, width - 1)
    #         shifted_image = np.tile(image_np, (1, 1, 1))  # H x (2W) x C
    #         flat_y = y_coords.flatten()
    #         flat_x = x_coords.flatten()
    #         flat_new_x = new_x.flatten()

    #         shifted_image[flat_y, flat_new_x] = image_np[flat_y, flat_x]

    #         # Convert back to torch tensor
    #         shifted_image_tensor = torch.from_numpy(shifted_image.astype(np.float32) / 255.0)#.unsqueeze(0).unsqueeze(0)  # 1 x C x H x (2W)
    #         out_tensors[i, :, :, :] = shifted_image_tensor

    #     return out_tensors.unsqueeze(0)
    
        batch_size, H, W, C = base_image.shape
        depth = depth_map.mean(3)

        # # If depth_map has more channels, average them
        # if depth.ndim > 2:
        #     depth = depth.mean(dim=0)  # Shape: (H, W)

        # Normalize depth_map if necessary (assuming depth is already normalized between 0 and 1)
        # If depth is not normalized, uncomment the following line:
        # depth = (depth - depth.min()) / (depth.max() - depth.min())

        # Calculate pixel shifts based on depth_map and depth_scale
        depth_scaling = depth_scale / W
        pixel_shift = (depth * depth_scaling).floor().to(torch.int32)  # Shape: (H, W)

        # Create coordinate grids
        y_coords = torch.arange(H, device=base_image.device).view(H, 1).expand(H, W)  # Shape: (H, W)
        x_coords = torch.arange(W, device=base_image.device).view(1, W).expand(H, W)  # Shape: (H, W)

        # Calculate new x coordinates with shift
        new_x = x_coords + pixel_shift  # Shape: (H, W)

        # Clamp new_x to ensure indices are within image boundaries
        new_x = torch.clamp(new_x, 0, W - 1)  # Shape: (H, W)

        # Initialize the side-by-side (sbs) image with double the width
        sbs_image = torch.zeros(C, H, 2 * W, device=image.device)

        # Assign the original image to the left half of the sbs_image
        sbs_image[:, :, :W] = image

        # Prepare for assigning the shifted image to the right half
        # Expand new_x to match the number of channels
        new_x_expanded = new_x.unsqueeze(0).expand(C, H, W)  # Shape: (C, H, W)

        # Assign the pixels from the original image to the shifted positions in the right half
        # Using scatter to map pixel values based on new_x coordinates
        sbs_image[:, :, W:] = sbs_image[:, :, W:].scatter(2, new_x_expanded, image)

        # If mode is "Cross-eyed", flip the shifted image horizontally
        if mode.lower() == "cross-eyed":
            sbs_image[:, :, W:] = torch.flip(sbs_image[:, :, W:], dims=[2])

        # Add a batch dimension
        shifted_image_tensor = sbs_image.unsqueeze(0)  # Shape: (1, C, H, 2W)

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

        # # Ensure images are in the expected format (Batch, Channels, H, W)
        # if left_image.ndim != 4 or right_image.ndim != 4:
        #     raise ValueError(
        #         f"Images must have 4 dimensions (Batch, Channels, H, W). "
        #         f"Found left: {left_image.ndim}D, right: {right_image.ndim}D."
        #     )

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
        if left_image.ndim == 4 or right_image.ndim == 4:

            sbs_image = torch.cat((left_image, right_image), dim=2).unsqueeze(0)  # Concatenate along width (W)

        elif left_image.ndim == 3 or right_image.ndim == 3:

            sbs_image = torch.cat((left_image, right_image), dim=1).unsqueeze(0)  # Concatenate along width (W)

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

