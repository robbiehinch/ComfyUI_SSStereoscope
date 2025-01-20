

from .sbs import SideBySide, ShiftedImage, PairImages

NODE_CLASS_MAPPINGS = {
    "SBS_by_SamSeen": SideBySide,
    "Shifted_By_Rob": ShiftedImage,
    "PairImages_By_Rob": PairImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SBS_by_SamSeen": "👀 Side By Side"
}
 