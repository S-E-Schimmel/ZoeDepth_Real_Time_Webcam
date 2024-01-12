from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
import torch

# Initialize ZoeDepth
conf = get_config("zoedepth", "infer") # Define model
model_zoe_n = build_model(conf) # Build the Monocular Depth Estimation model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU with NVIDIA CUDA cores if available
zoe = model_zoe_n.to(DEVICE) # Perform tensor device conversion

# Local file
from PIL import Image
image = Image.open(r"C:\Users\SESch\BEP\testimage.jpg").convert("RGB")  # load
depth = zoe.infer_pil(image) # Calculate metric depth map

# Save raw
from zoedepth.utils.misc import save_raw_16bit
fpath = r"C:\Users\SESch\BEP\output.png"
save_raw_16bit(depth, fpath)

# Colorize output
from zoedepth.utils.misc import colorize
colored = colorize(depth)

# Save colored output
fpath_colored = r"C:\Users\SESch\BEP\output_colored.png"
Image.fromarray(colored).save(fpath_colored)