import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
import numpy as np


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=500,
                       loop=0)

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "/data/users/jingminhao/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/0f2d55c1e358d608120344d3ea9c35fb5f2c31b3", torch_dtype=torch.float16, variant="fp16"
    #"stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

# Load the conditioning image
image = load_image("images/gaoyuanyuan.png")
image = image.resize((512, 320))

generator = torch.manual_seed(99)
frames = pipe(image, generator=generator).frames[0]

# export_to_video(frames, "generated.mp4", decode_chunk_size=8, fps=7)


frames[0].save(
    "generated.gif",
    format='GIF',
    append_images=frames[1:],
    save_all=True,
    duration=500,
    loop=0
)

print("....")
# import hf_transfer