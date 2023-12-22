import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

# Load the conditioning image
image = load_image("gaoyuanyuan.png")
image = image.resize((512, 320))

generator = torch.manual_seed(99)
frames = pipe(image, generator=generator).frames[0]

export_to_video(frames, "generated.mp4", fps=7)
import pdb; pdb.set_trace()
print("....")
# import hf_transfer