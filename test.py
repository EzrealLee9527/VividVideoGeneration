import torch
import megfile
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
import numpy as np
from diffusers import UNetSpatioTemporalConditionModel

def setup_pipeline(args):
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, 
        torch_dtype=torch.float16, variant="fp16"
        #"stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16"
    )
    if args.unet_path:
        # unet = UNetSpatioTemporalConditionModel.from_pretrained(args.unet_path,)
        # pipe.unet = unet

        with megfile.smart_open(args.unet_path, "rb") as rbf:
            state_dict = torch.load(rbf)
            import pdb; pdb.set_trace()
            pipe.unet.load_state_dict(state_dict)
            # pipe.unet = pipe.unet.eval()
    pipe.enable_model_cpu_offload()
    # pipe.to("cuda") # it is not necessary to move pipe to GPU manually if enable model_cpu_offload, because the pipe will be moved automatically.

    # import pdb; pdb.set_trace()
    return pipe


def run(args):
    pipe = setup_pipeline(args)
    
    # Load the conditioning image
    image = load_image("images/miaoji-female-americahighschool.png")
    image = image.resize((args.width, args.height))

    generator = torch.manual_seed(99)

    frames = pipe(
        image, 
        generator=generator, 
        height=args.height, 
        width=args.width,
        num_inference_steps=args.num_inference_steps
    ).frames[0]

    frames[0].save(
        "gifs/svd_i2v_sdp/miaoji-female-americahighschool.gif",
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=500,
        loop=0
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--unet_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=576,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
    )

    args = parser.parse_args()

    run(args)