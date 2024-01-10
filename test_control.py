# import os
# import math
# import wandb
# import random
# import logging
# import inspect
# import argparse
# import datetime
# import subprocess
# import torchvision
# import torch.nn.functional as F
# import torch.distributed as dist
# from torch.optim.swa_utils import AveragedModel
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# from pathlib import Path
# from omegaconf import OmegaConf
# from safetensors import safe_open
# from typing import Dict, Optional, Tuple

# import diffusers
# from diffusers import AutoencoderKL, DDIMScheduler
# from diffusers.models import UNet2DConditionModel
# from diffusers.pipelines import StableDiffusionPipeline
# from diffusers.optimization import get_scheduler
# from diffusers.utils import check_min_version
# from diffusers.utils.import_utils import is_xformers_available

# import transformers
# from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
# # from ip_adapter import IPAdapterFull
# from accelerate import Accelerator
# from einops import repeat
# from animate import MagicAnimate
# from animatediff.magic_animate.controlnet import ControlNetModel
# import importlib
# from animatediff.data.dataset import WebVid10M, PexelsDataset

# from animatediff.data.dataset import WebVid10M, PexelsDataset
from animatediff.utils.util import save_videos_grid, pad_image
from PIL import Image
from tqdm.auto import tqdm
from einops import rearrange
import torch
import numpy as np
from controlnet_aux_lib import DWposeDetector, SamDetector
import webdataset as wds
from animatediff.data.dataset_wds import S3VideosIterableDataset
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor



det_config = '/data/models/controlnet_aux/src/controlnet_aux/dwpose/yolox_config/yolox_l_8xb8-300e_coco.py'
det_ckpt = '/data/models/controlnet_aux/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
pose_config = '/data/models/controlnet_aux/src/controlnet_aux/dwpose/dwpose_config/dwpose-l_384x288.py'
pose_ckpt = '/data/models/controlnet_aux/dw-ll_ucoco_384.pth'

local_rank = 'cpu'
weight_type = 'float16'

dwpose_model = DWposeDetector(
    det_config=det_config,
    det_ckpt=det_ckpt,
    pose_config=pose_config,
    pose_ckpt=pose_ckpt,
    device=local_rank
)

model_type = "vit_t"
sam_checkpoint = "./weights/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

predictor = SamPredictor(mobile_sam)


dataset = S3VideosIterableDataset(
    ['s3://ljj/Datasets/Videos/processed/CelebV_webdataset_20231211',],
    video_length = 16,
    resolution = 512,
    frame_stride = 1,
)
# import pdb; pdb.set_trace()

dataloader = wds.WebLoader(
    dataset, 
    batch_size=1,
    shuffle=False,
    num_workers=8,
    collate_fn = None,
).with_length(len(dataset))
# pbar = tqdm()
save_num = 1000
for idx,batch in tqdm(enumerate(dataloader)):
    pixel_values = batch["pixel_values"].to(local_rank)
    video_length = pixel_values.shape[1]
    with torch.no_grad():
        pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
    with torch.inference_mode():
        video_values = rearrange(pixel_values, "b c h w -> b h w c")
        image_np = (video_values * 0.5 + 0.5) * 255
        image_np = image_np.cpu().numpy().astype(np.uint8)
        num_frames = image_np.shape[0]

        dwpose_conditions = []
        for frame_id in range(num_frames):
            pil_image = Image.fromarray(image_np[frame_id])
            dwpose_image = dwpose_model(pil_image, output_type='np', image_resolution=pixel_values.shape[-1])
            dwpose_conditions.append(dwpose_image)

            # debug
            # if accelerator.is_main_process:
            #     img = Image.fromarray(dwpose_image)
            #     img.save(f"pose_{frame_id}.jpg")

            predictor.set_image(image_np[frame_id])
            masks, _, _ = predictor.predict(<input_prompts>)

        pixel_values_pose = torch.Tensor(np.array(dwpose_conditions))
        pixel_values_pose = rearrange(pixel_values_pose, "(b f) h w c -> b c f h w", b=1)
        pixel_values_pose = pixel_values_pose.to(local_rank)
        pixel_values_pose = ((pixel_values_pose / 255.0) - 0.5) * 2

    pixel_values_pose = (pixel_values_pose+1.0)/2.0
    sample_video_values = rearrange(video_values, "(b f) h w c -> b c f h w", b=1)
    sample_video_values = (sample_video_values+1.0)/2.0

    samples = torch.concat([sample_video_values, pixel_values_pose])
    print('samples', samples.shape)
    save_path = f"debug/test_control{idx}.gif"
    save_videos_grid(samples, save_path)
    break