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
from controlnet_aux import DWposeDetector
from controlnet_resource.dense_aux.densepredictor import DensePosePredictor
from controlnet_resource.dense_dwpose.densedw import DenseDWposePredictor
import webdataset as wds
from face_dataset import S3VideosIterableDataset

local_rank = 0
weight_type = 'float16'
det_config = '/root/code/yangshurong/VividVideoGeneration/controlnet_resource/controlnet_aux/src/controlnet_aux/dwpose/yolox_config/yolox_l_8xb8-300e_coco.py'
det_ckpt = '/root/.cache/yangshurong/magic_pretrain/control_aux/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
pose_config = '/root/code/yangshurong/VividVideoGeneration/controlnet_resource/controlnet_aux/src/controlnet_aux/dwpose/dwpose_config/dwpose-l_384x288.py'
pose_ckpt = '/root/.cache/yangshurong/magic_pretrain/control_aux/dw-ll_ucoco_384.pth'

# dwpose_model = DWposeDetector(
#             det_config=det_config,
#             det_ckpt=det_ckpt,
#             pose_config=pose_config,
#             pose_ckpt=pose_ckpt,
#             device=local_rank)

# samdetect = SamDetector.from_pretrained("/root/.cache/yangshurong/magic_pretrain/models--ybelkada--segment-anything/checkpoints", filename="sam_vit_b_01ec64.pth", model_type='vit_b')
# densepredict = DensePosePredictor(local_rank)
dwpose_model = DenseDWposePredictor(local_rank)

dataset = S3VideosIterableDataset(
    [
        "s3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/wds/CelebV_webdataset_20231211_videoblip",
        # "s3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/wds/hdvila100m_20231216",
        "s3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/wds/pexels_20231217",
        # "s3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/wds/xiaohongshu_webdataset_20231212",
    ],
    video_length   = 16,
    resolution     = [512, 512],
    frame_stride   = 2,
    dataset_length = 100000,
    shuffle        = True,
    resampled      = True,
    
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
import cv2
save_num = 1000
for idx, batch in tqdm(enumerate(dataloader)):
    pixel_values = batch["pixel_values"].to(local_rank)
    video_length = pixel_values.shape[1]
    with torch.no_grad():
        pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
    with torch.inference_mode():
        video_values = rearrange(pixel_values, "b c h w -> b h w c")
        # image_np = (video_values * 0.5 + 0.5) * 255
        image_np = video_values.cpu().numpy().astype(np.uint8)
        num_frames = image_np.shape[0]

        densepose_list = []
        dwpose_list = []
        densepose_dwpose_list = []
        for frame_id in range(num_frames):
            pil_image = Image.fromarray(image_np[frame_id])
            dwpose_model_result_dict = dwpose_model(pil_image)
            densepose = dwpose_model_result_dict['densepose']
            dwpose = dwpose_model_result_dict['dwpose']
            densepose_dwpose = dwpose_model_result_dict['densepose_dwpose']
            densepose_list.append(densepose)
            dwpose_list.append(dwpose)
            densepose_dwpose_list.append(densepose_dwpose)


    densepose = torch.Tensor(np.array(densepose_list)).to(torch.uint8)
    densepose = rearrange(densepose, "(b f) h w c -> b c f h w", b=1)
    save_path = f"debug/my_data_{idx}_dense.gif"
    save_videos_grid(densepose.cpu(), save_path)

    dwpose = torch.Tensor(np.array(dwpose_list)).to(torch.uint8)
    dwpose = rearrange(dwpose, "(b f) h w c -> b c f h w", b=1)
    save_path = f"debug/my_data_{idx}_dwpose.gif"
    save_videos_grid(dwpose.cpu(), save_path)

    densepose_dwpose = torch.Tensor(np.array(densepose_dwpose_list)).to(torch.uint8)
    densepose_dwpose = rearrange(densepose_dwpose, "(b f) h w c -> b c f h w", b=1)
    save_path = f"debug/my_data_{idx}_dense_dw.gif"
    save_videos_grid(densepose_dwpose.cpu(), save_path)

    video_values = rearrange(video_values, '(b f) h w c -> b c f h w', b = 1)
    save_path = f"debug/my_data_{idx}.gif"
    save_videos_grid(video_values.cpu(), save_path)

    # break