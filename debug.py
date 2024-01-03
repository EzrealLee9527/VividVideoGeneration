import torch
import torch.nn as nn
import sys
from animatediff.magic_animate.unet_controlnet import UNet3DConditionModel
from omegaconf import OmegaConf

pretrained_model_path =  '/huggingface00/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9/unet'

inference_config = "configs/inference/magic_inference.yaml"
inference_config = OmegaConf.load(inference_config)
model = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))
for name, param in model.named_parameters():
    # if "appearance_encoder" in name or 'contol' in 
    if "motion" in name:
        print(name)

# appearance_controlnet_motion_checkpoint_path = "/work00/magic_animate_unofficial/outputs/aa_train_stage12_celebv-2023-12-19T05-00-26/checkpoints/checkpoint-steps15000.ckpt"
# appearance_controlnet_motion_checkpoint_path = torch.load(appearance_controlnet_motion_checkpoint_path, map_location="cpu")
# org_state_dict = appearance_controlnet_motion_checkpoint_path["state_dict"] if "state_dict" in appearance_controlnet_motion_checkpoint_path else appearance_controlnet_motion_checkpoint_path   

# example: module.unet.up_blocks.3.motion_modules.2.temporal_transformer.transformer_blocks.0.attention_blocks.1.to_v.weight
# for name, param in org_state_dict.items():
#     # if "appearance_encoder" in name or 'contol' in 
#     if "motion" in name:
#         print(name)