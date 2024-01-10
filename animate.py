# Copyright 2023 ByteDance and/or its affiliates.
#
# Copyright (2023) MagicAnimate Authors
#
# ByteDance, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from ByteDance or
# its affiliates is strictly prohibited.
import argparse
import argparse
import datetime
import inspect
import os
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from collections import OrderedDict

import torch

from diffusers import AutoencoderKL, DDIMScheduler, UniPCMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from animatediff.magic_animate.unet_controlnet import UNet3DConditionModel
from animatediff.magic_animate.controlnet import ControlNetModel
from animatediff.magic_animate.appearance_encoder import AppearanceEncoderModel
from animatediff.magic_animate.pipeline import AnimationPipeline as TrainPipeline
from accelerate.utils import set_seed
from animatediff.utils.videoreader import VideoReader
from einops import rearrange, repeat
from megfile import smart_open
import io

class MagicAnimate(torch.nn.Module):
    def __init__(self,
                 config="configs/training/animation.yaml",
                 device=torch.device("cuda"),
                 train_batch_size=1, unet_additional_kwargs=None):
        super().__init__()

        print("Initializing MagicAnimate Pipeline...")
        *_, func_args = inspect.getargvalues(inspect.currentframe())
        func_args = dict(func_args)

        if config == "configs/training/animation.yaml":
            config = OmegaConf.load(config)
        inference_config = OmegaConf.load(config['inference_config'])
        self.device = device
        self.train_batch_size = train_batch_size

        
        motion_module = config['motion_module']

        if unet_additional_kwargs is None:
            unet_additional_kwargs = OmegaConf.to_container(inference_config.unet_additional_kwargs)

        ### >>> create animation pipeline >>> ###
        self.tokenizer = CLIPTokenizer.from_pretrained(config['pretrained_model_path'], subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(config['pretrained_model_path'], subfolder="text_encoder")
        if config['pretrained_unet_path'] != "":
            self.unet = UNet3DConditionModel.from_pretrained_2d(config['pretrained_unet_path'],
                                                                unet_additional_kwargs=unet_additional_kwargs)
        else:
            self.unet = UNet3DConditionModel.from_pretrained_2d(config['pretrained_model_path'], subfolder="unet",
                                                                unet_additional_kwargs=unet_additional_kwargs)

        ########################LLZ TODO#############################
        if "ip_ckpt" in config.keys() and config['ip_ckpt'] != "":
            image_proj_state_dict = torch.load(config["ip_ckpt"], map_location="cpu")["image_proj"]
            image_proj_state_dict = {f'image_proj_model.{k}':v for k,v in image_proj_state_dict.items()}
            m, u = self.unet.load_state_dict(image_proj_state_dict, strict=False)
            print('image_proj_state_dict keys', len(list(image_proj_state_dict.keys())))
            print('load pretrained image_proj',len(m),len(u))


        # if 'pretrained_appearance_encoder_path' in config.keys() and config['pretrained_appearance_encoder_path'] != '':
        #     self.appearance_encoder = AppearanceEncoderModel.from_pretrained(config['pretrained_appearance_encoder_path'],
        #                                                                      subfolder="appearance_encoder").to(self.device) 
        # else:
        #     self.appearance_encoder = AppearanceEncoderModel()
        self.appearance_encoder = AppearanceEncoderModel.from_pretrained(config['pretrained_appearance_encoder_path'],
                                                                             subfolder="appearance_encoder").to(self.device) 

        if config['pretrained_vae_path'] != "":
            self.vae = AutoencoderKL.from_pretrained(config['pretrained_vae_path'])
        else:
            self.vae = AutoencoderKL.from_pretrained(config['pretrained_model_path'], subfolder="vae")

        ### Load controlnet
        self.controlnet = ControlNetModel.from_pretrained(config['pretrained_controlnet_path'])
        # if 'pretrained_controlnet_path' in config.keys() and config['pretrained_controlnet_path'] != '':
        #     self.controlnet = ControlNetModel.from_pretrained(config['pretrained_controlnet_path'])
        # else:
        #     self.controlnet = ControlNetModel()
            
        ###########################################
        # load stage1 and stage2 trained apperance_encoder, controlnet and motion module
        """
        appearance_encoder 662                                                                                                                                                             
        controlnet: 340                                                                                                                                                                     
        motion: 560  
        origin unet: 686
        """
        if "appearance_controlnet_motion_checkpoint_path" in config.keys() and config['appearance_controlnet_motion_checkpoint_path'] != "":
            appearance_controlnet_motion_checkpoint_path = config['appearance_controlnet_motion_checkpoint_path']
            print(f"from checkpoint: {appearance_controlnet_motion_checkpoint_path}")
            with smart_open(appearance_controlnet_motion_checkpoint_path, 'rb') as f:
                buffer = io.BytesIO(f.read())
                appearance_controlnet_motion_checkpoint_path = torch.load(buffer, map_location="cpu")
            if "global_step" in appearance_controlnet_motion_checkpoint_path: print(f"global_step: {appearance_controlnet_motion_checkpoint_path['global_step']}")
            org_state_dict = appearance_controlnet_motion_checkpoint_path["state_dict"] if "state_dict" in appearance_controlnet_motion_checkpoint_path else appearance_controlnet_motion_checkpoint_path        
            
            appearance_encoder_state_dict = {}
            controlnet_state_dict = {}
            unet_state_dict = {}
            for name, param in org_state_dict.items():
                if "appearance_encoder." in name:
                    if name.startswith('module.appearance_encoder.'):
                        name = name.split('module.appearance_encoder.')[-1]
                    appearance_encoder_state_dict[name] = param
                if "controlnet." in name:
                    if name.startswith('module.controlnet.'):
                        name = name.split('module.controlnet.')[-1]
                    controlnet_state_dict[name] = param
                if "unet." in name:
                    if name.startswith('module.unet.'):
                        name = name.split('module.unet.')[-1]
                    unet_state_dict[name] = param
            print('appearance_encoder_state_dict', len(list(appearance_encoder_state_dict.keys())))
            print('controlnet_state_dict', len(list(controlnet_state_dict.keys())))
            print('unet_state_dict', len(list(unet_state_dict.keys())))
            m, u = self.appearance_encoder.load_state_dict(appearance_encoder_state_dict, strict=False)
            print(f"appearance_encoder missing keys: {len(m)}, unexpected keys: {len(u)}")
            assert len(u) == 0
            m, u = self.controlnet.load_state_dict(controlnet_state_dict, strict=False)
            print(f"controlnet missing keys: {len(m)}, unexpected keys: {len(u)}")
            assert len(u) == 0
            m, u = self.unet.load_state_dict(unet_state_dict, strict=False)
            print(f"unet missing keys: {len(m)}, unexpected keys: {len(u)}")
            assert len(u) == 0, f"unexpected keys:{u}"
        ###########################################

        # 1. unet ckpt
        # 1.1 motion module
        if unet_additional_kwargs['use_motion_module'] and motion_module != "":
            motion_module_state_dict = torch.load(motion_module, map_location="cpu")
            if "global_step" in motion_module_state_dict: func_args.update(
                {"global_step": motion_module_state_dict["global_step"]})
            motion_module_state_dict = motion_module_state_dict[
                'state_dict'] if 'state_dict' in motion_module_state_dict else motion_module_state_dict
            try:
                # extra steps for self-trained models
                state_dict = OrderedDict()
                for key in motion_module_state_dict.keys():
                    if key.startswith("module."):
                        _key = key.split("module.")[-1]
                        state_dict[_key] = motion_module_state_dict[key]
                    else:
                        state_dict[key] = motion_module_state_dict[key]
                motion_module_state_dict = state_dict
                del state_dict
                missing, unexpected = self.unet.load_state_dict(motion_module_state_dict, strict=False)
                assert len(unexpected) == 0
            except:
                _tmp_ = OrderedDict()
                for key in motion_module_state_dict.keys():
                    if "motion_modules" in key:
                        if key.startswith("unet."):
                            _key = key.split('unet.')[-1]
                            _tmp_[_key] = motion_module_state_dict[key]
                        else:
                            _tmp_[key] = motion_module_state_dict[key]
                missing, unexpected = self.unet.load_state_dict(_tmp_, strict=False)
                assert len(unexpected) == 0
                del _tmp_
            del motion_module_state_dict

        

        self.vae.to(device=self.device, dtype=torch.float16)
        self.unet.to(device=self.device, dtype=torch.float16)
        self.text_encoder.to(device=self.device, dtype=torch.float16)
        self.controlnet.to(device=self.device, dtype=torch.float16)
        self.appearance_encoder.to(device=self.device, dtype=torch.float16)

        self.pipeline = TrainPipeline(
            vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer, unet=self.unet,
            controlnet=self.controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            # NOTE: UniPCMultistepScheduler
        ).to(device)

        if 'L' in config.keys():
            self.L = config['L']
        else:
            self.L = config['validation_data']['val_video_length']
        print("Initialization Done!")

    def infer(self, source_image, image_prompts, motion_sequence, random_seed, step, guidance_scale, context, size=(512, 768),froce_text_embedding_zero=False):
        prompt = n_prompt = ""
        random_seed = int(random_seed)
        step = int(step)
        guidance_scale = float(guidance_scale)
        samples_per_video = []
        # manually set random seed for reproduction
        if random_seed != -1:
            torch.manual_seed(random_seed)
            set_seed(random_seed)
        else:
            torch.seed()

        if isinstance(motion_sequence, str):
            if motion_sequence.endswith('.mp4'):
                control = VideoReader(motion_sequence).read()
                if control[0].shape[0] != size:
                    control = [np.array(Image.fromarray(c).resize(size)) for c in control]
                control = np.array(control)
            # TODO: 需要过一遍dwpose啊！！！！
                
        else:
            control = motion_sequence

        # if source_image.shape[0] != size:
        #     source_image = np.array(Image.fromarray(source_image).resize((size, size)))
        B, H, W, C = source_image.shape

        init_latents = None
        original_length = control.shape[1]
        # offset = control.shape[1] % self.L
        # if offset > 0:
        #     control= control[:,:-offset,...]
            # control = np.pad(control, ((0, self.L - control.shape[0] % self.L), (0, 0), (0, 0), (0, 0)), mode='edge')
        generator = torch.Generator(device=self.device)
        generator.manual_seed(torch.initial_seed())

        print('source_image0', source_image.shape)
        print('control0', control.shape)
        # print('image_prompts', image_prompts.shape)


        context_frames = context["context_frames"]
        context_stride = context["context_stride"]
        context_overlap = context["context_overlap"]
        sample = self.pipeline(
            prompt,
            negative_prompt=n_prompt,
            prompt_embeddings=image_prompts,
            num_inference_steps=step,
            guidance_scale=guidance_scale,
            width=W,
            height=H,
            video_length=control.shape[1],
            controlnet_condition=control,
            init_latents=init_latents,
            generator=generator,
            appearance_encoder=self.appearance_encoder,
            source_image=source_image,
            context_frames = context_frames,
            context_stride = context_stride,
            context_overlap = context_overlap,
            froce_text_embedding_zero = froce_text_embedding_zero,
        ).videos

        # TODO: save batch个视频
        # source_images = np.array([source_image[0].cpu()] * original_length)
        # source_images = rearrange(torch.from_numpy(source_images), "t h w c -> 1 c t h w") / 255.0

        print('source_image1', source_image.shape)
        print('control1', control.shape)

        source_images = rearrange(source_image[:1,...].repeat(original_length,1,1,1), "t h w c -> 1 c t h w") 
        source_images = (source_images+1.0)/2.0
        samples_per_video.append(source_images.cpu())

        control = (control+1.0)/2.0
        control = rearrange(control[0], "t h w c -> 1 c t h w")
        samples_per_video.append(control[:, :, :original_length].cpu())

        samples_per_video.append(sample[:, :, :original_length])

        samples_per_video = torch.cat(samples_per_video)

        return samples_per_video

    def forward(self, init_latents, image_prompts, timestep, source_image, motion_sequence, guidance_scale, random_seed,froce_text_embedding_zero=False):
        """
        :param init_latents: the most important input during training
        :param timestep: another important input during training
        :param source_image: an image in np.array (b, c, h, w)
        :param motion_sequence: np array, (b, f, h, w, c) (0, 255)
        :param random_seed:
        :param size: width=512, height=768 by default
        :return:
        """
        prompt = n_prompt = ""
        random_seed = int(random_seed)

        samples_per_video = []
        # manually set random seed for reproduction
        if random_seed != -1:
            torch.manual_seed(random_seed)
            set_seed(random_seed)
        else:
            torch.seed()

        control = motion_sequence
        B, H, W, C = source_image.shape

        generator = torch.Generator(device=self.device)
        generator.manual_seed(torch.initial_seed())

        # # project from (batch_size, 257, 1280) to (batch_size, 16, 768)
        
        if image_prompts is not None:
            image_prompts_clone = image_prompts.clone().detach()
            image_prompts_clone.requires_grad_(True)
            image_prompts_clone = self.unet.image_proj_model(image_prompts_clone)
        else:
            image_prompts_clone = None


        noise_pred = self.pipeline.train(
            prompt,
            prompt_embeddings=image_prompts_clone,
            negative_prompt=n_prompt,
            timestep=timestep,
            width=W,
            height=H,
            video_length=control.shape[1],
            controlnet_condition=control,
            init_latents=init_latents,  # add noise to latents
            generator=generator,
            appearance_encoder=self.appearance_encoder,
            source_image=source_image,
            context_frames = control.shape[1],
            context_batch_size = control.shape[1],
            guidance_scale = guidance_scale,
            froce_text_embedding_zero = froce_text_embedding_zero,
        )

        return noise_pred
