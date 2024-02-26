import os
import math
import random
import logging
import inspect
import argparse
import datetime
import subprocess

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple
from PIL import Image

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
# from ip_adapter import IPAdapterFull

# from animatediff.data.dataset import WebVid10M, PexelsDataset
from animatediff.utils.util import save_videos_grid, pad_image
from accelerate import Accelerator
from einops import repeat
from accelerate.utils import set_seed
import webdataset as wds

from face_dataset import S3VideosIterableDataset
import webdataset as wds


import facer

def main(
        image_finetune: bool,

        name: str,
        use_wandb: bool,
        launcher: str,

        output_dir: str,

        data_module: str,
        data_class: str,
        train_data: Dict,
        validation_data: Dict,
        context: Dict,
        cfg_random_null_text: bool = True,
        cfg_random_null_text_ratio: float = 0.1,

        pretrained_model_path: str = "",
        pretrained_appearance_encoder_path: str = "",
        pretrained_controlnet_path: str = "",
        pretrained_vae_path: str = "",
        motion_module: str = "",
        appearance_controlnet_motion_checkpoint_path: str = "",
        pretrained_unet_path: str = "",
        inference_config: str = "",
        unet_checkpoint_path: str = "",
        unet_additional_kwargs: Dict = {},
        ema_decay: float = 0.9999,
        noise_scheduler_kwargs=None,

        max_train_epoch: int = -1,
        max_train_steps: int = 100,
        validation_steps: int = 100,
        validation_steps_tuple: Tuple = (-1,),

        learning_rate: float = 3e-5,
        scale_lr: bool = False,
        lr_warmup_steps: int = 0,
        lr_scheduler: str = "constant",

        trainable_modules: Tuple[str] = (None,),
        num_workers: int = 8,
        train_batch_size: int = 1,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        checkpointing_epochs: int = 5,
        checkpointing_steps: int = -1,

        mixed_precision_training: bool = True,
        enable_xformers_memory_efficient_attention: bool = True,

        valid_seed: int = 42,
        is_debug: bool = False,

        dwpose_only_face = False,
        froce_text_embedding_zero = False,

        ip_ckpt=None,
        use_2d_cross_rate: float = 0.0,
        max_cross_frame_stride_rate: int = 4,

        control_aux_type: str = 'dwpose',
        controlnet_type: str = '2d',
        controlnet_config: str = '',

        model_type: str = "unet",

        clip_image_type: str = '',
        remove_background: bool = False,
        concat_noise_image_type: str = '',
        ref_image_type: str = "origin",
        do_classifier_free_guidance: bool = True,
        add_noise_image_type: str = '',
        special_ref_index: int = -1,
        remove_referencenet: bool = False,

        
):

    # # check params is true to run
    
    # if remove_background:
    #     assert clip_image_type == 'foreground' or not use_image_encoder, "\
    #         background is removed, only can encode foreground"
    

    if 'unet.conv_in' in trainable_modules:
        assert concat_noise_image_type != "", "\
            For training conv_in, must concat_noise_image_type"

    weight_type = torch.float16
    # Accelerate
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision='fp16'
    )

    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if accelerator.is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="animatediff", name=folder_name, config=config)



    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        # OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    if accelerator.state.deepspeed_plugin is not None and \
            accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == "auto":
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = train_batch_size

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
    local_rank = accelerator.device

    # load dwpose detector, see controlnet_aux: https://github.com/patrickvonplaten/controlnet_aux
    # specify configs, ckpts and device, or it will be downloaded automatically and use cpu by default
    
    from controlnet_resource.dense_dwpose.densedw import DenseDWposePredictor
    dwpose_model = DenseDWposePredictor(local_rank)
    
    from animatediff.magic_animate.unet_model.animate import MagicAnimate
    model = MagicAnimate(config=config,
                         train_batch_size=train_batch_size,
                         device=local_rank,
                         unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs),
                         mixed_precision_training=True,
                         trainable_modules=trainable_modules,
                         is_main_process=accelerator.is_main_process,)

    # ----- load image encoder ----- #
    """
    使用IP-adapter，主要包含image_encoder，clip_image_processor和image_proj_model
    image_proj_model在Resampler里定义
    """
    arcface_encoder = None
    image_processor = None
    image_encoder = None
    face_detector = None
    if clip_image_type != "":
        
        if accelerator.is_main_process:
            print(f"use clip code image, image type is {clip_image_type}")
        image_processor = CLIPImageProcessor.from_pretrained(pretrained_model_path, subfolder="feature_extractor")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_path, subfolder="image_encoder")
        
        # from controlnet_resource.arcface_backbones import get_model
        # arcface_encoder = get_model('r100', fp16=False)
        # arcface_weight_path = '/root/.cache/yangshurong/magic_pretrain/arcface_backbone.pth'
        # arcface_encoder.load_state_dict(torch.load(arcface_weight_path))
        # arcface_encoder.to(local_rank, weight_type)
        # arcface_encoder.requires_grad_(False)
        # arcface_encoder.eval()

        image_encoder.to(local_rank, weight_type)
        image_encoder.requires_grad_(False)

        # face_detector = facer.face_detector('retinaface/mobilenet', device=local_rank)
        # face_detector.requires_grad_(False)

    # Set trainable parameters
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if accelerator.is_main_process:
        print('trainable_params', len(trainable_params))
    # with open('model.txt', 'w') as fp:
    #     for item in list(model.state_dict().keys()):
    #         fp.write("%s\n" % item)
    # fp.close()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    if accelerator.is_main_process:
        accelerator.print(f"trainable params number: {len(trainable_params)}")
        accelerator.print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        model.unet.enable_gradient_checkpointing()
        model.appearance_encoder.enable_gradient_checkpointing()
        model.controlnet.enable_gradient_checkpointing()

    
    model.to(local_rank)
    

    # Get the training dataset
    # dataset_cls = getattr(importlib.import_module(data_module, package=None), data_class)
    # train_dataset = dataset_cls(**train_data, is_image=image_finetune)
    #
    # train_dataset = PexelsDataset(**train_data)

    # distributed_sampler = DistributedSampler(
    #     train_dataset,
    #     num_replicas=accelerator.num_processes,
    #     rank=accelerator.process_index,
    #     shuffle=True,
    #     seed=global_seed,
    # )

    # # DataLoaders creation:
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=train_batch_size,
    #     shuffle=False,
    #     sampler=distributed_sampler,
    #     num_workers=num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    def worker_init_fn(_):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        worker_id = worker_info.id
        return np.random.seed(np.random.get_state()[1][0] + worker_id)

    # train_dataloader = wds.WebLoader(
    #             train_dataset, 
    #             batch_size=train_batch_size,
    #             shuffle=False,
    #             num_workers=num_workers, 
    #             worker_init_fn=None,
    #         ).with_length(len(train_dataset))
    video_length   = train_data["video_length"]
    train_dataset = S3VideosIterableDataset(
        data_dirs = train_data['data_dirs'],
        video_length   = video_length,
        resolution     = train_data['resolution'],
        frame_stride   = train_data['frame_stride'],
        dataset_length = 1000000,
        shuffle        = True,
        resampled      = True,
        max_cross_frame_stride_rate = max_cross_frame_stride_rate,
        batch_from_same_video = True,
    )

    train_dataloader = wds.WebLoader(
        train_dataset, 
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=num_workers,
        # this must be zeros since in mul GPU
        collate_fn = None,
    ).with_length(len(train_dataset))

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)

    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    num_processes = torch.cuda.device_count()
    if accelerator.is_main_process:
        print("***** Running training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num Epochs = {num_train_epochs}")
        print(f"  Instantaneous batch size per device = {train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        print(f"  Total optimization steps = {max_train_steps}")
        print(f"  num_processes = {num_processes}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    seed = 0
    set_seed(seed)
    for epoch in range(first_epoch, num_train_epochs):
        # TODO: check webdataset在多卡的随机性问题
        # train_dataloader.sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Data batch sanity check
                if global_step % 1000 == 0:
                    # pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                    pixel_values = batch['pixel_values'].cpu()
                    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                    for idx, pixel_value in enumerate(pixel_values):
                        pixel_value = pixel_value[None, ...]
                        save_videos_grid(pixel_value,
                                        f"{output_dir}/sanity_check/global_{global_step}_pixel_value.gif")

                ### >>>> Training >>>> ###

                # Convert videos to latent space            
                pixel_values = batch["pixel_values"].to(local_rank, dtype=weight_type)
                pixel_values_ref_img = pixel_values[:, special_ref_index, ...]

                # NOTE: not use_2d_cross_rate, pixel_values all set to pixel_values_ref_img
                # if np.random.rand() > use_2d_cross_rate:
                #     pixel_values = pixel_values_ref_img.repeat(1, video_length, 1, 1, 1)
                    
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")

                # work for control video process, only for pixel_values
                pixel_values_dwpose_model_result_dicts = []
                with torch.no_grad():
                    image_np = rearrange(pixel_values, "b c h w -> b h w c")
                    image_np = image_np.cpu().numpy().astype(np.uint8)
                    num_frames = image_np.shape[0]

                    dwpose_conditions = []
                    pixel_values_remove_back = []
                    for frame_id in range(num_frames):
                        pil_image = Image.fromarray(image_np[frame_id])
                        dwpose_model_result_dict = dwpose_model(pil_image)
                        pixel_values_dwpose_model_result_dicts.append(dwpose_model_result_dict)
                        dwpose_image = dwpose_model_result_dict[control_aux_type]

                        if remove_background:                        
                            pixel_values_remove = dwpose_model_result_dict['foreground']
                            pixel_values_remove_back.append(pixel_values_remove)

                        dwpose_conditions.append(dwpose_image)    

                    pixel_values_pose = torch.Tensor(np.array(dwpose_conditions))
                    pixel_values_pose = rearrange(pixel_values_pose, "(b f) h w c -> b f h w c", b=train_batch_size)
                    pixel_values_pose = pixel_values_pose.to(local_rank, dtype=weight_type)
                    pixel_values_pose = pixel_values_pose / 255.0                    

                # remove background only for ref_image
                if ref_image_type != "origin" or concat_noise_image_type != "" or add_noise_image_type != "" or clip_image_type != "" or remove_background:
                    with torch.no_grad():
                        ref_img_foregrounds = []
                        ref_concat_image_noises = []
                        ref_add_image_noises = []
                        ref_img_background_masks = []
                        ref_img_clips = []
                        ref_img_converts = []
                        image_np = rearrange(pixel_values_ref_img, "b c h w -> b h w c")
                        image_np = image_np.cpu().numpy().astype(np.uint8)
                        for i, ref_img in enumerate(image_np):
                            ref_img = Image.fromarray(ref_img)
                            
                            dwpose_model_result_dict = dwpose_model(ref_img)
                            # NOTE: foreground used for remove background
                            ref_img_foreground = dwpose_model_result_dict['foreground']
                            ref_img_foregrounds.append(ref_img_foreground)
                            ref_img_convert = dwpose_model_result_dict[ref_image_type]
                            ref_img_converts.append(ref_img_convert)

                            # NOTE: background_mask used for concat to noise
                            if concat_noise_image_type != "":
                                ref_concat_image_noise = dwpose_model_result_dict[concat_noise_image_type]
                                ref_concat_image_noises.append(ref_concat_image_noise)
                                ref_img_background_mask = dwpose_model_result_dict['background_mask']
                                ref_img_background_masks.append(ref_img_background_mask)                                

                            if add_noise_image_type != "":
                                print(f'WARNING it is use add_noise_image_type is {add_noise_image_type}')
                                ref_add_image_noise = dwpose_model_result_dict[add_noise_image_type]
                                ref_add_image_noises.append(ref_add_image_noise)

                            if clip_image_type != "":
                                ref_img_clip = dwpose_model_result_dict[clip_image_type]
                                ref_img_clips.append(ref_img_clip)


                # NOTE: remove background for reference image and origin video
                with torch.no_grad():
                    if remove_background:
                        pixel_values = torch.Tensor(np.array(pixel_values_remove_back)).to(local_rank, dtype=weight_type)
                        pixel_values = rearrange(pixel_values, 'b h w c -> b c h w')

                        pixel_values_ref_img = torch.Tensor(np.array(ref_img_foregrounds)).to(local_rank, dtype=weight_type)
                        pixel_values_ref_img = rearrange(pixel_values_ref_img, 'b h w c -> b c h w')

                # work for imageencoder   
                with torch.no_grad():
                    image_prompt_embeddings = None
                    if clip_image_type != "":
                        # faces = face_detector(pixel_values_ref_img)
                        # if 'image_ids' not in faces.keys() or faces['image_ids'].numel() == 0:
                        #     face_rect = [None] * len(pixel_values_ref_img)
                        # else:
                        #     face_rect = []
                        #     for i in range(len(pixel_values_ref_img)):
                        #         for j, ids in enumerate(faces['image_ids']):
                        #             if i == ids:
                        #                 face_rect.append(faces['rects'][j])
                        #                 break
                        #         if len(face_rect) == i:
                        #             face_rect.append(None)                                                

                        # face_image_list = []

                        # for i, face_rect_item in enumerate(face_rect):
                        #     face_image = crop_and_resize_tensor(pixel_values_ref_img[i], target_size=(112, 112), crop_rect=face_rect_item, is_arcface=True)                        
                        #     face_image_list.append(face_image.unsqueeze(0))
                        
                        # face_image_list = torch.cat(face_image_list)
                        # face_image_list = face_image_list / 127.5 - 1.
                        # face_image_emb = arcface_encoder(face_image_list) # (B, 512)


                        clip_images = []
                        for i, ref_image_clip in enumerate(ref_img_clips):
                            ref_image_clip = Image.fromarray(ref_image_clip)
                                                    
                            clip_image = image_processor(
                                images=ref_image_clip, return_tensors="pt").pixel_values
                            clip_images.append(clip_image)

                        clip_images = torch.cat(clip_images)

                        image_emb = image_encoder(clip_images.to(
                            local_rank, dtype=weight_type), output_hidden_states=True).last_hidden_state
                        image_emb = image_encoder.vision_model.post_layernorm(image_emb)
                        image_emb = image_encoder.visual_projection(image_emb)
                        
                        # print('image_emb shape is', image_emb.shape)
                        # image_emb = image_encoder.vision_model.post_layernorm(image_emb)

                        # face_image_emb = face_image_emb.unsqueeze(1).repeat(1, image_emb.shape[1], 1)
                        # image_emb = torch.cat([image_emb, face_image_emb], dim=2)

                        image_prompt_embeddings = image_emb

                # NOTE: convert pixel_values(origin video) to latent by vae
                pixel_values = pixel_values / 127.5 - 1
                # print('train pixel_values unique is', pixel_values.unique())
                with torch.no_grad():
                    if not image_finetune:
                        
                        latents = model.module.vae.encode(pixel_values).latent_dist
                        latents = latents.sample()
                        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                    else:
                        latents = model.module.vae.encode(pixel_values).latent_dist
                        latents = latents.sample()

                    latents = latents * 0.18215

                # NOTE: concat_noise_image_type: turn background to latent by concat
                with torch.no_grad():
                    if concat_noise_image_type != "":
                        # Image.fromarray(ref_concat_image_noises[0].astype('uint8')).save('ref_concat_image_noise.png')
                        one_img_have_more = False
                        ref_concat_image_noises = torch.Tensor(np.array(ref_concat_image_noises)).to(local_rank, dtype=weight_type)
                        if len(ref_concat_image_noises.shape) == 5:
                            ref_concat_image_noises = rearrange(ref_concat_image_noises, 'b f h w c -> (b f) h w c')
                            one_img_have_more = True
                        ref_concat_image_noises = rearrange(ref_concat_image_noises, 'b h w c -> b c h w')
                        ref_concat_image_noises = ref_concat_image_noises / 127.5 - 1
                        # print('ref_img_backgrounds unique is', ref_img_backgrounds.unique())
                        ref_concat_image_noises_latents = model.module.vae.encode(ref_concat_image_noises).latent_dist
                        ref_concat_image_noises_latents = ref_concat_image_noises_latents.sample().unsqueeze(2)
                        ref_concat_image_noises_latents = ref_concat_image_noises_latents * 0.18215
                        # b c 1 h w b c f h w

                        if one_img_have_more == True:
                            B, C, _, H, W = ref_concat_image_noises_latents.shape
                            ref_concat_image_noises_latents = ref_concat_image_noises_latents.reshape(B//2, C*2, _, H, W)

                        ref_img_back_mask_latents = torch.tensor(np.array(ref_img_background_masks).transpose(0, 3, 1, 2)).to(local_rank, dtype=weight_type)
                        H, W = ref_concat_image_noises_latents.shape[3:]
                        ref_img_back_mask_latents = F.interpolate(ref_img_back_mask_latents, size=(H, W), mode='nearest').unsqueeze(2)
                        # print('train ref_img_back_mask_latents unique is', ref_img_back_mask_latents.unique())
                        # ref_img_backgrounds_latents = ref_img_backgrounds_latents.repeat(1, 1, latents.shape[2], 1, 1)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each video
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if concat_noise_image_type != "":
                    ref_concat_image_noises_latents = ref_concat_image_noises_latents.repeat(1, 1, noisy_latents.shape[2], 1, 1)
                    ref_img_back_mask_latents = ref_img_back_mask_latents.repeat(1, 1, noisy_latents.shape[2], 1, 1)
                    noisy_latents = torch.cat([noisy_latents, ref_concat_image_noises_latents, ref_img_back_mask_latents], dim=1)
                
                if ref_image_type != "origin":
                    ref_img_converts = torch.Tensor(np.array(ref_img_converts)).to(local_rank, dtype=weight_type)
                    ref_img_converts = rearrange(ref_img_converts, 'b h w c -> b c h w')
                    pixel_values_ref_img = ref_img_converts

                # show_ref_img = pixel_values_ref_img.cpu().numpy().astype('uint8')[0]
                # Image.fromarray(show_ref_img.transpose(1, 2, 0)).save('show_ref_img.png')
                pixel_values_ref_img = pixel_values_ref_img / 127.5 - 1.
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    raise NotImplementedError
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                # import pdb;pdb.set_trace()
                # for x in [noisy_latents, encoder_hidden_states, np.array(ref_pil_images[0]), control_conditions]:
                #     print(x.shape)
                """
                noisy_latents: torch.Size([1, 4, 8, 64, 64])
                encoder_hidden_states: torch.Size([1, 257, 1280])
                np.array(ref_pil_images[0]): (512, 512, 3)
                control_conditions: (8, 512, 512, 3)
                """

                """
                TODO：pose改成b f h w c格式 待会适配下， ref_pil_images改成了b h w c格式
                """
                
                with accelerator.autocast():
                    model_pred = model(init_latents=noisy_latents,
                                    image_prompts=image_prompt_embeddings,
                                    timestep=timesteps,
                                    guidance_scale=1.0,
                                    source_image=pixel_values_ref_img, 
                                    motion_sequence=pixel_values_pose,
                                    random_seed=seed,
                                    froce_text_embedding_zero=froce_text_embedding_zero,
                                    )
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                    train_loss += avg_loss.item() / gradient_accumulation_steps

                    # use accelerator
                accelerator.backward(loss)
                model.module.clear_reference_control()

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                    seed = global_step
                    set_seed(seed)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                    
                progress_bar.update(1)
                global_step += 1

                ### <<<< Training <<<< ###
                is_main_process = accelerator.is_main_process

                # Save checkpoint
                if is_main_process and (global_step % checkpointing_steps == 0 or global_step in validation_steps_tuple or global_step % validation_steps == 0):
                    if global_step >= checkpointing_steps and global_step % checkpointing_steps == 0:
                        save_path = os.path.join(output_dir, f"checkpoints")
                        state_dict = {
                            "epoch": epoch,
                            "global_step": global_step,
                            "state_dict": model.state_dict(),
                        }
                        if step == len(train_dataloader) - 1:
                            torch.save(state_dict, os.path.join(save_path, f"checkpoint-epoch-{epoch + 1}.ckpt"))
                        else:
                            torch.save(state_dict, os.path.join(save_path, f"checkpoint-steps{global_step}.ckpt"))
                        logging.info(f"Saved state to {save_path} (global_step: {global_step})")

                    eval_model(validation_data, 
                                model, 
                                local_rank, 
                                weight_type, 
                                context, 
                                output_dir, 
                                global_step,
                                accelerator,
                                valid_seed,
                                dwpose_model,
                                arcface_encoder,
                                image_processor,
                                image_encoder,
                                face_detector,
                                control_aux_type,
                                clip_image_type,
                                ref_image_type,
                                concat_noise_image_type,
                                do_classifier_free_guidance,
                                add_noise_image_type,
                                )

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    dist.destroy_process_group()

def crop_and_resize(frame, target_size, crop_rect=None, is_arcface=False):
    height, width = frame.size
    if is_arcface:
        target_size = (112, 112)
    if crop_rect is not None:
        left, top, right, bottom = crop_rect
        face_w = right-left
        face_h = bottom-top
        padding = max(face_w, face_h) // 2
        if face_w < face_h:
            left = left - (face_h-face_w)//2
            right = right + (face_h-face_w)//2
        else:
            top = top - (face_h-face_w)//2
            bottom = bottom + (face_h-face_w)//2
        left, top, right, bottom = left-padding, top-padding, right+padding, bottom+padding 
    else:
        short_edge = min(height, width)
        width, height = frame.size
        top = (height - short_edge) // 2
        left = (width - short_edge) // 2
        right = (width + short_edge) // 2
        bottom = (height + short_edge) // 2
    frame_cropped = frame.crop((left, top, right, bottom))
    frame_resized = frame_cropped.resize(target_size, Image.ANTIALIAS)
    return frame_resized

import torch.nn.functional as F

def crop_and_resize_tensor(frame, target_size, crop_rect=None, is_arcface=False):
    # 假设 frame 是 (C, H, W) 的格式
    _, height, width = frame.shape
    
    if is_arcface:
        target_size = (112, 112)

    if crop_rect is not None:
        left, top, right, bottom = crop_rect
        face_w = right - left
        face_h = bottom - top
        padding = max(face_w, face_h) // 2
        
        if face_w < face_h:
            left = left - (face_h - face_w) // 2
            right = right + (face_h - face_w) // 2
        else:
            top = top - (face_w - face_h) // 2
            bottom = bottom + (face_w - face_h) // 2
        
        left, top, right, bottom = left - padding, top - padding, right + padding, bottom + padding
        left, top, right, bottom = max(left, 0), max(top, 0), min(right, width), min(bottom, height)
    else:
        short_edge = min(height, width)
        left = (width - short_edge) // 2
        top = (height - short_edge) // 2
        right = left + short_edge
        bottom = top + short_edge
    
    frame_cropped = frame[:, int(top):int(bottom), int(left):int(right)]
    
    # 使用PyTorch的interpolate进行resize
    # target_size应该是 (target_height, target_width)
    target_height, target_width = target_size
    frame_resized = F.interpolate(frame_cropped.unsqueeze(0), size=(target_height, target_width), mode='bilinear', align_corners=False).squeeze(0)
    
    return frame_resized



from animatediff.utils.videoreader import VideoReader


def eval_model(validation_data,
                 model, 
                 local_rank, 
                 weight_type, 
                 context,
                 output_dir,
                 global_step,
                 accelerator,
                 valid_seed,
                 dwpose_model,
                 arcface_encoder=None,
                 image_processor=None,
                 image_encoder=None,
                 face_detector=None,
                 control_aux_type='',
                 clip_image_type="",
                 ref_image_type="",
                 concat_noise_image_type="",
                 do_classifier_free_guidance=True,
                 add_noise_image_type="",
                 ):
    sample_size = validation_data['sample_size']
    guidance_scale = validation_data['guidance_scale']

    # input test videos (either source video/ conditions)

    test_videos = validation_data['video_path']
    source_images = validation_data['source_image']

    # read size, step from yaml file
    sizes = [sample_size] * len(test_videos)
    steps = [validation_data['S']] * len(test_videos)

    for idx, (source_image, test_video, size, step) in tqdm(
        enumerate(zip(source_images, test_videos, sizes, steps)),
        total=len(test_videos)
    ):

        if test_video.endswith('.mp4') or test_video.endswith('.gif'):
            print('test_video', test_video)
            control = VideoReader(test_video).read()
            video_length = control.shape[0]
            print('control', control.shape)

            if control_aux_type == "densepose_dwpose_concat":
                dwpose_conditions = np.load(test_video[:-4]+"_concat.npy")
            else:
                dwpose_conditions = control

        pixel_values_pose = torch.Tensor(np.array(dwpose_conditions))
        pixel_values_pose = rearrange(
            pixel_values_pose, "(b f) h w c -> b f h w c", b=1)
        pixel_values_pose = pixel_values_pose.to(local_rank, dtype=weight_type)
        pixel_values_pose = pixel_values_pose / 255.

        if source_image.endswith(".mp4") or source_image.endswith(".gif"):
            origin_video = VideoReader(source_image).read()
            source_image = Image.fromarray(origin_video[0])
        else:
            source_image = Image.open(source_image)

        # img_for_face_det = torch.tensor(np.array(source_image)).to(local_rank, torch.uint8).unsqueeze(0).permute(0, 3, 1, 2)
        # if image_encoder is not None:
            # with torch.inference_mode():
            #     # img_for_face_det is B C H W
            #     faces = face_detector(img_for_face_det)
            #     if 'image_ids' not in faces.keys() or faces['image_ids'].numel() == 0:
            #         face_rect = None
            #     else:
            #         face_rect = faces['rects'][0].cpu().numpy()

            # face_image_pil = crop_and_resize(source_image, size, crop_rect=face_rect, is_arcface=True)
            # face_image = np.array(face_image_pil)
            # face_image = ((torch.Tensor(face_image).unsqueeze(
            #             0).to(local_rank, dtype=weight_type) / 255.0) - 0.5) * 2

        source_image_pil = crop_and_resize(source_image, size, crop_rect=None)
        if ref_image_type != "origin" or concat_noise_image_type != "" or add_noise_image_type != "" or clip_image_type != "":
            with torch.inference_mode():
                dwpose_model_result_dict = dwpose_model(source_image_pil)
                # Image.fromarray(ref_image_control).save('ref_image_control.png')
                ref_img_foreground = dwpose_model_result_dict['foreground']
                ref_img_convert = dwpose_model_result_dict[ref_image_type]
                if concat_noise_image_type != "":
                    ref_concat_image_noise = dwpose_model_result_dict[concat_noise_image_type]
                    ref_img_background_mask = dwpose_model_result_dict['background_mask']
                if add_noise_image_type != "":
                    ref_add_image_noise = dwpose_model_result_dict[add_noise_image_type]
                if clip_image_type != "":
                    ref_img_clip = dwpose_model_result_dict[clip_image_type]  
                    ref_img_clip = Image.fromarray(ref_img_clip)

        # Image.fromarray(ref_image_background.astype('uint8')).save('backtest.png')

        source_image = np.array(source_image_pil)
        if ref_image_type != "origin":
            source_image = ref_img_convert
        source_image = ((torch.Tensor(source_image).unsqueeze(
            0).to(local_rank, dtype=weight_type) / 255.0) - 0.5) * 2

        B, H, W, C = source_image.shape
        
        # concat noise with background latents
        ref_concat_image_noises_latents = None
        if concat_noise_image_type != "":
            ref_concat_image_noises = torch.Tensor(np.array(ref_concat_image_noise)).unsqueeze(0).to(local_rank, dtype=weight_type)
            one_img_have_more = False
            if len(ref_concat_image_noises.shape) == 5:
                ref_concat_image_noises = rearrange(ref_concat_image_noises, 'b f h w c -> (b f) h w c')
                one_img_have_more = True
            ref_concat_image_noises = rearrange(ref_concat_image_noises, 'b h w c -> b c h w')
            ref_concat_image_noises = ref_concat_image_noises / 127.5 - 1
            # print('ref_img_backgrounds unique is', ref_img_backgrounds.unique())
            ref_concat_image_noises_latents = model.module.vae.encode(ref_concat_image_noises).latent_dist
            ref_concat_image_noises_latents = ref_concat_image_noises_latents.sample().unsqueeze(2)
            ref_concat_image_noises_latents = ref_concat_image_noises_latents * 0.18215

            if one_img_have_more == True:
                B, C, _, H, W = ref_concat_image_noises_latents.shape
                ref_concat_image_noises_latents = ref_concat_image_noises_latents.reshape(B//2, C*2, _, H, W)

            
            ref_img_back_mask_latents = torch.tensor(np.array(ref_img_background_mask)[None, ...].transpose(0, 3, 1, 2)).to(local_rank, dtype=weight_type)
            H, W = ref_concat_image_noises_latents.shape[3:]
            ref_img_back_mask_latents = F.interpolate(ref_img_back_mask_latents, size=(H, W), mode='nearest').unsqueeze(2)
            # print('infer ref_img_back_mask_latents unique is', ref_image_back_mask_latents.unique())
            ref_concat_image_noises_latents = torch.cat([
                ref_concat_image_noises_latents, ref_img_back_mask_latents
            ], dim=1).repeat(1, 1, video_length, 1, 1)

            if guidance_scale > 1.0 and do_classifier_free_guidance:
                ref_concat_image_noises_latents = torch.cat([ref_concat_image_noises_latents,
                 ref_concat_image_noises_latents])

            # ref_img_backgrounds_latents = ref_img_backgrounds_latents.repeat(1, 1, latents.shape[2], 1, 1)


        ######################### image encoder#########################
        image_prompt_embeddings = None
        if clip_image_type != "":
            with torch.inference_mode():
                clip_image = image_processor(
                    images=ref_img_clip, return_tensors="pt").pixel_values
                image_emb = image_encoder(clip_image.to(
                    local_rank, dtype=weight_type), output_hidden_states=True).last_hidden_state
                image_emb = image_encoder.vision_model.post_layernorm(image_emb)
                image_emb = image_encoder.visual_projection(image_emb)# image_emb = image_encoder.vision_model.post_layernorm(image_emb)

                # face_image = rearrange(face_image, 'b h w c -> b c h w')
                # face_image_emb = arcface_encoder(face_image) # (B, 512)
                # face_image_emb = face_image_emb.unsqueeze(1).repeat(1, image_emb.shape[1], 1)

                # image_emb = torch.cat([image_emb, face_image_emb], dim=2)
                # negative image embeddings
                # image_np_neg = np.zeros_like(source_image_pil)
                # ref_pil_image_neg = Image.fromarray(
                #     image_np_neg.astype(np.uint8))
                # ref_pil_image_pad = pad_image(ref_pil_image_neg)
                # clip_image_neg = image_processor(
                #     images=ref_pil_image_pad, return_tensors="pt").pixel_values
                # image_emb_neg = image_encoder(clip_image_neg.to(
                #     device, dtype=weight_type), output_hidden_states=True).hidden_states[-2]

                # image_prompt_embeddings = torch.cat([image_emb_neg, image_emb])
                image_prompt_embeddings = image_emb
                if guidance_scale > 1.0 and do_classifier_free_guidance:
                    # guidance free
                    image_prompt_embeddings = torch.cat([image_emb, image_emb])


        with torch.no_grad():
            with accelerator.autocast():
                source_image = rearrange(source_image, 'b h w c -> b c h w')
                samples = model.module.infer(
                    source_image=source_image,
                    image_prompts=image_prompt_embeddings,
                    motion_sequence=pixel_values_pose,
                    step=validation_data['num_inference_steps'],
                    guidance_scale=guidance_scale,
                    context=context,
                    size=sample_size,
                    random_seed=valid_seed,
                    froce_text_embedding_zero=False,
                    ref_concat_image_noises_latents=ref_concat_image_noises_latents,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    add_noise_image_type=add_noise_image_type,
                )
            if control_aux_type == "densepose_dwpose_concat":
                control = torch.tensor(control).unsqueeze(0)
                control = rearrange(control, 'b t h w c -> b c t h w') / 255.
                samples[1] = control
            
            # shape need to be 1 c t h w
            source_image = np.array(source_image_pil) # h w c
            source_image = torch.Tensor(source_image).unsqueeze(
                        0) / 255.
            source_image = source_image.repeat(video_length, 1, 1, 1)
            samples[0] = rearrange(source_image, "t h w c -> 1 c t h w") 

            
            samples = torch.cat(samples)
            # print('eval save samples shape is', samples.shape)
            
        os.makedirs(f"{output_dir}/samples/sample_{global_step}", exist_ok=True)
        video_name = os.path.basename(test_video)[:-4]
        source_name = os.path.basename(validation_data['source_image'][idx]).split(".")[0]

        save_path = f"{output_dir}/samples/sample_{global_step}/{source_name}_{video_name}.gif"
        save_videos_grid(samples, save_path, save_every_image=True)
        accelerator.print(f"Saved samples to {save_path}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)
