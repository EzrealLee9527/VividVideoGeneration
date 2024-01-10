import os
import math
import wandb
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
from animate import MagicAnimate
from animatediff.magic_animate.controlnet import ControlNetModel
import importlib
from animatediff.data.dataset import WebVid10M, PexelsDataset
import webdataset as wds
from animatediff.data.dataset_wds import S3VideosIterableDataset

def main(

        # >>>>>> new params >>>>>> #
        image_encoder_path: str,

        # <<<<<< new params <<<<<< #

        image_finetune: bool,

        name: str,
        use_wandb: bool,
        launcher: str,

        output_dir: str,
        pretrained_model_path: str,
        pretrained_appearance_encoder_path: str,
        pretrained_controlnet_path: str,
        pretrained_vae_path: str,
        motion_module: str,
        appearance_controlnet_motion_checkpoint_path: str,
        pretrained_unet_path: str,
        inference_config: str,

        data_module: str,
        data_class: str,
        train_data: Dict,
        validation_data: Dict,
        context: Dict,
        cfg_random_null_text: bool = True,
        cfg_random_null_text_ratio: float = 0.1,

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

        global_seed: int = 42,
        is_debug: bool = False,

        dwpose_only_face = False,
        froce_text_embedding_zero = False,

        ip_ckpt=None,

        
):
    weight_type = torch.float16
    # Accelerate
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
    )
    seed = global_seed + accelerator.process_index
    torch.manual_seed(seed)

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
    det_config = '/data/models/controlnet_aux/src/controlnet_aux/dwpose/yolox_config/yolox_l_8xb8-300e_coco.py'
    det_ckpt = '/data/models/controlnet_aux/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
    pose_config = '/data/models/controlnet_aux/src/controlnet_aux/dwpose/dwpose_config/dwpose-l_384x288.py'
    pose_ckpt = '/data/models/controlnet_aux/dw-ll_ucoco_384.pth'

    if dwpose_only_face == True:
        from controlnet_aux_lib import DWposeDetector
    else:
        from controlnet_aux import DWposeDetector
    dwpose_model = DWposeDetector(
        det_config=det_config,
        det_ckpt=det_ckpt,
        pose_config=pose_config,
        pose_ckpt=pose_ckpt,
        device=local_rank
    )

    # -------- magic_animate --------#
    model = MagicAnimate(config=config,
                         train_batch_size=train_batch_size,
                         device=local_rank,
                         unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs))

    # ----- load image encoder ----- #
    """
    使用IP-adapter，主要包含image_encoder，clip_image_processor和image_proj_model
    image_proj_model在Resampler里定义
    """
    if image_encoder_path != "":
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
        image_encoder.requires_grad_(False)
        image_processor = CLIPImageProcessor()
        image_encoder.to(local_rank, dtype=weight_type)

    # Set trainable parameters
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
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

    model.unet.enable_xformers_memory_efficient_attention()
    model.appearance_encoder.enable_xformers_memory_efficient_attention()
    model.controlnet.enable_xformers_memory_efficient_attention()

    
    model.to(local_rank, dtype=weight_type)
    

    # Get the training dataset
    dataset_cls = getattr(importlib.import_module(data_module, package=None), data_class)
    train_dataset = dataset_cls(**train_data, is_image=image_finetune)
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

    train_dataloader = wds.WebLoader(
                train_dataset, 
                batch_size=train_batch_size,
                shuffle=False,
                num_workers=num_workers, 
                worker_init_fn=None,
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

    if accelerator.is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    model, optimizer = accelerator.prepare(model, optimizer)

    for epoch in range(first_epoch, num_train_epochs):
        # TODO: check webdataset在多卡的随机性问题
        # train_dataloader.sampler.set_epoch(epoch)
        model.train()

        for step, batch in enumerate(train_dataloader):

            # Data batch sanity check
            if global_step % 1000 == 0:
                # pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                pixel_values = batch['pixel_values'].cpu()
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                for idx, pixel_value in enumerate(pixel_values):
                    pixel_value = pixel_value[None, ...]
                    save_videos_grid(pixel_value,
                                     f"{output_dir}/sanity_check/global_{global_step}.gif",
                                     rescale=True)

            ### >>>> Training >>>> ###

            # Convert videos to latent space            
            pixel_values = batch["pixel_values"].to(local_rank, dtype=weight_type)
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    latents = model.vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                else:
                    latents = model.vae.encode(pixel_values).latent_dist
                    latents = latents.sample()

                latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)


            """
            pixel_values: torch.Size([1, 14, 3, 512, 512])
            pixel_values_pose: torch.Size([1, 14, 3, 512, 512])
            clip_ref_image: torch.Size([1, 1, 3, 224, 224])
            pixel_values_ref_img: torch.Size([1, 3, 512, 512])
            drop_image_embeds: torch.Size([1])

            noisy_latents: torch.Size([1, 4, 8, 64, 64])
            encoder_hidden_states: torch.Size([1, 257, 1280])
            np.array(ref_pil_images[0]): (512, 512, 3)
            control_conditions: (8, 512, 512, 3)
            """

            # >>>>>>>>>>>> get control conditions >>>>>>>>>>>> #
            # TODO： dwpose去掉
            # pixel_values_pose = batch["pixel_values_pose"].to(local_rank, dtype=weight_type)
            # with torch.no_grad():
            #     pixel_values_pose = rearrange(pixel_values_pose, "b f c h w -> b f h w c")
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

                pixel_values_pose = torch.Tensor(np.array(dwpose_conditions))
                pixel_values_pose = rearrange(pixel_values_pose, "(b f) h w c -> b f h w c", b=train_batch_size)
                pixel_values_pose = pixel_values_pose.to(local_rank, dtype=weight_type)
                pixel_values_pose = ((pixel_values_pose / 255.0) - 0.5) * 2
            # print('pixel_values_pose', pixel_values_pose.shape, pixel_values_pose.max(), pixel_values_pose.min())

            # >>>>>>>>>>>> get reference image conditions >>>>>>>>>>>> #
            # b c h w
            pixel_values_ref_img = batch["pixel_values_ref_img"].to(local_rank, dtype=weight_type)
            with torch.no_grad():
                pixel_values_ref_img = rearrange(pixel_values_ref_img, "b c h w -> b h w c")

            # >>>>>>>>>>>> Get the image embedding for conditioning >>>>>>>>>>>>#
            # encoder_hidden_states = torch.zeros(10, 257, 1280).to(local_rank, dtype=weight_type)
            # encoder_hidden_states_val = torch.zeros(20, 257, 1280).to(local_rank, dtype=weight_type)
            # encoder_hidden_states = torch.zeros(train_batch_size, 16, 768).to(local_rank, dtype=weight_type)
            # encoder_hidden_states_val = torch.zeros(train_batch_size*2, 16, 768).to(local_rank, dtype=weight_type)
            with torch.inference_mode():
                ref_pil_images = []
                encoder_hidden_states = []
                encoder_hidden_states_val = []

                for batch_id in range(pixel_values_ref_img.shape[0]):
                    image_np = pixel_values_ref_img[batch_id].cpu().numpy()
                    image_np = (image_np * 0.5 + 0.5) * 255
                    ref_pil_image = Image.fromarray(image_np.astype(np.uint8))
                    ref_pil_images.append(ref_pil_image)

                    # debug
                    # if accelerator.is_main_process:
                    #     ref_pil_images[0].save("ref_img.jpg")

                    if image_encoder_path != "":
                        # get fine-grained embeddings
                        ref_pil_image_pad = pad_image(ref_pil_image)
                        clip_image = image_processor(images=ref_pil_image_pad, return_tensors="pt").pixel_values
                        image_emb = image_encoder(clip_image.to(local_rank, dtype=weight_type),
                                                output_hidden_states=True).hidden_states[-2]
                        encoder_hidden_states.append(image_emb)
                        
                        # negative image embeddings
                        image_np_neg = np.zeros_like(image_np)
                        ref_pil_image_neg = Image.fromarray(image_np_neg.astype(np.uint8))
                        ref_pil_image_pad = pad_image(ref_pil_image_neg)
                        clip_image_neg = image_processor(images=ref_pil_image_pad, return_tensors="pt").pixel_values
                        image_emb_neg = image_encoder(clip_image_neg.to(local_rank, dtype=weight_type),
                                                        output_hidden_states=True).hidden_states[-2]

                        image_emb = torch.cat([image_emb_neg, image_emb])

                        encoder_hidden_states_val.append(image_emb)
                    

                if image_encoder_path != "":
                    encoder_hidden_states = torch.cat(encoder_hidden_states)
                    encoder_hidden_states_val = torch.cat(encoder_hidden_states_val)
                    # <<<<<<<<<<< Drop the image embedding for cfg train <<<<<<<<<<<<<#
                    drop_image_embeds = batch["drop_image_embeds"].to(local_rank)
                    mask = drop_image_embeds > 0
                    mask = mask.unsqueeze(1).unsqueeze(2).expand_as(encoder_hidden_states)
                    encoder_hidden_states[mask] = 0
                else:
                    encoder_hidden_states = None
                    encoder_hidden_states_val = None

                
                # print('encoder_hidden_states', encoder_hidden_states.shape)
                # print('encoder_hidden_states_val', encoder_hidden_states_val.shape)
            # project from (batch_size, 257, 1280) to (batch_size, 16, 768)
        
            # encoder_hidden_states = model.unet.image_proj_model(encoder_hidden_states)
            # encoder_hidden_states_val = model.unet.image_proj_model(encoder_hidden_states_val)

            

            # <<<<<<<<<<< Get the image embedding for conditioning <<<<<<<<<<<<<#
            
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
            model_pred = model(init_latents=noisy_latents,
                               image_prompts=encoder_hidden_states,
                               timestep=timesteps,
                               guidance_scale=1.0,
                               source_image=pixel_values_ref_img, 
                               motion_sequence=pixel_values_pose,
                               random_seed=seed,
                               froce_text_embedding_zero=froce_text_embedding_zero
                               )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # use accelerator
            accelerator.backward(loss, retain_graph=True)
            accelerator.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            # if mixed_precision_training:
            #     scaler.scale(loss).backward(retain_graph=True)
            #     """ >>> gradient clipping >>> """
            #     scaler.unscale_(optimizer)
            #     torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            #     """ <<< gradient clipping <<< """
            #     scaler.step(optimizer)
            #     scaler.update()
            # else:
            #     loss.backward()
            #     """ >>> gradient clipping >>> """
            #     accelerator.clip_grad_norm_(trainable_params, max_grad_norm)
            #     """ <<< gradient clipping <<< """
            #     optimizer.step()

            optimizer.zero_grad()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1

            ### <<<< Training <<<< ###
            is_main_process = accelerator.is_main_process

            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)

            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0 or step == len(train_dataloader) - 1):
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

            # Periodically validation
            if is_main_process and (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
                config.random_seed = []
                sample_size = validation_data['sample_size']
                guidance_scale = validation_data['guidance_scale']
                eval_dataset = S3VideosIterableDataset(**config.validation_data['dataset'])

                for idx, batch in tqdm(enumerate(eval_dataset)):
                    if idx > int(config.max_count):
                        break
                    # >>>>>>>>>>>> get control conditions >>>>>>>>>>>> #
                    with torch.inference_mode():
                        pixel_values = batch["pixel_values"]
                        pixel_values = rearrange(pixel_values, "b f c h w -> (b f) h w c")
                        image_np = (pixel_values * 0.5 + 0.5) * 255
                        image_np = image_np.cpu().numpy().astype(np.uint8)
                        num_frames = image_np.shape[0]

                        dwpose_conditions = []
                        for frame_id in range(num_frames):
                            pil_image = Image.fromarray(image_np[frame_id])
                            dwpose_image = dwpose_model(pil_image, output_type='np')
                            dwpose_conditions.append(dwpose_image)
                        pixel_values_pose = torch.Tensor(np.array(dwpose_conditions)).to(local_rank, dtype=weight_type)
                        pixel_values_pose = rearrange(pixel_values_pose, "(b f) h w c -> b f h w c", b=1)
                        pixel_values_pose = ((pixel_values_pose / 255.0) - 0.5) * 2

                    # >>>>>>>>>>>> get reference image conditions >>>>>>>>>>>> #
                    pixel_values_ref_img = pixel_values[:1,...].to(local_rank, dtype=weight_type)
                    # >>>>>>>>>>>> Get the image embedding for conditioning >>>>>>>>>>>>#
                    with torch.inference_mode():
                        ref_pil_images = []
                        encoder_hidden_states_val = []

                        for batch_id in range(pixel_values_ref_img.shape[0]):
                            image_np = pixel_values_ref_img[batch_id].cpu().numpy()
                            image_np = (image_np * 0.5 + 0.5) * 255
                            ref_pil_image = Image.fromarray(image_np.astype(np.uint8))
                            ref_pil_images.append(ref_pil_image)
                            if config.image_encoder_path != "":
                                # get fine-grained embeddings
                                ref_pil_image_pad = pad_image(ref_pil_image)
                                clip_image = image_processor(images=ref_pil_image_pad, return_tensors="pt").pixel_values
                                image_emb = image_encoder(clip_image.to(local_rank, dtype=weight_type),
                                                        output_hidden_states=True).hidden_states[-2]
                                
                                # negative image embeddings
                                image_np_neg = np.zeros_like(image_np)
                                ref_pil_image_neg = Image.fromarray(image_np_neg.astype(np.uint8))
                                ref_pil_image_pad = pad_image(ref_pil_image_neg)
                                clip_image_neg = image_processor(images=ref_pil_image_pad, return_tensors="pt").pixel_values
                                image_emb_neg = image_encoder(clip_image_neg.to(local_rank, dtype=weight_type),
                                                                output_hidden_states=True).hidden_states[-2]

                                image_emb = torch.cat([image_emb_neg, image_emb])

                                encoder_hidden_states_val.append(image_emb)
                            

                        if config.image_encoder_path != "":
                            encoder_hidden_states_val = torch.cat(encoder_hidden_states_val)
                        else:
                            encoder_hidden_states_val = None

                    generator = torch.Generator(device=torch.device("cuda:0"))
                    generator.manual_seed(torch.initial_seed())

                    print('source_image', pixel_values_ref_img.max(), pixel_values_ref_img.min())
                    # pixel_values_pose = (pixel_values_pose + 1.0)/2.0
                    print('pixel_values_pose', pixel_values_pose.max(),pixel_values_pose.min())
                    if encoder_hidden_states_val is not None:
                        encoder_hidden_states_val = encoder_hidden_states_val[:2]
                    sample = model.infer(
                        source_image=pixel_values_ref_img[:1],
                        image_prompts=encoder_hidden_states_val,
                        motion_sequence=pixel_values_pose[:1],
                        random_seed=seed,
                        step=validation_data['num_inference_steps'],
                        guidance_scale=guidance_scale[idx],
                        context=context,
                        size=(sample_size[1], sample_size[0]),
                        froce_text_embedding_zero=config['froce_text_embedding_zero']
                    )
                    # save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{idx}.gif")
                    samples.append(sample)

                samples = torch.concat(samples)
                save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                save_videos_grid(samples, save_path)
                logging.info(f"Saved samples to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)
