import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import CLIPVisionModelWithProjection
from diffusers import StableVideoDiffusionPipeline

# type hint
from typing import Tuple, Dict
from omegaconf import DictConfig
from diffusers import (
    AutoencoderKLTemporalDecoder,
    UNetSpatioTemporalConditionModel,
    EulerDiscreteScheduler,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from controlnet import UNetSpatioTemporalConditionModelV2, ControlNetSpatioTemporalModel


class StableVideoDiffusion(nn.Module):
    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        config: DictConfig,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.image_encoder = image_encoder
        self.unet = UNetSpatioTemporalConditionModelV2.from_config(unet.config)
        self.scheduler = scheduler

        self.num_frames: int = self.unet.config.num_frames
        self.num_train_timesteps: int = self.scheduler.config.num_train_timesteps

        self.unet.load_state_dict(unet.state_dict())
        self.controlnet = ControlNetSpatioTemporalModel.from_unet(self.unet)

        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.controlnet.requires_grad_(True)

        self.config = config

    def forward(
        self,
        clip_input,
        vae_image_input,
        vae_video_input,
        added_time_ids,
        controlnet_cond,
    ) -> torch.Tensor:
        return self.train_forward(
            clip_input,
            vae_image_input,
            vae_video_input,
            added_time_ids,
            controlnet_cond,
        )

    def train_forward(
        self,
        clip_input,
        vae_image_input,
        vae_video_input,
        added_time_ids,
        controlnet_cond,
    ) -> torch.Tensor:
        image_embeddings = self._clip_encode_image(clip_input)
        image_latents = self._vae_encode_image(vae_image_input)
        video_latents = self._vae_encode_video(vae_video_input)

        batch_size = video_latents.size(0)
        timesteps = torch.randint(0, self.num_train_timesteps, size=(batch_size,))
        timesteps = self.scheduler.timesteps[timesteps]
        noise = torch.randn_like(video_latents)
        noisy_video_latents = self.scheduler.add_noise(video_latents, noise, timesteps)

        unet_inputs = self._scale_model_input(noisy_video_latents, timesteps)
        unet_inputs = torch.cat([unet_inputs, image_latents], dim=2)
        control_model_input = unet_inputs

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            control_model_input,
            timesteps,
            encoder_hidden_states=image_embeddings,
            added_time_ids=added_time_ids,
            controlnet_cond=controlnet_cond,
            return_dict=False,
        )

        noise_preds = self.unet(
            unet_inputs,
            timesteps,
            encoder_hidden_states=image_embeddings,
            added_time_ids=added_time_ids,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]

        pred_original_samples = self.pred_original_samples(
            noise_preds, timesteps, noisy_video_latents
        )
        loss = F.mse_loss(pred_original_samples.float(), video_latents.float())
        return 

    def configure_optimizers(self) -> Tuple[Optimizer, LRScheduler]:
        optimizer = optim.AdamW(
            self.controlnet.parameters(),
            lr=self.config.train.optimizer.lr,
            betas=(
                self.config.train.optimizer.beta1,
                self.config.train.optimizer.beta2,
            ),
            weight_decay=self.config.train.optimizer.weight_decay,
        )
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.train.num_iteration,
            eta_min=self.config.train.optimizer.min_lr,
        )
        return optimizer, lr_scheduler

    def set_timesteps(self, device):
        self.scheduler.set_timesteps(self.num_train_timesteps, device=device)

    @torch.no_grad()
    def _clip_encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)
        return image_embeddings

    @torch.no_grad()
    def _vae_encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image_latents = self.vae.encode(image).latent_dist.mode()
        image_latents = einops.repeat(
            image_latents, "b c h w -> b f c h w", f=self.num_frames
        )
        return image_latents

    @torch.no_grad()
    def _vae_encode_video(self, video: torch.Tensor) -> torch.Tensor:
        video_frames = einops.rearrange(video, "b f c h w -> (b f) c h w")
        video_latents = self.vae.encode(video_frames).latent_dist.sample()
        video_latents = einops.rearrange(
            video_latents, "(b f) c h w -> b f c h w ", f=self.num_frames
        )
        video_latents = video_latents * self.vae.config.scaling_factor
        return video_latents

    @torch.no_grad()
    def _scale_model_input(self, noisy_video_latents, timesteps):
        unet_inputs = []
        for noisy_video_latent, timestep in zip(noisy_video_latents, timesteps):
            self.scheduler._step_index = None
            unet_input = self.scheduler.scale_model_input(noisy_video_latent, timestep)
            unet_inputs.append(unet_input)
        return torch.stack(unet_inputs)

    def pred_original_samples(self, noise_preds, timesteps, noisy_video_latents):
        pred_original_samples = []
        for noise_pred, timestep, noisy_video_latent in zip(
            noise_preds, timesteps, noisy_video_latents
        ):
            self.scheduler._step_index = None
            pred_original_samples.append(
                self.scheduler.step(
                    noise_pred, timestep, noisy_video_latent
                ).pred_original_sample
            )
        return torch.stack(pred_original_samples)


def get_pipeline(model_id: str) -> StableVideoDiffusionPipeline:
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, local_files_only=True
    )
    return pipeline


def get_model(
    pipeline: StableVideoDiffusionPipeline, config: DictConfig
) -> StableVideoDiffusion:
    model = StableVideoDiffusion(
        pipeline.vae, pipeline.image_encoder, pipeline.unet, pipeline.scheduler, config
    )
    return model
