import einops
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import CLIPVisionModelWithProjection
from diffusers import StableVideoDiffusionPipeline

# type hint
from typing import Tuple
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

        self.vae.eval()
        self.vae.requires_grad_(False)
        self.image_encoder.eval()
        self.image_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.controlnet.requires_grad_(True)

        self.scheduler.set_timesteps(self.num_train_timesteps)
        self.config = config

    def forward(self, image, video, added_time_ids, controlnet_cond) -> torch.Tensor:
        return self.train_forward(image, video, added_time_ids, controlnet_cond)

    def train_forward(
        self, image, video, added_time_ids, controlnet_cond
    ) -> torch.Tensor:
        image_embeddings = self._clip_encode_image(image)

        vae_image_inputs = self._add_noise(video[:, 0, ...], added_time_ids[:, 2])
        image_latents = self._vae_encode_image(vae_image_inputs)

        video_latents = self._vae_encode_video(video)
        sigmas, timesteps = self._sample_sigmas_and_timesteps(video_latents.size(0))
        sigmas = sigmas.to(video_latents.device, video_latents.dtype)
        timesteps = timesteps.to(video_latents.device, video_latents.dtype)
        noisy_video_latents = self._add_noise(video_latents, sigmas)
        unet_inputs = self._scale_model_input(noisy_video_latents, sigmas)
        unet_inputs = torch.cat([unet_inputs, image_latents], dim=2)
        control_model_input = unet_inputs

        null_mask = (
            torch.rand(size=(video_latents.size(0),))
            > self.config.train.null_embedding_prob
        ).to(image_embeddings.device, image_embeddings.dtype)
        null_mask = einops.rearrange(null_mask, "b -> b 1 1")
        image_embeddings = image_embeddings * null_mask

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

        pred_original_samples = self._pred_original_samples(
            noisy_video_latents, noise_preds, sigmas
        )
        loss = self._calc_loss(pred_original_samples, video_latents, sigmas)
        return loss

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
        if hasattr(self.config.train.optimizer, "schedule"):
            if self.config.train.optimizer.schedule == "constant":
                lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1)
        return optimizer, lr_scheduler

    @torch.no_grad()
    def _clip_encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = einops.rearrange(image_embeddings, "b c -> b 1 c")
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
        video_latents = self.vae.encode(video_frames).latent_dist.mode()
        video_latents = einops.rearrange(
            video_latents, "(b f) c h w -> b f c h w ", f=self.num_frames
        )
        video_latents = video_latents * self.vae.config.scaling_factor
        return video_latents

    @torch.no_grad()
    def _add_noise(
        self, original_samples: torch.Tensor, strength: torch.Tensor
    ) -> torch.Tensor:
        pattern = " ".join(["1" for _ in range(original_samples.dim() - 1)])
        strength = einops.rearrange(strength, f"b -> b {pattern}")
        noise = torch.randn_like(original_samples)
        noisy_samples = original_samples + noise * strength
        return noisy_samples

    @torch.no_grad()
    def _sample_sigmas_and_timesteps(self, batch_size):
        timestep_idx = torch.randint(0, self.num_train_timesteps, size=(batch_size,))
        sigmas = self.scheduler.sigmas[timestep_idx]
        timesteps = self.scheduler.timesteps[timestep_idx]
        return sigmas, timesteps

    @torch.no_grad()
    def _scale_model_input(
        self, noisy_video_latents: torch.Tensor, sigmas: torch.Tensor
    ) -> torch.Tensor:
        sigmas = einops.rearrange(sigmas, "b -> b 1 1 1 1")
        c_in = 1 / (sigmas**2 + 1) ** 0.5
        noisy_video_latents = noisy_video_latents * c_in
        return noisy_video_latents

    def _pred_original_samples(
        self,
        noisy_video_latents: torch.Tensor,
        noise_preds: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        sigmas = einops.rearrange(sigmas, "b -> b 1 1 1 1")
        c_skip = 1 / (sigmas**2 + 1)
        c_out = -sigmas / (sigmas**2 + 1) ** 0.5
        pred_original_sample = c_skip * noisy_video_latents + c_out * noise_preds
        return pred_original_sample

    def _calc_loss(
        self,
        pred_original_samples: torch.Tensor,
        video_latents: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        sigmas = einops.rearrange(sigmas, "b -> b 1 1 1 1")
        loss_weight = (sigmas**2 + 1) / sigmas**2
        loss = F.mse_loss(
            pred_original_samples.float(), video_latents.float(), reduction="none"
        )
        loss *= loss_weight
        return loss.mean()


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
