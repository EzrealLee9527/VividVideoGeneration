import einops
import torch
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
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


class StableVideoDiffusion(L.LightningModule):
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
        self.unet = unet
        self.scheduler = scheduler

        self.num_frames: int = self.unet.config.num_frames
        self.num_train_timesteps: int = self.scheduler.config.num_train_timesteps

        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.unet.requires_grad_(True)
        self.scheduler.set_timesteps(self.num_train_timesteps)
        self.config = config

    def forward(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.training_step(batch, batch_idx)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        clip_input = batch["clip_input"]
        vae_image_input = batch["vae_image_input"]
        vae_video_input = batch["vae_video_input"]
        added_time_ids = batch["added_time_ids"]

        image_embeddings = self._clip_encode_image(clip_input)
        image_latents = self._vae_encode_image(vae_image_input)
        video_latents = self._vae_encode_video(vae_video_input)

        batch_size = video_latents.size(0)
        timesteps = torch.randint(0, self.num_train_timesteps, size=(batch_size,))
        timesteps = self.scheduler.timesteps[timesteps]
        noise = torch.randn_like(video_latents)
        noisy_video_latents = self.scheduler.add_noise(video_latents, noise, timesteps)
        unet_input = torch.cat([noisy_video_latents, image_latents], dim=2)

        noise_pred = self.unet(
            unet_input,
            timesteps,
            encoder_hidden_states=image_embeddings,
            added_time_ids=added_time_ids,
            return_dict=False,
        )[0]

        # FIXME: wrong, need fix loss func
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def configure_optimizers(self) -> Tuple[Optimizer, LRScheduler]:
        # FIXME: optmizer and scheduler need to be finished
        optimizer = optim.AdamW(
            self.unet.parameters(),
            lr=self.config.train.optimizer.lr,
            weight_decay=self.config.train.optimizer.weight_decay,
        )
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.config.train.num_iteration
        )
        return optimizer, lr_scheduler

    def _clip_encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image_embeddings = self.image_encoder(image).image_embeds
        # TODO: check image_embeddings shape
        image_embeddings = image_embeddings.unsqueeze(1)
        return image_embeddings

    def _vae_encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image_latents = self.vae.encode(image).latent_dist.mode()
        # TODO: check image_latents shape
        image_latents = einops.repeat(
            image_latents, "b c h w -> b f c h w", f=self.num_frames
        )
        return image_latents

    def _vae_encode_video(self, video: torch.Tensor) -> torch.Tensor:
        video_frames = einops.rearrange(video, "b f c h w -> (b f) c h w")
        video_latents = self.vae.encode(video_frames).latent_dist.sample()
        video_latents = einops.rearrange(
            video_latents, "(b f) c h w -> b f c h w ", f=self.num_frames
        )
        return video_latents


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
