import os
import einops
import megfile
import random
import webdataset as wds
import torch
from torch.nn import Module
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

# type hint
from typing import List, Dict, Tuple
from torch import Tensor
from omegaconf import DictConfig

URL_PATTERN = "pipe: aws --endpoint-url={} s3 cp {} -"
ENDPOINT_URL = "http://tos-s3-cn-shanghai.ivolces.com"


class OnlyVideo(Module):
    def __init__(self, video_idx: int) -> None:
        super().__init__()
        self.video_idx = video_idx

    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        video, json = x
        return video[self.video_idx], json


class ToChannelFirst(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        video, json = x
        video = einops.rearrange(video, "f h w c -> f c h w")
        return video, json


class RandomClip(Module):
    def __init__(self, num_frames: int, frame_stride: int) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.origin_clip_frames = num_frames * frame_stride

    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        video, json = x
        total_frames = video.size(0)
        start_frame_idx = random.randint(0, total_frames - self.origin_clip_frames)
        frame_idices = list(
            range(
                start_frame_idx,
                start_frame_idx + self.origin_clip_frames,
                self.frame_stride,
            )
        )
        video = video[frame_idices]
        # FIXME: fps should change ?
        return video, json


class ClipResize(Resize):
    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        video, json = x
        video_resized = []
        for frame in video:
            video_resized.append(super().forward(frame))
        video = torch.stack(video_resized)
        return video, json


class ClipCenterCrop(CenterCrop):
    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        video, json = x
        video_cropped = []
        for frame in video:
            video_cropped.append(super().forward(frame))
        video = torch.stack(video_cropped)
        return video, json


class ParseKeypoint(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        video, json = x
        # TODO: unfinish
        return video, json


class ClipNormalize(Normalize):
    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        video, json = x
        video = super().forward(video)
        return video, json


def _get_transforms(config: DictConfig) -> Compose:
    transforms = Compose(
        [
            OnlyVideo(video_idx=0),
            ToChannelFirst(),
            Resize(config.data.size),
            CenterCrop(config.data.size),
            RandomClip(config.data.num_frames, config.data.frame_stride),
            ParseKeypoint(),
            ClipNormalize(config.data.norm_mean, config.data.norm_std),
        ]
    )
    return transforms


def _get_data_urls(root_dirs: List[str]) -> List[str]:
    urls = []
    for root_dir in root_dirs:
        tarfile_paths = megfile.smart_glob(os.path.join(root_dir, "*.tar"))
        for tarfile_path in tarfile_paths:
            if tarfile_path.startswith("tos://"):
                tarfile_path = tarfile_path.replace("tos://", "s3://")
            urls.append(URL_PATTERN.format(ENDPOINT_URL, tarfile_path))
    return sorted(urls)


def get_dataloader(config: DictConfig) -> wds.WebLoader:
    urls = _get_data_urls(config.data.root_dirs)
    dataset = (
        wds.WebDataset(urls, resampled=True)
        .shuffle(100)
        .decode(wds.torch_video)
        .to_tuple("mp4", "json")
        .map(_get_transforms(config))
        .batched(config.train.batch_size, partial=False)
    )
    dataloader = (
        wds.WebLoader(dataset, batch_size=None, num_workers=config.data.num_workers)
        .unbatched()
        .shuffle(100)
    )
    return dataloader
