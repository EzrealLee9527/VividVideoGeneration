import os
import math
import einops
import megfile
import random
from torchvision.transforms.functional import InterpolationMode
import webdataset as wds
import torch
from torch.nn import Module
from torchvision.transforms import Compose, Resize, CenterCrop, RandomCrop, Normalize
from copy import deepcopy
from dwpose import DWposeDetector

# type hint
from typing import List, Dict, Tuple
from torch import Tensor
from omegaconf import DictConfig

URL_PATTERN = "pipe:aws --endpoint-url={} s3 cp {} -"
ENDPOINT_URL = "http://oss.i.shaipower.com"
IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]


def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners
    )
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1
    )

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (
        torch.arange(window_size, device=sigma.device, dtype=sigma.dtype)
        - window_size // 2
    ).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


class FilterData(Module):
    def __init__(self, video_idx: int) -> None:
        super().__init__()
        self.video_idx = video_idx

    @torch.no_grad()
    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        data, json = x
        fps = data[2]["video_fps"]
        motion = json["tag"]["motionscore"]
        motion_bucket_id = motion * 255 if motion != -1 else -1
        noise_aug_strength = 0.0
        info = {}
        info["added_time_ids"] = torch.tensor(
            [fps, motion_bucket_id, noise_aug_strength]
        )
        return data[self.video_idx], info


class ToChannelFirst(Module):
    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        video, info = x
        video = einops.rearrange(video, "f h w c -> f c h w")
        return video, info


class RandomClip(Module):
    def __init__(self, num_frames: int, frame_stride: int) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.origin_clip_frames = num_frames * frame_stride

    @torch.no_grad()
    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        video, info = x
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
        return video, info


class ClipResize(Resize):
    def __init__(
        self,
        size,
        interpolation=InterpolationMode.BILINEAR,
        max_size=None,
        antialias=True,
    ):
        super().__init__(size, interpolation, max_size, antialias)

    @torch.no_grad()
    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        video, info = x
        video = super().forward(video)
        return video, info


class ClipCenterCrop(CenterCrop):
    @torch.no_grad()
    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        video, info = x
        video = super().forward(video)
        return video, info


class ClipRandomCrop(RandomCrop):
    @torch.no_grad()
    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        video, info = x
        video = super().forward(video)
        return video, info


class ParseCondition(Module):
    def __init__(self) -> None:
        super().__init__()
        self.dwpose = DWposeDetector()

    @torch.no_grad()
    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        video, info = x
        controlnet_cond = []
        for frame in video:
            frame = frame.numpy().transpose(1, 2, 0)
            cond = self.dwpose(frame)
            cond = torch.from_numpy(cond.transpose(2, 0, 1))
            controlnet_cond.append(cond)
        controlnet_cond = torch.stack(controlnet_cond)
        info["controlnet_cond"] = controlnet_cond
        return video, info


class ConditionJitter(Module):
    def __init__(self, size, min_ratio: float = 0.5) -> None:
        super().__init__()
        padding = round(size - math.sqrt(min_ratio * (size**2)))
        self.random_crop = RandomCrop(size, padding=padding)

    @torch.no_grad()
    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        video, info = x
        info["controlnet_cond"] = self.random_crop(info["controlnet_cond"])
        return video, info


class ClipNormalize(Module):
    def __init__(self):
        super().__init__()
        self.clip_normalize = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        self.vae_normalize = Normalize(mean=0.5, std=0.5)

    @torch.no_grad()
    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        video, info = x
        video = video / 255
        # for clip
        clip_input = video[0].unsqueeze(0)
        clip_input = clip_input * 2.0 - 1.0
        clip_input = _resize_with_antialiasing(clip_input, (224, 224))
        clip_input = (clip_input + 1.0) / 2.0
        clip_input = self.clip_normalize(clip_input)
        clip_input = clip_input[0]
        # for vae
        vae_video_input = []
        for frame in video:
            vae_video_input.append(self.vae_normalize(frame))
        vae_video_input = torch.stack(vae_video_input)
        vae_image_input = deepcopy(vae_video_input[0])

        info["controlnet_cond"] = info["controlnet_cond"] / 255
        info["clip_input"] = clip_input
        info["vae_video_input"] = vae_video_input
        info["vae_image_input"] = vae_image_input
        return info


class Identity(Module):
    @torch.no_grad()
    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        return x


def _get_transforms(config: DictConfig) -> Compose:
    use_condition_jitter = False
    if hasattr(config.data, "use_condition_jitter"):
        use_condition_jitter = config.data.use_condition_jitter
    transforms = Compose(
        [
            FilterData(video_idx=0),
            ToChannelFirst(),
            ClipResize(config.data.size),
            ClipRandomCrop(config.data.size),
            RandomClip(config.data.num_frames, config.data.frame_stride),
            ParseCondition(),
            ConditionJitter(config.data.size, config.data.condition_jitter_min_ratio)
            if use_condition_jitter
            else Identity(),
            ClipNormalize(),
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


def collation_fn(samples):
    batch = {}
    batch["clip_input"] = torch.stack([sample["clip_input"] for sample in samples])
    batch["vae_image_input"] = torch.stack(
        [sample["vae_image_input"] for sample in samples]
    )
    batch["vae_video_input"] = torch.stack(
        [sample["vae_video_input"] for sample in samples]
    )
    batch["controlnet_cond"] = torch.stack(
        [sample["controlnet_cond"] for sample in samples]
    )
    batch["added_time_ids"] = torch.stack(
        [sample["added_time_ids"] for sample in samples]
    )
    return batch


def get_dataloader(config: DictConfig) -> wds.WebLoader:
    def data_filter(data):
        if data["mp4"][0].size(0) < config.data.num_frames * config.data.frame_stride:
            return False
        return True

    urls = _get_data_urls(config.data.root_dirs)
    dataset = wds.DataPipeline(
        wds.ResampledShards(urls),
        wds.shuffle(10),
        wds.split_by_node,
        wds.cached_tarfile_to_samples(
            cache_dir=config.data.cache_dir,
            cache_size=config.data.cache_size_in_gb * 1024 * 1024 * 1024,
        ),
        wds.shuffle(10),
        wds.decode(wds.torch_video),
        wds.select(data_filter),
        wds.to_tuple("mp4", "json"),
        wds.map(_get_transforms(config)),
        wds.batched(config.train.batch_size, collation_fn=collation_fn, partial=False),
    )
    dataloader = wds.WebLoader(
        dataset, batch_size=None, num_workers=config.data.num_workers
    )
    return dataloader
