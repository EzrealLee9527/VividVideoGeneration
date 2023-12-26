import os
import einops
import megfile
import random
import webdataset as wds
import torch
from torch.nn import Module
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from copy import deepcopy
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPImageProcessor
from dwpose.pipeline import DWPose

# type hint
from typing import List, Dict, Tuple
from torch import Tensor
from omegaconf import DictConfig

URL_PATTERN = "pipe: aws --endpoint-url={} s3 cp {} -"
ENDPOINT_URL = "http://tos-s3-cn-shanghai.ivolces.com"


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


def _prepare_for_clip(clip_image_processor, image):
    image = image * 2.0 - 1.0
    image = _resize_with_antialiasing(image, (224, 224))
    image = (image + 1.0) / 2.0

    # Normalize the image with for CLIP input
    image = clip_image_processor(
        images=image,
        do_normalize=True,
        do_center_crop=False,
        do_resize=False,
        do_rescale=False,
        return_tensors="pt",
    ).pixel_values
    return image


class PrepareImage(Module):
    def __init__(
        self,
        clip_image_processor: CLIPImageProcessor,
        vae_image_processor: VaeImageProcessor,
        height: int,
        width: int,
    ) -> None:
        super().__init__()
        self.clip_image_processor = clip_image_processor
        self.vae_image_processor = vae_image_processor
        self.height = height
        self.width = width

    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        video, info = x
        clip_input = _prepare_for_clip(self.clip_image_processor, video[0])
        vae_video_input = []
        for frame in video:
            vae_video_input.append(
                self.vae_image_processor.preprocess(
                    video[0], height=self.height, width=self.width
                )
            )
        vae_image_input = deepcopy(vae_video_input[0])


class FilterData(Module):
    def __init__(self, video_idx: int) -> None:
        super().__init__()
        self.video_idx = video_idx

    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        data, json = x
        fps = round(data[2]["video_fps"])
        motion = json["tag"]["motionscore"]
        info = {"fps": fps, "motion": motion}
        return data[self.video_idx], info


class ToChannelFirst(Module):
    def __init__(self) -> None:
        super().__init__()

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
    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        video, info = x
        video_resized = []
        for frame in video:
            video_resized.append(super().forward(frame))
        video = torch.stack(video_resized)
        return video, info


class ClipCenterCrop(CenterCrop):
    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        video, info = x
        video_cropped = []
        for frame in video:
            video_cropped.append(super().forward(frame))
        video = torch.stack(video_cropped)
        return video, info


class ParseKeypoint(Module):
    def __init__(self, det_model_path, pose_model_path) -> None:
        super().__init__()
        self.dwpose = DWPose(det_model_path, pose_model_path)

    def forward(self, x: Tuple[Tensor, Dict]) -> Tuple[Tensor, Dict]:
        video, info = x
        for frame in video:
            kps = self.dwpose(frame)
        # TODO: kps need covert to image
        return video, info


def _get_transforms(config: DictConfig) -> Compose:
    transforms = Compose(
        [
            FilterData(video_idx=0),
            ToChannelFirst(),
            Resize(config.data.size),
            CenterCrop(config.data.size),
            RandomClip(config.data.num_frames, config.data.frame_stride),
            ParseKeypoint(
                config.control.det_model_path, config.control.pose_model_path
            ),
            PrepareImage(),
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
