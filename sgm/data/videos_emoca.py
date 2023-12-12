from torch.utils.data import IterableDataset
import os
import random
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
from glob import glob
import tarfile
from megfile import smart_open as open
from megfile import smart_glob
import msgpack
from einops import rearrange, repeat
import math
# from functools import lru_cache
import webdataset as wds
import imageio.v3 as iio
import json
import traceback
from typing import List
import torchvision.transforms.functional as F
import megfile


DEBUG = os.environ.get('DEBUG')
if DEBUG:
    print('DATASET DEBUG MODE')
else:
    DEBUG = 0

# 放到data_utils里
def load_msgpack_list(file_path: str):
    loaded_data = []
    with open(file_path, 'rb') as f:
        unpacker = msgpack.Unpacker(f,strict_map_key = False)
        for unpacked_item in unpacker:
            loaded_data.append(unpacked_item)
        return loaded_data


# @lru_cache(maxsize=128)
def load_tar(p):
    return tarfile.open(fileobj=open(p, 'rb'))


def load_img_from_tar(img_path):
    tar_fname,img_fname = img_path.rsplit("/",1)
    tar_obj = load_tar(tar_fname)
    img = Image.open(tar_obj.extractfile(img_fname)).convert("RGB")
    return np.array(img)

def read_remote_img(p):
    with open(p, 'rb') as rf:
        return Image.open(rf).convert("RGB")

def gen_landmark_control_input(img_tensor, landmarks):
    cols = torch.tensor([int(y) for x,y in landmarks])
    rows = torch.tensor([int(x) for x,y in landmarks])
    img_tensor = img_tensor.index_put_(indices=(cols, rows), values=torch.ones(106))
    return img_tensor.unsqueeze(-1)


class S3VideosIterableDataset(IterableDataset):
    def __init__(
        self,  
        data_dirs,
        video_length=16,
        resolution=[512, 512],
        frame_stride=1,
        is_image=False,
    ):
        self.wds_dataset      = wds.WebDataset(data_dirs)
        self.video_length     = video_length
        self.frame_stride     = frame_stride
        self.resolution       = resolution
        self.is_image         = is_image

    def get_random_clip_indices(self, n_frames:int) -> List[int]:
        all_indices = np.linspace(0, n_frames, self.frame_stride, dtype=int).tolist()
        if len(all_indices) < self.video_length:
            frame_stride = n_frames // self.video_length
            assert (frame_stride != 0)
            all_indices = list(range(0, n_frames, frame_stride))
        
        rand_idx = random.randint(0, len(all_indices) - self.video_length)
        clip_indices = all_indices[rand_idx:rand_idx+self.video_length]
        return clip_indices


    def get_clip_frames(self, video_bytes:bytes) -> torch.Tensor:
        frames = []
        with iio.imopen(video_bytes, "r", plugin="pyav") as file:
            n_frames = file.properties().shape[0]
            clip_indices = self.get_random_clip_indices(n_frames)
            frames = file.read(index=...)  # np.array, [n_frames,h,w,c],  rgb, uint8
            frames = frames[clip_indices, ...]
            frames = torch.tensor(frames).permute(0, 3, 1, 2).float()
            # for _, idx in enumerate(clip_indices):
            #     frame_np = file.read(index=idx) # np.array, hwc, rgb, uint8
            #     frames.append(frame_np)
            # frames = torch.tensor(np.array(frames)).permute(0, 3, 1, 2).float() # tchw
        return frames

    # Center Crop
    def center_crop(self, frames:torch.Tensor):
        new_height, new_width = self.resolution 
        size = (new_height, new_width)
        # # RGB mode
        # t, c, h, w  = frames.shape
        crops = F.center_crop(frames, size)
        return crops


    def get_conditions(self, frames_tensors):
        # for training
        value_dict = {}
        cond_aug = 0.02
        value_dict["motion_bucket_id"] = 127
        value_dict["fps_id"] = 6
        value_dict["cond_aug"] = cond_aug
        anchor_image = frames_tensors[:1,...]
        value_dict["cond_frames_without_noise"] = anchor_image
        value_dict["cond_frames"] = anchor_image + cond_aug * torch.randn_like(anchor_image)
        value_dict["cond_aug"] = cond_aug

        keys = [ 'motion_bucket_id', 'fps_id', 'cond_aug', 'cond_frames','cond_frames_without_noise']
        N = [1, self.video_length]
        for key in keys:
            if key == "fps_id":
                value_dict[key] = (
                    torch.tensor([value_dict["fps_id"]])
                    .repeat(int(math.prod(N)))
                )
            elif key == "motion_bucket_id":
                value_dict[key] = (
                    torch.tensor([value_dict["motion_bucket_id"]])
                    .repeat(int(math.prod(N)))
                )
            elif key == "cond_aug":
                value_dict[key] = repeat(
                    torch.tensor([value_dict["cond_aug"]]),
                    "1 -> b",
                    b=math.prod(N),
                )
            elif key == "cond_frames":
                value_dict[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=math.prod(N))
            elif key == "cond_frames_without_noise":
                value_dict[key] = repeat(
                    value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=math.prod(N)
                )
            else:
                value_dict[key] = value_dict[key]
        
            value_dict['num_video_frames'] = self.video_length
            value_dict['image_only_indicator'] = torch.zeros(1, self.video_length)
        return value_dict


    def __iter__(self):
        for data in self.wds_dataset:
            key          = data["__key__"]
            url          = data["__url__"]
            video_bytes  = data["mp4"]
            meta_dic     = json.loads(data["json"])

            try:
                frames = self.get_clip_frames(video_bytes)
            except Exception as e:
                traceback.print_exc()
                continue
            
            frames = (frames / 255.0 - 0.5) * 2

            frames = self.center_crop(frames)

            if self.is_image:
                frames = frames[0]

            frames_captions = ['A photo of a face'] * self.video_length
            assert(len(frames_captions) > 0)

            cond_dict = self.get_conditions(frames)
            
            sample_dic = dict(
                pixel_values=frames, 
                texts=frames_captions,
                )
            sample_dic.update(cond_dict)
            yield sample_dic


def interpolate(data, crop_y1, crop_y2, crop_x1, crop_x2, size):
    data = torch.tensor(np.array(data)).permute(0, 3, 1, 2).float()
    data = F.interpolate(input=data[...,crop_y1:crop_y2, crop_x1:crop_x2], size=size, mode='bilinear', align_corners=False)
    return data


def gaussian(x, y, sigma):
    exponent = -(x**2 + y**2) / (2 * sigma**2)
    return np.exp(exponent) / (2 * np.pi * sigma**2)


def generate_gaussian_response(image_shape, landmarks, sigma=3):
    height, width = image_shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    for x, y in landmarks:
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height:
            for i in range(-sigma, sigma+1):
                for j in range(-sigma, sigma+1):
                    new_x, new_y = x + i, y + j
                    if 0 <= new_x < width and 0 <= new_y < height:
                        heatmap[new_y, new_x] += gaussian(i, j, sigma)                        
    
    heatmap[np.isnan(heatmap)] = 0
    max_value = np.max(heatmap)
    if max_value != 0:
        heatmap /= max_value
    heatmap = heatmap[:,:,np.newaxis]
    return heatmap 


def get_tarfile_name_list(bucket, object_dir):
    tarfile_path_list = megfile.smart_listdir(
        megfile.smart_path_join(f"s3://{bucket}", object_dir)
    )
    tarfile_name_list = [tarfile_path for tarfile_path in tarfile_path_list if tarfile_path.endswith(".tar")]
    return tarfile_name_list


# def train_collate_fn(examples):
#     images = torch.stack([example["image"] for example in examples])
#     images = images.to(memory_format=torch.contiguous_format).float()
#     masked_images = torch.stack([example["masked_image"] for example in examples])
#     masked_images = masked_images.to(memory_format=torch.contiguous_format).float()
#     masks = torch.stack([example["mask"] for example in examples])
#     masks = masks.to(memory_format=torch.contiguous_format).float()
#     caption_tokens = torch.stack([example["caption_token"] for example in examples])
#     caption_tokens = caption_tokens.to(memory_format=torch.contiguous_format).long()
#     return {
#         "image"          : images, 
#         "masked_image"   : masked_images, 
#         "mask"           : masks, 
#         "caption_token"  : caption_tokens,
#     }




if __name__ == "__main__":
    import resource
    from tqdm import tqdm
    

    bucket           = "weisiyuan-sh"
    object_dir       = "datasets/xiaohongshu_webdataset_20231212"
    tarfile_name_list = get_tarfile_name_list(bucket, object_dir)

    WDS_URL_LIST = [
        f"pipe: aws --endpoint-url=http://tos-s3-cn-shanghai.ivolces.com s3 cp s3://{bucket}/{object_dir}/{tarfile_name} -"
        for tarfile_name in tarfile_name_list]

    dataset = S3VideosIterableDataset(
        WDS_URL_LIST,
        video_length = 14,
        resolution = [512,512],
        frame_stride = 4,
    )

    dataloader = wds.WebLoader(
        dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn = None,
    )
    pbar = tqdm()
    for data in dataloader:
        # import pdb; pdb.set_trace()
        pbar.update(1)
        print(f"RAM PEAK: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3):.4f}")
        pass
 
    print("...")