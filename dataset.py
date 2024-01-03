from torch.utils.data import IterableDataset
import random
# import torch.nn.functional as F
import torch
import numpy as np
from einops import rearrange, repeat
import math
import webdataset as wds
import imageio.v3 as iio
import json
import traceback
from typing import List
import torchvision.transforms.functional as F
import megfile
import torchvision.transforms as transforms


# def gopen_megfile(url, mode="rb", bufsize=8192, **kw):
#     """Open a URL with `curl`.

#     :param url: url (usually, http:// etc.)
#     :param mode: file mode
#     :param bufsize: buffer size
#     """
#     if mode[0] == "r":
#         return megfile.smart_open(url, mode="rb", bufsize=8192, **kw)
#     elif mode[0] == "w":
#         raise NotImplementedError
#     else:
#         raise ValueError(f"{mode}: unknown mode")
# wds.gopen_schemes["s3"] = gopen_megfile 



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
        resolution=[512, 512], # h,w
        frame_stride=1,
        dataset_length=100000,
        is_image=False,
        shuffle = True,
        resampled = True,
        endpoint_url = "http://tos-s3-cn-shanghai.ivolces.com"
    ):
        self.tarfilepath_list = self.get_tarfilepath_list(data_dirs)
        self.wds_shuffle      = shuffle
        self.wds_resampled    = resampled
        self.endpoint_url     = endpoint_url
        self.wds_dataset      = self.get_webdataset()
        
        self.video_length     = video_length
        self.frame_stride     = frame_stride
        self.resolution       = resolution
        self.is_image         = is_image
        self.dataset_length   = int(dataset_length)

        # assert resolution[0] == resolution[1], "the patch width must be the same with its height."
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(resolution[1]),
            transforms.CenterCrop(resolution),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])


    def get_tarfilepath_list(self, data_dirs):
        tarfile_path_list = []
        for data_dir in data_dirs:
            if megfile.smart_isdir(data_dir):
                file_path_list = megfile.smart_listdir(data_dir)
                tarfile_path_list += [
                    megfile.smart_path_join(data_dir, file_path)
                    for file_path in file_path_list if file_path.endswith(".tar")]
            elif data_dir.endswith(".tar"):
                tarfile_path_list.append(data_dir)
            else:
                raise NotImplementedError("仅支持输入以下路径:(1)以.tar结尾的Tar包路径; (2)文件夹路径")
        assert len(tarfile_path_list)>0, "没找到任何Tar包文件"
        return tarfile_path_list


    def get_webdataset(self, ):
        url_list = []
        for tarfilepath in self.tarfilepath_list:
            if tarfilepath.startswith("s3://") or tarfilepath.startswith("tos://"):
                tarfilepath = tarfilepath.replace("tos://", "s3://")
                url_list.append(
                    f"pipe: aws --endpoint-url={self.endpoint_url} s3 cp {tarfilepath} -"
                )
                # url_list.append(tarfilepath)
            else:
                url_list.append(tarfilepath)
        
        dataset = wds.WebDataset(url_list, resampled=self.wds_resampled)
        if self.wds_shuffle:
            dataset  = dataset.shuffle(100)
        return dataset


    def __len__(self, ):
        return self.dataset_length


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
            assert n_frames >= self.video_length, f"len(VideoClip) < {self.video_length}"
            clip_indices = self.get_random_clip_indices(n_frames)
            frames = file.read(index=...)  # np.array, [n_frames,h,w,c],  rgb, uint8
            frames = frames[clip_indices, ...]
            frames = torch.tensor(frames).permute(0, 3, 1, 2).float()
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
            # key          = data["__key__"]
            # url          = data["__url__"]
            video_bytes  = data["mp4"]
            meta_dic     = json.loads(data["json"])

            try:
                frames = self.get_clip_frames(video_bytes)
            except Exception as e:
                traceback.print_exc()
                continue
            
            # # hardcore here, direct reisze to h,w=320,512
            # frames = F.resize(frames, size=(self.resolution[0],self.resolution[1]), antialias=True)
            # # frames = (frames / 255.0 - 0.5) * 2
            # frames = frames/255.  # 

            frames = frames/255. 
            frames = self.pixel_transforms(frames)
            
            if self.is_image:
                frames = frames[0]

            frames_captions = [meta_dic["tag"]["caption_coca"]] * self.video_length
            assert(len(frames_captions) > 0)
            
            sample_dic = dict(
                pixel_values=frames, 
                texts=frames_captions,
                )
            
            # cond_dict = self.get_conditions(frames)
            # sample_dic.update(cond_dict)
            yield sample_dic


def train_collate_fn(examples):
    images = torch.stack([example["pixel_values"] for example in examples])
    images = images.to(memory_format=torch.contiguous_format).float()
    masked_images = torch.stack([example["masked_image"] for example in examples])
    masked_images = masked_images.to(memory_format=torch.contiguous_format).float()
    masks = torch.stack([example["mask"] for example in examples])
    masks = masks.to(memory_format=torch.contiguous_format).float()
    caption_tokens = torch.stack([example["caption_token"] for example in examples])
    caption_tokens = caption_tokens.to(memory_format=torch.contiguous_format).long()
    caption_tokens_2 = torch.stack([example["caption_token_2"] for example in examples])
    caption_tokens_2 = caption_tokens_2.to(memory_format=torch.contiguous_format).long()
    return {
        "image"           : images, 
        "masked_image"    : masked_images, 
        "mask"            : masks, 
        "caption_token"   : caption_tokens,
        "caption_token_2" : caption_tokens_2,
    }



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


if __name__ == "__main__":    
    import resource
    from tqdm import tqdm
    from PIL import Image

    dataset = S3VideosIterableDataset(
        ["s3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/wds/CelebV_webdataset_20231211",],
        video_length = 14,
        resolution = [320,512],
        frame_stride = 4,
    )

    dataloader = wds.WebLoader(
        dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn = None,
    ).with_length(len(dataset))
    # pbar = tqdm()
    for data in tqdm(dataloader):
        
        img = data["pixel_values"][0,0,...].numpy() # chw
        img = img.transpose((1,2,0))
        img = np.clip((img*2+1)*255., 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
        img_pil.save("data_debug.png")
        import pdb; pdb.set_trace()
        # print(f"RAM PEAK: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3):.4f}")
        pass
 
    print("...")
    

    '''
    import accelerate
    from tqdm import tqdm

    
    accelerator = accelerate.Accelerator()

    dataset = S3VideosIterableDataset(
        ["s3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/wds/CelebV_webdataset_20231211",],
        video_length = 14,
        resolution = [320,512],
        frame_stride = 2,
    )

    dataloader = wds.WebLoader(
        dataset, 
        batch_size=1,
        num_workers=8,
        collate_fn = None,
    ).with_length(len(dataset))
    # pbar = tqdm()
    for data in tqdm(dataloader, disable=not accelerator.is_main_process):
        # pbar.update(1)
        # import pdb; pdb.set_trace()
        # print(f"RAM PEAK: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3):.4f}")
        pass
 
    print("...")
    '''