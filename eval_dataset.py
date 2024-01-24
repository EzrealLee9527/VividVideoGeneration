import argparse
import datetime
import inspect
import os
import random
import numpy as np

from PIL import Image
from omegaconf import OmegaConf
from collections import OrderedDict

import torch
import torch.distributed as dist

from diffusers import AutoencoderKL, DDIMScheduler, UniPCMultistepScheduler

from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.magic_animate.unet_controlnet import UNet3DConditionModel
from animatediff.magic_animate.controlnet import ControlNetModel
from animatediff.magic_animate.appearance_encoder import AppearanceEncoderModel
from animatediff.magic_animate.mutual_self_attention import ReferenceAttentionControl
from animatediff.magic_animate.pipeline import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.dist_tools import distributed_init
from accelerate.utils import set_seed

from animatediff.utils.videoreader import VideoReader

from einops import rearrange
from glob import glob
from pathlib import Path
from megfile import smart_open
import io
from animate import MagicAnimate
from animatediff.utils.util import save_videos_grid, pad_image
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
import facer
face_detector = facer.face_detector('retinaface/mobilenet', device="cpu")

def crop_and_resize(frame, target_size, crop_rect=None):
    height, width = frame.size
    
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


def main(args):

    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)

    config = OmegaConf.load(args.config)

    if 'dwpose_only_face' in config.keys() and config['dwpose_only_face'] == True:
        print('dwpose_only_face!!!!!!!!!!!!!!!!!!!!!!!')
        from controlnet_aux_lib import DWposeDetector
    else:
        from controlnet_aux import DWposeDetector
    det_config = '/data/models/controlnet_aux/src/controlnet_aux/dwpose/yolox_config/yolox_l_8xb8-300e_coco.py'
    det_ckpt = '/data/models/controlnet_aux/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
    pose_config = '/data/models/controlnet_aux/src/controlnet_aux/dwpose/dwpose_config/dwpose-l_384x288.py'
    pose_ckpt = '/data/models/controlnet_aux/dw-ll_ucoco_384.pth'

    dwpose_model = DWposeDetector(
        det_config=det_config,
        det_ckpt=det_ckpt,
        pose_config=pose_config,
        pose_ckpt=pose_ckpt,
    )


    # Initialize distributed training
    device = torch.device(f"cuda:{args.rank}")
    weight_type = torch.float16
    dist_kwargs = {"rank": args.rank,
                   "world_size": args.world_size, "dist": args.dist}

    if config.savename is None:
        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        savedir = f"samples/{Path(args.config).stem}-{time_str}"
    else:
        savedir = f"samples/{config.savename}"

    if args.dist:
        dist.broadcast_object_list([savedir], 0)
        dist.barrier()

    if args.rank == 0:
        os.makedirs(savedir, exist_ok=True)

    # inference_config = OmegaConf.load(config.inference_config)

    from animate import MagicAnimate
    pipeline = MagicAnimate(config=config,
                            train_batch_size=1,
                            device=device,
                            unet_additional_kwargs=OmegaConf.to_container(config.unet_additional_kwargs))
    pipeline.to(device, dtype=weight_type)

    # -------- IP adapter encoder--------#
    if config.image_encoder_path != "":
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            config.image_encoder_path).to(device)
        image_encoder.requires_grad_(False)
        image_processor = CLIPImageProcessor()
        image_encoder.to(weight_type)

    ### <<< create validation pipeline <<< ###
    import webdataset as wds
    from animatediff.data.dataset_wds import S3VideosIterableDataset
    print('eval data:', config.eval_data)
    dataset = S3VideosIterableDataset(**config.eval_data)
    dataloader = wds.WebLoader(
                dataset, 
                batch_size=1,
                shuffle=False,
                num_workers=0, 
                worker_init_fn=None,
            ).with_length(len(dataset))

    num_actual_inference_steps = config.get(
        "num_actual_inference_steps", config.steps)

    config.random_seed = []
    prompt = n_prompt = ""
    random_seed = config.seed

    for idx, batch in enumerate(dataloader):
        if idx > int(config.max_count):
            break
        samples_per_video = []
        samples_per_clip = []
        # manually set random seed for reproduction
        if random_seed != -1:
            torch.manual_seed(random_seed)
            set_seed(random_seed)
        else:
            torch.seed()
        config.random_seed.append(torch.initial_seed())

        # >>>>>>>>>>>> get control conditions >>>>>>>>>>>> #
        # TODO： dwpose去掉
        # pixel_values_pose = batch["pixel_values_pose"].to(local_rank, dtype=weight_type)
        # with torch.no_grad():
        #     pixel_values_pose = rearrange(pixel_values_pose, "b f c h w -> b f h w c")
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

            pixel_values_pose = torch.Tensor(np.array(dwpose_conditions)).to(device, dtype=weight_type)
            pixel_values_pose = rearrange(pixel_values_pose, "(b f) h w c -> b f h w c", b=1)
            # pixel_values_pose = pixel_values_pose.to(local_rank, dtype=weight_type)
            pixel_values_pose = ((pixel_values_pose / 255.0) - 0.5) * 2
        # print('pixel_values_pose', pixel_values_pose.shape, pixel_values_pose.max(), pixel_values_pose.min())

        # >>>>>>>>>>>> get reference image conditions >>>>>>>>>>>> #
        # b c h w
        pixel_values_ref_img0 = batch["pixel_values_ref_img"].to(device, dtype=weight_type)
        with torch.no_grad():
            pixel_values_ref_img0 = rearrange(pixel_values_ref_img0, "b c h w -> b h w c")
        pixel_values_ref_img = pixel_values[:1,...].to(device, dtype=weight_type)
        print('pixel_values_ref_img0', pixel_values_ref_img0.shape, pixel_values_ref_img0.max(), pixel_values_ref_img0.min())
        print('pixel_values_ref_img', pixel_values_ref_img.shape, pixel_values_ref_img.max(), pixel_values_ref_img.min())
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
                    image_emb = image_encoder(clip_image.to(device, dtype=weight_type),
                                            output_hidden_states=True).hidden_states[-2]
                    
                    # negative image embeddings
                    image_np_neg = np.zeros_like(image_np)
                    ref_pil_image_neg = Image.fromarray(image_np_neg.astype(np.uint8))
                    ref_pil_image_pad = pad_image(ref_pil_image_neg)
                    clip_image_neg = image_processor(images=ref_pil_image_pad, return_tensors="pt").pixel_values
                    image_emb_neg = image_encoder(clip_image_neg.to(device, dtype=weight_type),
                                                    output_hidden_states=True).hidden_states[-2]

                    image_emb = torch.cat([image_emb_neg, image_emb])

                    encoder_hidden_states_val.append(image_emb)
                

            if config.image_encoder_path != "":
                encoder_hidden_states_val = torch.cat(encoder_hidden_states_val)
            else:
                encoder_hidden_states_val = None


        # offset = pixel_values_pose.shape[1] % config.context['context_frames']
        # if offset > 0:
        #     pixel_values_pose = pixel_values_pose[:, :-offset, ...]



        # print(f"sampling {prompt} ...")
        # if control.shape[0] % config.L > 0:
        #     control = np.pad(control, ((0, config.L-control.shape[0] % config.L), (0, 0), (0, 0), (0, 0)), mode='edge')
        generator = torch.Generator(device=torch.device("cuda:0"))
        generator.manual_seed(torch.initial_seed())

        print('source_image', pixel_values_ref_img.max(), pixel_values_ref_img.min())
        # pixel_values_pose = (pixel_values_pose + 1.0)/2.0
        print('pixel_values_pose', pixel_values_pose.max(),pixel_values_pose.min())
        samples_per_video = pipeline.infer(
            source_image=pixel_values_ref_img,
            image_prompts=encoder_hidden_states_val,
            motion_sequence=pixel_values_pose,
            step=config.steps,
            guidance_scale=config.guidance_scale,
            random_seed=random_seed,
            context=config.context,
            size=config.size,
            froce_text_embedding_zero=config.froce_text_embedding_zero
        )

        if args.rank == 0:
            save_name = str(idx).zfill(5)
            save_videos_grid(samples_per_video, f"{savedir}/videos/{save_name}.mp4")
            save_videos_grid(samples_per_video[2:3], f"{savedir}/videos/{save_name}_gen.mp4")
            # save_videos_grid(
            #     samples_per_video[-1:], f"{savedir}/videos/{source_name}_{video_name}.gif")
            # save_videos_grid(
            #     samples_per_video, f"{savedir}/videos/{source_name}_{video_name}/grid.gif")

            if config.save_individual_videos:
                save_videos_grid(
                    samples_per_video[1:2], f"{savedir}/videos/{save_name}/ctrl.gif")
                save_videos_grid(
                    samples_per_video[0:1], f"{savedir}/videos/{save_name}/orig.gif")

        if args.dist:
            dist.barrier()

    if args.rank == 0:
        OmegaConf.save(config, f"{savedir}/config.yaml")


def distributed_main(device_id, args):
    args.rank = device_id
    args.device_id = device_id
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
        torch.cuda.init()
    distributed_init(args)
    main(args)


def run(args):

    if args.dist:
        args.world_size = max(1, torch.cuda.device_count())
        assert args.world_size <= torch.cuda.device_count()

        if args.world_size > 0 and torch.cuda.device_count() > 1:
            port = random.randint(10000, 20000)
            args.init_method = f"tcp://localhost:{port}"
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args,),
                nprocs=args.world_size,
            )
    else:
        main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dist", action="store_true", required=False)
    parser.add_argument("--rank", type=int, default=0, required=False)
    parser.add_argument("--world_size", type=int, default=1, required=False)

    args = parser.parse_args()
    run(args)
