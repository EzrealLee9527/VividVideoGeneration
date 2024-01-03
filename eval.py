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

from pathlib import Path
from controlnet_aux import DWposeDetector
from megfile import smart_open
import io
from animate import MagicAnimate
from animatediff.utils.util import save_videos_grid, pad_image
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor

def center_crop_and_resize_org(frame, target_size):
    height, width = frame.size
    short_edge = min(height, width)

    width, height = frame.size
    top = (height - short_edge) // 2
    left = (width - short_edge) // 2
    right = (width + short_edge) // 2
    bottom = (height + short_edge) // 2
    frame_cropped = frame.crop((left, top, right, bottom))
    frame_resized = frame_cropped.resize(target_size, Image.ANTIALIAS)
    return frame_resized

def center_crop_and_resize(frame, target_size):
    height, width = frame.size
    short_edge = min(height, width)

    width, height = frame.size
    top = (height - short_edge) // 2
    bottom = (height + short_edge) // 2
    left = width - short_edge
    right = width 
    frame_cropped = frame.crop((left, top, right, bottom))
    frame_resized = frame_cropped.resize(target_size, Image.ANTIALIAS)
    return frame_resized

def main(args):

    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    config  = OmegaConf.load(args.config)
      
    # Initialize distributed training
    device = torch.device(f"cuda:{args.rank}")
    weight_type = torch.float16
    dist_kwargs = {"rank":args.rank, "world_size":args.world_size, "dist":args.dist}
    
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

    inference_config = OmegaConf.load(config.inference_config)
        
    motion_module = config.motion_module
    
    ### >>> create animation pipeline >>> ###
    tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_path, subfolder="text_encoder").to(device)
    if config.pretrained_unet_path:
        unet = UNet3DConditionModel.from_pretrained_2d(config.pretrained_unet_path, unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).to(device)
    else:
        unet = UNet3DConditionModel.from_pretrained_2d(config.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).to(device)
    appearance_encoder = AppearanceEncoderModel.from_pretrained(config.pretrained_appearance_encoder_path, subfolder="appearance_encoder").to(device)
    
    reference_control_writer = ReferenceAttentionControl(appearance_encoder, do_classifier_free_guidance=True, mode='write', fusion_blocks=config.fusion_blocks)
    reference_control_reader = ReferenceAttentionControl(unet, do_classifier_free_guidance=True, mode='read', fusion_blocks=config.fusion_blocks)
    if config.pretrained_vae_path is not None:
        vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(config.pretrained_model_path, subfolder="vae").to(device)
    
    ### Load controlnet
    controlnet = ControlNetModel.from_pretrained(config.pretrained_controlnet_path).to(device)

    ###########################################
    # load stage1 and stage2 trained apperance_encoder, controlnet and motion module
    """
    appearance_encoder 662                                                                                                                                                             
    controlnet: 340                                                                                                                                                                     
    motion: 560  
    origin unet: 686
    """
    if "appearance_controlnet_motion_checkpoint_path" in config.keys() and config['appearance_controlnet_motion_checkpoint_path'] != "":
        appearance_controlnet_motion_checkpoint_path = config['appearance_controlnet_motion_checkpoint_path']
        print(f"from checkpoint: {appearance_controlnet_motion_checkpoint_path}")
        with smart_open(appearance_controlnet_motion_checkpoint_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
            appearance_controlnet_motion_checkpoint_path = torch.load(buffer, map_location="cpu")
        if "global_step" in appearance_controlnet_motion_checkpoint_path: print(f"global_step: {appearance_controlnet_motion_checkpoint_path['global_step']}")
        org_state_dict = appearance_controlnet_motion_checkpoint_path["state_dict"] if "state_dict" in appearance_controlnet_motion_checkpoint_path else appearance_controlnet_motion_checkpoint_path        
        
        appearance_encoder_state_dict = {}
        controlnet_state_dict = {}
        motion_state_dict = {}
        for name, param in org_state_dict.items():
            if "appearance_encoder." in name:
                if name.startswith('module.appearance_encoder.'):
                    name = name.split('module.appearance_encoder.')[-1]
                appearance_encoder_state_dict[name] = param
            if "controlnet." in name:
                if name.startswith('module.controlnet.'):
                    name = name.split('module.controlnet.')[-1]
                controlnet_state_dict[name] = param
            if "motion_modules." in name:
                if name.startswith('module.unet.'):
                    name = name.split('module.unet.')[-1]
                motion_state_dict[name] = param
        print('appearance_encoder_state_dict', len(list(appearance_encoder_state_dict.keys())))
        print('controlnet_state_dict', len(list(controlnet_state_dict.keys())))
        print('motion_state_dict', len(list(motion_state_dict.keys())))
        m, u = appearance_encoder.load_state_dict(appearance_encoder_state_dict, strict=False)
        print(f"appearance_encoder missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
        m, u = controlnet.load_state_dict(controlnet_state_dict, strict=False)
        print(f"controlnet missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
        m, u = unet.load_state_dict(motion_state_dict, strict=False)
        print(f"motion_modules missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    unet.enable_xformers_memory_efficient_attention()
    appearance_encoder.enable_xformers_memory_efficient_attention()
    controlnet.enable_xformers_memory_efficient_attention()

    pipeline = AnimationPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, controlnet=controlnet,
        scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        # NOTE: UniPCMultistepScheduler
    )
    # -------- IP adapter encoder--------#
    if config.image_encoder_path != "":
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(config.image_encoder_path).to(device)
        image_encoder.requires_grad_(False)
        image_processor = CLIPImageProcessor()
        image_encoder.to(weight_type)

    vae.to(weight_type)
    unet.to(weight_type)
    text_encoder.to(weight_type)
    appearance_encoder.to(weight_type)
    controlnet.to(weight_type)
    

    # 1. unet ckpt
    # 1.1 motion module
    if motion_module != "":
        motion_module_state_dict = torch.load(motion_module, map_location="cpu")
        if "global_step" in motion_module_state_dict: func_args.update({"global_step": motion_module_state_dict["global_step"]})
        motion_module_state_dict = motion_module_state_dict['state_dict'] if 'state_dict' in motion_module_state_dict else motion_module_state_dict
        try:
            # extra steps for self-trained models
            state_dict = OrderedDict()
            for key in motion_module_state_dict.keys():
                if key.startswith("module."):
                    _key = key.split("module.")[-1]
                    state_dict[_key] = motion_module_state_dict[key]
                else:
                    state_dict[key] = motion_module_state_dict[key]
            motion_module_state_dict = state_dict
            del state_dict
            missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
            assert len(unexpected) == 0
        except:
            _tmp_ = OrderedDict()
            for key in motion_module_state_dict.keys():
                if "motion_modules" in key:
                    if key.startswith("unet."):
                        _key = key.split('unet.')[-1]
                        _tmp_[_key] = motion_module_state_dict[key]
                    else:
                        _tmp_[key] = motion_module_state_dict[key]
            missing, unexpected = unet.load_state_dict(_tmp_, strict=False)
            assert len(unexpected) == 0
            del _tmp_
        del motion_module_state_dict

    pipeline.to(device)
    ### <<< create validation pipeline <<< ###
    
    random_seeds = config.get("seed", [-1])
    random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
    random_seeds = random_seeds * len(config.source_image) if len(random_seeds) == 1 else random_seeds
    
    # input test videos (either source video/ conditions)
    
    test_videos = config.video_path
    source_images = config.source_image
    num_actual_inference_steps = config.get("num_actual_inference_steps", config.steps)

    # read size, step from yaml file
    sizes = [config.size] * len(test_videos)
    steps = [config.S] * len(test_videos)

    config.random_seed = []
    prompt = n_prompt = ""
    for idx, (source_image, test_video, random_seed, size, step) in tqdm(
        enumerate(zip(source_images, test_videos, random_seeds, sizes, steps)), 
        total=len(test_videos), 
        disable=(args.rank!=0)
    ):
        samples_per_video = []
        samples_per_clip = []
        # manually set random seed for reproduction
        if random_seed != -1: 
            torch.manual_seed(random_seed)
            set_seed(random_seed)
        else:
            torch.seed()
        config.random_seed.append(torch.initial_seed())
    
        if test_video.endswith('.mp4'):
            print('test_video', test_video)
            control = VideoReader(test_video).read()
            print('control', control.shape)
            if control[0].shape[:2] != size:
                print('size', size)
                print('control', control[0].shape[:2])
                control = [np.array(center_crop_and_resize(Image.fromarray(c),size)) for c in control]
            if config.max_length is not None:
                control = control[config.offset: (config.offset+config.max_length)]
            
        

            ##################################
            # load dwpose detector, see controlnet_aux: https://github.com/patrickvonplaten/controlnet_aux
            # specify configs, ckpts and device, or it will be downloaded automatically and use cpu by default
            if config.use_dwpose:
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

                # get pose conditions
                with torch.inference_mode():
                    dwpose_conditions = []
                    for pil_image in control:
                        dwpose_image = dwpose_model(pil_image, output_type='np')
                        dwpose_conditions.append(dwpose_image)

        pixel_values_pose = torch.Tensor(np.array(dwpose_conditions))
        pixel_values_pose = rearrange(pixel_values_pose, "(b f) h w c -> b f h w c", b=1)
        pixel_values_pose = pixel_values_pose.to(device, dtype=weight_type)
        control = ((pixel_values_pose / 255.0) - 0.5) * 2

        if source_image.endswith(".mp4"):
            source_image_pil = center_crop_and_resize(Image.fromarray(VideoReader(source_image).read()[0]),size)
            source_image = np.array(source_image_pil)
        else:
            source_image_pil = center_crop_and_resize(Image.open(source_image),size)
            source_image = np.array(source_image_pil)[:,:,:3]
        
        source_image = ((torch.Tensor(source_image).unsqueeze(0).to(device, dtype=weight_type) / 255.0) - 0.5) * 2
        B, H, W, C = source_image.shape

        #########################image encoder#########################
        image_prompt_embeddings = None
        if config.image_encoder_path != "":
            with torch.inference_mode():
                # get fine-grained embeddings
                ref_pil_image_pad = pad_image(source_image_pil)
                clip_image = image_processor(images=ref_pil_image_pad, return_tensors="pt").pixel_values
                image_emb = image_encoder(clip_image.to(device, dtype=weight_type), output_hidden_states=True).hidden_states[-2]

                # negative image embeddings
                image_np_neg = np.zeros_like(source_image_pil)
                ref_pil_image_neg = Image.fromarray(image_np_neg.astype(np.uint8))
                ref_pil_image_pad = pad_image(ref_pil_image_neg)
                clip_image_neg = image_processor(images=ref_pil_image_pad, return_tensors="pt").pixel_values
                image_emb_neg = image_encoder(clip_image_neg.to(device, dtype=weight_type), output_hidden_states=True).hidden_states[-2]

                # image_prompt_embeddings = torch.cat([image_emb_neg, image_emb])
                image_prompt_embeddings = torch.cat([image_emb, image_emb])
                print('image_prompt_embeddings', image_prompt_embeddings.shape)

        
        print(f"current seed: {torch.initial_seed()}")
        init_latents = None
        
        # print(f"sampling {prompt} ...")
        original_length = control.shape[1]
        # if control.shape[0] % config.L > 0:
        #     control = np.pad(control, ((0, config.L-control.shape[0] % config.L), (0, 0), (0, 0), (0, 0)), mode='edge')
        generator = torch.Generator(device=torch.device("cuda:0"))
        generator.manual_seed(torch.initial_seed())

        # samples_per_video = pipeline.infer(
        #         source_image=source_image,
        #         image_prompts=image_prompt_embeddings,
        #         motion_sequence=pixel_values_pose,
        #         step=config.steps,
        #         guidance_scale=config.guidance_scale,
        #         size=config.size,
        #         seed=42,
        #         context=config.context,
        #     )
        
        context_frames = context["context_frames"]
        context_stride = context["context_stride"]
        context_overlap = context["context_overlap"]
        sample = pipeline(
            prompt,
            negative_prompt         = n_prompt,
            num_inference_steps     = config.steps,
            guidance_scale          = config.guidance_scale,
            width                   = W,
            height                  = H,
            video_length            = config.L,
            controlnet_condition    = control,
            init_latents            = init_latents,
            generator               = generator,
            num_actual_inference_steps = num_actual_inference_steps,
            appearance_encoder       = appearance_encoder, 
            reference_control_writer = reference_control_writer,
            reference_control_reader = reference_control_reader,
            source_image             = source_image,
            prompt_embeddings        = image_prompt_embeddings,
            context_frames           = config.L,
            **dist_kwargs,
        ).videos

        if args.rank == 0:
            print('source_image', source_image.shape)
            print('control', control.shape)
            print('samples_per_video', samples_per_video.shape)
            # source_images = np.array([source_image.cpu()] * original_length)
            # source_images = rearrange(torch.from_numpy(source_images), "t h w c -> 1 c t h w") / 255.0
            # samples_per_video.append(source_images)
            
            # control = control / 255.0
            # control = rearrange(control, "t h w c -> 1 c t h w")
            # control = torch.from_numpy(control).cpu()
            # samples_per_video.append(control[:, :, :original_length])

            # samples_per_video.append(sample[:, :, :original_length])
                
            # samples_per_video = torch.cat(samples_per_video)

            

            # source_images = rearrange(source_image[:1,...].repeat(original_length,1,1,1), "t h w c -> 1 c t h w") 
            # source_images = (source_images+1.0)/2.0
            # samples_per_video.append(source_images.cpu())

            # control = (control+1.0)/2.0
            # control = rearrange(control[0], "t h w c -> 1 c t h w")
            # samples_per_video.append(control[:, :, :original_length].cpu())

            # samples_per_video.append(sample[:, :, :original_length])

            # samples_per_video = torch.cat(samples_per_video)

            video_name = os.path.basename(test_video)[:-4]
            source_name = os.path.basename(config.source_image[idx]).split(".")[0]
            save_videos_grid(samples_per_video[-1:], f"{savedir}/videos/{source_name}_{video_name}.mp4")
            save_videos_grid(samples_per_video, f"{savedir}/videos/{source_name}_{video_name}/grid.mp4")

            if config.save_individual_videos:
                save_videos_grid(samples_per_video[1:2], f"{savedir}/videos/{source_name}_{video_name}/ctrl.mp4")
                save_videos_grid(samples_per_video[0:1], f"{savedir}/videos/{source_name}_{video_name}/orig.mp4")
                
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
