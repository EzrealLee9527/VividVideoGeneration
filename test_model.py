from animatediff.magic_animate.appearance_encoder import AppearanceEncoderModel
from animatediff.magic_animate.controlnet import ControlNetModel
from megfile import smart_open, smart_exists, smart_sync, smart_remove, smart_glob
import torch
import io

# pretrained_appearance_encoder_path = '/models00/pretrained_models/MagicAnimate/appearance_encoder'
# appearance_controlnet_checkpoint_path = 'outputs/aa_train_stage1_celebv-2023-12-18T04-10-27/checkpoints/checkpoint.ckpt'
# appearance_encoder = AppearanceEncoderModel.from_pretrained(pretrained_appearance_encoder_path)
# pretrained_controlnet_path = "/models00/pretrained_models/MagicAnimate/densepose_controlnet"
# controlnet = ControlNetModel.from_pretrained(pretrained_controlnet_path)

# if appearance_controlnet_checkpoint_path != "":
#         print(f"from checkpoint: {appearance_controlnet_checkpoint_path}")
#         with smart_open(appearance_controlnet_checkpoint_path, 'rb') as f:
#             buffer = io.BytesIO(f.read())
#             appearance_controlnet_checkpoint_path = torch.load(buffer, map_location="cpu")
#         if "global_step" in appearance_controlnet_checkpoint_path: print(f"global_step: {appearance_controlnet_checkpoint_path['global_step']}")
#         org_state_dict = appearance_controlnet_checkpoint_path["state_dict"] if "state_dict" in appearance_controlnet_checkpoint_path else appearance_controlnet_checkpoint_path        
        
#         appearance_encoder_state_dict = {}
#         controlnet_state_dict = {}
#         for name, param in org_state_dict.items():
#             if "appearance_encoder." in name:
#                 if name.startswith('module.appearance_encoder.'):
#                     name = name.split('module.appearance_encoder.')[-1]
#                 appearance_encoder_state_dict[name] = param
#             if "controlnet." in name:
#                 if name.startswith('module.controlnet.'):
#                     name = name.split('module.controlnet.')[-1]
#                 controlnet_state_dict[name] = param
#         print('appearance_encoder_state_dict', len(list(appearance_encoder_state_dict.keys())))
#         print('controlnet_state_dict', len(list(controlnet_state_dict.keys())))
#         m, u = appearance_encoder.load_state_dict(appearance_encoder_state_dict, strict=False)
#         print(f"appearance_encoder missing keys: {len(m)}, unexpected keys: {len(u)}")
#         assert len(u) == 0
#         m, u = controlnet.load_state_dict(controlnet_state_dict, strict=False)
#         print(f"controlnet missing keys: {len(m)}, unexpected keys: {len(u)}")
#         assert len(u) == 0


"""
appearance_encoder 662                                                                                                                                                             
controlnet: 340                                                                                                                                                                     
motion: 560  
origin unet: 686
"""
appearance_controlnet_motion_checkpoint_path = '/data/work/animate_based_ap_ctrl/outputs/train_stage1_w_imageencoder_celebv_nocenterface_cfg_train_image_proj_model-2024-01-05T10-48-09/checkpoints/checkpoint-steps1500.ckpt'
print(f"from checkpoint: {appearance_controlnet_motion_checkpoint_path}")
with smart_open(appearance_controlnet_motion_checkpoint_path, 'rb') as f:
    buffer = io.BytesIO(f.read())
    appearance_controlnet_motion_checkpoint_path = torch.load(buffer, map_location="cpu")
if "global_step" in appearance_controlnet_motion_checkpoint_path: print(f"global_step: {appearance_controlnet_motion_checkpoint_path['global_step']}")
org_state_dict = appearance_controlnet_motion_checkpoint_path["state_dict"] if "state_dict" in appearance_controlnet_motion_checkpoint_path else appearance_controlnet_motion_checkpoint_path        

appearance_encoder_state_dict = {}
controlnet_state_dict = {}
motion_state_dict = {}
image_encoder_state_dict = {}
unet_state_dict = {}
vae_state_dict = {}
text_encoder_state_dict = {}
for name, param in org_state_dict.items():
    if "appearance_encoder." in name:
        if name.startswith('module.appearance_encoder.'):
            name = name.split('module.appearance_encoder.')[-1]
        appearance_encoder_state_dict[name] = param
    elif "controlnet." in name:
        if name.startswith('module.controlnet.'):
            name = name.split('module.controlnet.')[-1]
        controlnet_state_dict[name] = param
    elif "motion_modules." in name:
        if name.startswith('module.unet.'):
            name = name.split('module.unet.')[-1]
        motion_state_dict[name] = param
    elif "unet." in name:
        if name.startswith('module.unet.'):
            name = name.split('module.unet.')[-1]
        unet_state_dict[name] = param
    elif "image_proj_model." in name:
        # if name.startswith('module.unet.'):
        #     name = name.split('module.unet.')[-1]
        image_encoder_state_dict[name] = param
        print(name)
    elif "vae." in name:
        if name.startswith('module.vae.'):
            name = name.split('module.vae.')[-1]
        vae_state_dict[name] = param
    elif "text_encoder." in name:
        if name.startswith('module.text_encoder.'):
            name = name.split('module.text_encoder.')[-1]
        text_encoder_state_dict[name] = param
    else:
        print(name)
# print('appearance_encoder_state_dict', len(list(appearance_encoder_state_dict.keys())))
# print('controlnet_state_dict', len(list(controlnet_state_dict.keys())))
# print('motion_state_dict', len(list(motion_state_dict.keys())))
# m, u = appearance_encoder.load_state_dict(appearance_encoder_state_dict, strict=False)
# print(f"appearance_encoder missing keys: {len(m)}, unexpected keys: {len(u)}")
# assert len(u) == 0
# m, u = controlnet.load_state_dict(controlnet_state_dict, strict=False)
# print(f"controlnet missing keys: {len(m)}, unexpected keys: {len(u)}")
# assert len(u) == 0
# m, u = unet.load_state_dict(motion_state_dict, strict=False)
# print(f"motion_modules missing keys: {len(m)}, unexpected keys: {len(u)}")
# assert len(u) == 0
from animatediff.magic_animate.unet_controlnet import UNet3DConditionModel  
pretrained_model_path = '/data/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9'
unet_additional_kwargs = {
    "unet_use_cross_frame_attention": False,
    "unet_use_temporal_attention": False,
    "use_motion_module": True,
    "motion_module_resolutions": [1, 2, 4, 8],
    "motion_module_mid_block": False,
    "motion_module_decoder_only": False,
    "motion_module_type": "Vanilla",
    "motion_module_kwargs": {
        "num_attention_heads": 8,
        "num_transformer_block": 1,
        "attention_block_types": ["Temporal_Self", "Temporal_Self"],
        "temporal_position_encoding": True,
        "temporal_position_encoding_max_len": 24,
        "temporal_attention_dim_div": 1
    },
    "use_image_condition": True
}
unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet",
                                                                unet_additional_kwargs=unet_additional_kwargs)

# import pdb;pdb.set_trace()
# unet.image_proj_model
m, u = unet.load_state_dict(unet_state_dict, strict=False)
print(u)