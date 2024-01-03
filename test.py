from animatediff.magic_animate.appearance_encoder import AppearanceEncoderModel
from animatediff.magic_animate.controlnet import ControlNetModel
from megfile import smart_open, smart_exists, smart_sync, smart_remove, smart_glob
import torch
import io

pretrained_appearance_encoder_path = '/models00/pretrained_models/MagicAnimate/appearance_encoder'
appearance_controlnet_checkpoint_path = 'outputs/aa_train_stage1_celebv-2023-12-18T04-10-27/checkpoints/checkpoint.ckpt'
appearance_encoder = AppearanceEncoderModel.from_pretrained(pretrained_appearance_encoder_path)
pretrained_controlnet_path = "/models00/pretrained_models/MagicAnimate/densepose_controlnet"
controlnet = ControlNetModel.from_pretrained(pretrained_controlnet_path)

if appearance_controlnet_checkpoint_path != "":
        print(f"from checkpoint: {appearance_controlnet_checkpoint_path}")
        with smart_open(appearance_controlnet_checkpoint_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
            appearance_controlnet_checkpoint_path = torch.load(buffer, map_location="cpu")
        if "global_step" in appearance_controlnet_checkpoint_path: print(f"global_step: {appearance_controlnet_checkpoint_path['global_step']}")
        org_state_dict = appearance_controlnet_checkpoint_path["state_dict"] if "state_dict" in appearance_controlnet_checkpoint_path else appearance_controlnet_checkpoint_path        
        
        appearance_encoder_state_dict = {}
        controlnet_state_dict = {}
        for name, param in org_state_dict.items():
            if "appearance_encoder." in name:
                if name.startswith('module.appearance_encoder.'):
                    name = name.split('module.appearance_encoder.')[-1]
                appearance_encoder_state_dict[name] = param
            if "controlnet." in name:
                if name.startswith('module.controlnet.'):
                    name = name.split('module.controlnet.')[-1]
                controlnet_state_dict[name] = param
        print('appearance_encoder_state_dict', len(list(appearance_encoder_state_dict.keys())))
        print('controlnet_state_dict', len(list(controlnet_state_dict.keys())))
        m, u = appearance_encoder.load_state_dict(appearance_encoder_state_dict, strict=False)
        print(f"appearance_encoder missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
        m, u = controlnet.load_state_dict(controlnet_state_dict, strict=False)
        print(f"controlnet missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0



