# Finetuning SVD with Diffusers+Accelerate .
# News
**2023.12.25**
- support for FSDP now. I test on SDP configure: accelerate_configs/8gpu_fp16_sdp_config.yaml

**2023.12.22**
- support for fine-tuning Image2Video SVD. I will record a detailed worklog in https://wiki.megvii-inc.com/pages/viewpage.action?pageId=523140390 later.

# environment setting
- conda ：environment.yaml

# Train:
`bash ./train.sh accelerate_configs/8gpu_fp16_sdp_config.yaml` for fsdp
`bash ./train.sh accelerate_configs/8gpu_fp16_config.yaml` for ddp