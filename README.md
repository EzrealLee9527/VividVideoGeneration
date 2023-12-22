# Finetuning SVD with Diffusers+Accelerate .
# News
**2023.12.22**
- support for fine-tuning Image2Video SVD. I will record a detailed worklog in https://wiki.megvii-inc.com/pages/viewpage.action?pageId=523140390 later.

# environment setting
- conda ：environment.yaml

# Train:
`bash ./train.sh accelerate_configs/8gpu_fp16_config.yaml`