
## Requirements

see 'requirements.txt'

To support DWPose which is dependent on MMDetection, MMCV and MMPose
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"
```


## Training example

```bash
rlaunch -P 4 --charged-group=monet_video --gpu=8 --cpu=32 --memory=1024000 --private-machine=yes -- bash run.sh configs/accelerate.yaml configs/training/train_stage1_w_imageencoder_celebv_nocenterface_cfg_train_zeroembed_plusbank0.5_ipadapter.yaml 32 train_with_face.py
```

## Evalutaion
```bash
rlaunch --charged-group=monet_video --gpu=1 --cpu=32 --memory=120000 --private-machine=yes -- python eval_v2.py --config configs/prompts_0115/eval_pexels_g2_magicmm_gif_magic_plusbank0.5_cwt.yaml

```

