
## Requirements

you need to do this below
```
conda env create -f environment.yaml
pip install -U openmim
# 自行安装torchvision cuda版本
# https://download.pytorch.org/whl/torch_stable.html
# wget https://download.pytorch.org/whl/cu117/torchvision-0.15.1%2Bcu117-cp310-cp310-linux_x86_64.whl
pip install torchvision-0.15.1%2Bcu117-cp310-cp310-linux_x86_64.whl

mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"
```

需要参考以下仓库安装，clone detectron2到当前目录下，以支持densepose
https://github.com/Flode-Labs/vid2densepose

所有模型参数，如果找不到的话，可以在
s3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/Exps/yangshurong/cache/magic_pretrain/
找到，需要先下载并在代码里替换成相应的路径

训练和推理均可以使用shell脚本运行

## Training example

```bash
sh train_a100_1.sh
```

## Evalutaion
```bash
sh infer_cuda2.sh
```

