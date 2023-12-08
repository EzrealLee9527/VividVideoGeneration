# Vivid Video Generation

## News
**2023.12.04**
- finetune from svd


## Training:

As an finetuning example, run

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --base configs/train/finetune_from_svd.yaml -t --devices 0,1,2,3,4,5,6,7
```

## DEBUG:
Reading datas from Object Storage is slow, as a debugging example, run

```
DEBUG=2 CUDA_VISIBLE_DEVICES=0, python main.py --base configs/train/finetune_from_svd.yaml -t --devices 0,
```