unset http_proxy
unset https_proxy
export NCCL_IB_HCA=$(echo $NCCL_IB_HCA | tr ' ' ',')
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_NET_GDR_READ=1
export NCCL_TREE_THRESHOLD=0
export CUDA_VISIBLE_DEVICES=0
export CUDA_CACHE_MAXSIZE=2147483647
export CUDA_CACHE_PATH=/root/.cache/yangshurong

nohup accelerate launch \
    --config_file configs/accelerate_debug.yaml \
    train2d_from_same_video_new.py \
    --config configs/training_me/train1_cro08_withback_fromone_concatorigin2noise_refisindex4_datacelebv_stride2.yaml \
    > debug.log &

# clear
# export CUDA_VISIBLE_DEVICES=0
# python eval_v2.py \
#     --config ./configs/prompts_me/origin_goodata_debug.yaml \
#     > infer.log