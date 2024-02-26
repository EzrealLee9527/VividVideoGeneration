unset http_proxy
unset https_proxy
# export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_IB_HCA=$(echo $NCCL_IB_HCA | tr ' ' ',')
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_NET_GDR_READ=1
export NCCL_TREE_THRESHOLD=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_CACHE_MAXSIZE=2147483647
export CUDA_CACHE_PATH=/root/.cache/yangshurong

    # --config_file configs/accelerate_debug.yaml \

# training stage 1    
# nohup accelerate launch \
#     train2d_for_3d.py \
#     --config configs/training_me/train1_cro08_for3d_refisindex4_stride2_codeback.yaml \
#     > train_a100_1.log &

# training stage 2   
nohup accelerate launch \
    --config_file configs/accelerate_debug.yaml \
    train3d_all.py \
    --config configs/training_me/train12_cro08_for3d_refisindex4_stride2_codeback_frommagic.yaml \
    > train_a100_1.log &
