# train example
# stage1
train_stage1:
	rlaunch -P 10 --charged-group=monet_video --gpu=8 --cpu=32 --memory=1024000 --private-machine=yes -- bash run.sh configs/accelerate.yaml configs/training/train_stage1_w_imageencoder_celebv.yaml 80 train_with_dwpose.py
# stage2
train_stage2:
	rlaunch -P 8 --charged-group=monet_video --gpu=8 --cpu=32 --memory=1024000 --private-machine=yes -- bash run.sh configs/accelerate_accumulation_steps10.yaml configs/training/train_stage2_w_imageencoder_celebv.yaml 64 train_with_dwpose.py

# eval example
eval:
	rlaunch --charged-group=monet_video --gpu=1 --cpu=32 --memory=120000 --private-machine=yes -- python eval_v2.py --config configs/prompts/20230102_face.yaml

# 在主节点上：
torchrun --nproc_per_node=8 \
         --nnodes=4 \
         --node_rank=0 \
         --master_addr="192.18.54.154" \
         --master_port=12345 \
         train.py --config configs/training/train_stage_1_face_concat.yaml


# 在其他节点上：
torchrun --nproc_per_node=8 \
         --nnodes=4 \
         --node_rank=1 \
         --master_addr="192.18.54.154" \
         --master_port=12345 \
         train.py --config configs/training/train_stage_1_face_concat.yaml

torchrun --nproc_per_node=8 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr=localhost \
         --master_port=12345 \
         train.py --config configs/training/train_stage_1_face_concat.yaml
