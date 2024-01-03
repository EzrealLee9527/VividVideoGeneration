# setup NCCL and InfiniteBand
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106
export NCCL_IB_HCA=mlx5_0  # Removed additional equals sign here
export NCCL_TREE_THRESHOLD=0

CONFIG=$1
DEVICES=$2
NUM_NODES=${RLAUNCH_REPLICA_TOTAL}
NODE_RANK=${RLAUNCH_REPLICA}

# get master ip addr
if [ "${NODE_RANK}" -eq 0 ]; then
  MASTER_IP=$(ip addr show | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | cut -d/ -f1 | head -n 1)
  echo ${MASTER_IP} > MASTER_ADDR
else
  sleep 5
  if [ ! -f MASTER_ADDR ]; then
    echo "Error: MASTER_ADDR file not found."
    exit 1
  fi
fi
MASTER_ADDR=$(cat MASTER_ADDR)

lightning run model \
  --strategy deepspeed_stage_3 \
  --precision 16-mixed \
  --devices ${DEVICES} \
  --num-nodes ${NUM_NODES} \
  --node-rank ${NODE_RANK} \
  --main-address ${MASTER_ADDR} \
  main.py -c ${CONFIG}
