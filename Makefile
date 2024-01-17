CHARGED_GROUP ?= monet_video
CPU ?= 192
GPU ?= 8
MEM_IN_GB ?= 768
NUM_NODES ?= 1
CONFIG ?=

override MEM = $$((${MEM_IN_GB}*1024))
override RLAUNCH_CMD = rlaunch --charged-group ${CHARGED_GROUP} --cpu ${CPU} --gpu ${GPU} --memory ${MEM} --replica ${NUM_NODES} --topo-group yes --preemptible no --private-machine yes
override DIST_TRAIN_CMD = sh train_dist.sh ${CONFIG} ${GPU}
override TRAIN_CMD = sh train.sh ${CONFIG} ${GPU}
override DIST_TRAIN_ACCU_CMD = sh train_dist_acc.sh ${CONFIG} ${GPU}

train:
	${RLAUNCH_CMD} -- ${TRAIN_CMD}

train_dist:
	${RLAUNCH_CMD} -- ${DIST_TRAIN_CMD}

train_dist_accu:
	${RLAUNCH_CMD} -- ${DIST_TRAIN_ACCU_CMD}

train_local:
	${TRAIN_CMD}

check:
	${RLAUNCH_CMD} --predict-only

debug:
	rlaunch --charged-group ${CHARGED_GROUP} --cpu ${CPU} --gpu ${GPU} --memory ${MEM} --preemptible no --private-machine yes -- zsh
