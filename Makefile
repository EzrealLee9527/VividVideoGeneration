CHARGED_GROUP ?= monet_video
CPU ?= 192
GPU ?= 8
MEM_IN_GB ?= 768
CONFIG ?= 

override MEM = $$((${MEM_IN_GB}*1024))

override TRAIN_RLAUNCH = rlaunch --charged-group ${CHARGED_GROUP} --cpu ${CPU} --gpu ${GPU} --memory ${MEM} --preemptible no --private-machine yes

override TRAIN_COMMAND = python3 -u main.py -c ${CONFIG}

# rlaunch
train:
	${TRAIN_RLAUNCH} -- ${TRAIN_COMMAND}

# local
train_local:
	${TRAIN_COMMAND}

check:
	${TRAIN_RLAUNCH} --predict-only

debug:
	${TRAIN_RLAUNCH} -- zsh

