CONFIG=$1
DEVICES=$2

lightning run model \
  --strategy deepspeed_stage_3 \
  --precision 16-mixed \
  --devices ${DEVICES} \
  main.py -c ${CONFIG}
