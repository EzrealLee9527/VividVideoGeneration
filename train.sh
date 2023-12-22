export MODEL_NAME="stabilityai/stable-video-diffusion-img2vid"
ABSOLUTE_DIRNAME=`pwd`
EXPNAME=`basename $ABSOLUTE_DIRNAME`

while1(){
  while true;do
      $@
      if [ $? = 233 ];then
          breakss
      fi
      sleep 10
  done
}

# 数据:OpenImages + InstanceSegMask
accelerate launch --config_file $1 train_svd_condition_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --per_gpu_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1000000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=500 \
  --output_dir="/dev/shm/"$EXPNAME"/saved" \
  --num_workers=8 \
  --checkpoints_total_limit=3 \
  --checkpointing_steps=1000 \
  --resume_from_checkpoint="latest" \
  --enable_xformers_memory_efficient_attention \
  --validation_steps=500 \
  --report_to="wandb" \
  --seed=99
#   --gradient_checkpointing \
  # --use_8bit_adam \
  # --use_ema \
  # --checkpoints_total_limit=3 \