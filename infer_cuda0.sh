
clear
export CUDA_VISIBLE_DEVICES=0




# python create_control_from_file.py \
#     --config ./configs/prompts_me/create_data.yaml \
#     > infer_cuda0.log

python eval_v2.py \
    --config ./configs/prompts_me/infer12_cro08_for3d_fromone_refisindex4_stride2_codeback_newdata.yaml \
    > infer_cuda0.log

