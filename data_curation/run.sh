PYTHON_CMD=/root/miniconda/envs/aigc/bin/python3
mkdir -p logs/celebv
mkdir -p logs/hdvila_100m
mkdir -p logs/xiaohongshu

export PATH=/data/cuda/cuda-11.7/cuda/bin:$PATH
export LD_LIBRARY_PATH=/data/cuda/cuda-11.7/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/cuda/cuda-11.7/cudnn/v8.7.0/include:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/cuda/cuda-11.7/cudnn/v8.7.0/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/cuda/cuda-11.7/tensorrt/v8.5.3.1/lib:$LD_LIBRARY_PATH



# $PYTHON_CMD filter_human_videos_and_detect_clip.py -i s3://weisiyuan-sh/datasets/CelebV-Text/ -n CelebV --from_tar > logs/celebv/${HOSTNAME}.txt  2>&1 &
# CONTENT_DET_TH=10 $PYTHON_CMD filter_clip_dump_tar.py -i s3://nlp-data-map/video/hd-vila-100/hdvila_100m/download_videos -o s3://weisiyuan-sh/datasets/hdvila100m_20231216 > logs/hdvila_100m/${HOSTNAME}.txt  2>&1 &
# $PYTHON_CMD filter_human_videos_and_detect_clip.py -i "s3://weisiyuan-sh/datasets/xiaohongshu/*/vedio/" -n xiaohongshu_20231209 > logs/xiaohongshu/${HOSTNAME}.txt  2>&1 &
skip_movie_start_and_end_min=1 det_movie_cations_once=1 AUTO_DOWNSCALE=1 SHOW_SPLIT_PROGRESS=1 \
FRAME_SKIP=2 \
$PYTHON_CMD filter_clip_dump_tar.py -i s3://a-collections-sh/pexels/video/ -o s3://weisiyuan-sh/datasets/pexels_20231217
