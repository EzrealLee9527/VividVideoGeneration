PYTHON_CMD=/root/miniconda/envs/aigc/bin/python3
mkdir -p logs/celebv
mkdir -p logs/hdvila_100m
mkdir -p logs/xiaohongshu
# $PYTHON_CMD filter_human_videos_and_detect_clip.py -i s3://weisiyuan-sh/datasets/CelebV-Text/ -n CelebV --from_tar > logs/celebv/${HOSTNAME}.txt  2>&1 &
# CONTENT_DET_TH=10 $PYTHON_CMD filter_clip_dump_tar.py -i s3://nlp-data-map/video/hd-vila-100/hdvila_100m/download_videos -o s3://weisiyuan-sh/datasets/hdvila100m_20231216 > logs/hdvila_100m/${HOSTNAME}.txt  2>&1 &
# $PYTHON_CMD filter_human_videos_and_detect_clip.py -i "s3://weisiyuan-sh/datasets/xiaohongshu/*/vedio/" -n xiaohongshu_20231209 > logs/xiaohongshu/${HOSTNAME}.txt  2>&1 &
$PYTHON_CMD filter_clip_dump_tar.py -i s3://a-collections-sh/pexels/video/ -o s3://weisiyuan-sh/datasets/pexels_20231217
