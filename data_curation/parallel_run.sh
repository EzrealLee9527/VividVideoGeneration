# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# echo "Work dir"
# echo $SCRIPT_DIR
# cd "$SCRIPT_DIR"
# sudo apt-get install ninja-build
python_alias=/root/miniconda/envs/aigc/bin/python3
parallel=$1
logdir=./logs/video_meta2clip_meta
mkdir -p $logdir
for ((rank=0; rank<parallel; rank++))
do
echo $rank"/"$parallel
 WORLD=$parallel RANK=$rank  ${python_alias} filter_clip_dump_tar.py -i s3://weisiyuan-sh/datasets/CelebV/ -o s3://weisiyuan-sh/datasets/CelebV_webdataset/ \
    > $logdir/${rank}.txt  2>&1 &

done