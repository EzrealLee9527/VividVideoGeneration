# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# echo "Work dir"
# echo $SCRIPT_DIR
# cd "$SCRIPT_DIR"
# sudo apt-get install ninja-build
python_alias=/root/miniconda/envs/aigc/bin/python3
parallel=$1
mkdir ./logs
for ((rank=0; rank<parallel; rank++))
do
echo $rank"/"$parallel
 CONTENT_DET_TH=10 WORLD=$parallel RANK=$rank  ${python_alias}  filter_human_videos_and_detect_clip.py -i s3://nlp-data-map/video/hd-vila-100/hdvila_100m/download_videos -n hdvila_100m_20231209 > logs/${script}.${rank}.txt  2>&1 &

done