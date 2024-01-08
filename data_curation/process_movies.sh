
python_alias=python3
parallel=$1
logdir=./logs/video_meta2clip_meta
mkdir -p $logdir
for ((rank=0; rank<parallel; rank++))
do
echo $rank"/"$parallel
 WORLD=$parallel RANK=$rank  AUTO_DOWNSCALE=1 \
 ${python_alias} filter_clip_dump_tar.py -i "s3://ljj/Datasets/Raw/movie/20231228/**/" -o "s3://ljj/Datasets/Videos/processed/movies_batch20231228" \
    > $logdir/${rank}.txt  2>&1 &

done