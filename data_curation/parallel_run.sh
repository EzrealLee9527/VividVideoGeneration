# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# echo "Work dir"
# echo $SCRIPT_DIR
# cd "$SCRIPT_DIR"
# sudo apt-get install ninja-build


export PATH=/data/cuda/cuda-11.7/cuda/bin:$PATH
export LD_LIBRARY_PATH=/data/cuda/cuda-11.7/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/cuda/cuda-11.7/cudnn/v8.7.0/include:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/cuda/cuda-11.7/cudnn/v8.7.0/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/cuda/cuda-11.7/tensorrt/v8.5.3.1/lib:$LD_LIBRARY_PATH

parallel=$1
logdir=./logs/filter_clip_dump_tar

skip_movie_start_and_end_min=1 det_movie_cations_once=1 AUTO_DOWNSCALE=1 FRAME_SKIP=1 \
# rlaunch --cpu=16 --memory=98304 --charged-group core --replica $parallel -- \
#     --replica-restart never
#     python3 filter_clip_dump_tar.py \
#     -i "s3://ljj/Datasets/Raw/movie/20231228/**/" \
#      -o s3://ljj/Datasets/Videos/processed/movies_20240117_aesthetics5 \
#      -c ./logs/filter_clip_dump_tar.processed_videos.log


skip_movie_start_and_end_min=1 det_movie_cations_once=1 AUTO_DOWNSCALE=1 FRAME_SKIP=1 \
    rlaunch --cpu=16 --memory=98304 --charged-group core --replica $parallel -- \
    python3 filter_clip_dump_tar.py \
    -i "s3://ljj/Datasets/Raw/movie/20231228/**/" \
     -o s3://ljj/Datasets/Videos/processed/movies_20240117_aesthetics5_resume \
     -c logs/movies_20240117_aesthetics5.processed_src_videos.txt

# mkdir -p $logdir
# for ((rank=0; rank<parallel; rank++))
# do
# echo $rank"/"$parallel
#  WORLD=$parallel RANK=$rank \
#  SHOW_SPLIT_PROGRESS=1 skip_movie_start_and_end_min=1 det_movie_cations_once=1 AUTO_DOWNSCALE=1 FRAME_SKIP=1 \
#  python3 filter_clip_dump_tar.py -i "s3://ljj/Datasets/Raw/movie/20231228/**/" -o s3://ljj/Datasets/Videos/processed/movies_20240111\
#     > $logdir/${rank}.txt  2>&1 &

done