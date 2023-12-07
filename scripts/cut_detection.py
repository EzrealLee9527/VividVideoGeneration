import json
from scenedetect import detect, ContentDetector, split_video_ffmpeg
import megfile

def cut_detection(video_s3path,):
    video_name = video_s3path.split('/')[-1]
    video_localpath = f"/dev/shm/pexels/{video_name}"
    megfile.smart_copy(video_s3path, video_localpath)
    
    # 对于SuddenChange类型的分镜, 直接使用ContentDetector就可以了
    scene_list = detect(video_localpath, ContentDetector())
    if scene_list:
        split_video_ffmpeg(video_localpath, scene_list)
        clippath_list = list(megfile.smart_glob(f'{video_name.split(".")[0]}*.mp4'))
        for clippath in clippath_list:
            clipname = clippath.split("/")[-1]
            megfile.smart_move(
                clippath, 
                f"s3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/SVD/DataCuration/pexels/clips/{clipname}"
            )
    else:
        save_clippath = f"s3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/SVD/DataCuration/pexels/clips/{video_name}"
        megfile.smart_move(video_localpath, save_clippath)


if __name__ == "__main__":
    from tqdm import tqdm
    
    mp4path_list = list(megfile.smart_glob("s3://a-collections-sh/pexels/video/*.mp4"))
    megfile.smart_makedirs(f"/dev/shm/pexels", exist_ok=True)
    megfile.smart_makedirs(f"/dev/shm/pexels/clips", exist_ok=True)

    for mp4path in tqdm(mp4path_list):
        # mp4name = mp4path.split("/")[-1]
        # local_mp4path = f"/dev/shm/pexels/{mp4name}"y

        # megfile.smart_copy(mp4path, local_mp4path)
        cut_detection(mp4path)
        # megfile.smart_copy(mp4path, local_mp4path)