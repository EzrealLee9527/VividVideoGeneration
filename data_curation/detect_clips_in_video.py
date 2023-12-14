import tempfile
import os
import sys
from tqdm import tqdm
from data_fetcher import fetch_video_from_tars,generate_hash_from_paths,DEFAULT_TEMPFILE_DIR
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
from scenedetect.frame_timecode import FrameTimecode


def frametimecode2dict(code):
    
    return code.framerate, code.frame_num

def split_video_into_scenes(video_path,outf_template, threshold=27.0):
# Open our video, create a scene manager, and add a detector.
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()
    
    return scene_list

    if len(scene_list)>0:
        #  ffmpeg error
        # NOTE: do not use ffmpeg from conda( check it by observing the path prefix of command "ffmpeg")
        # add other ffmpeg bin path to the PATH env
        split_video_ffmpeg(video_path, scene_list, output_file_template=outf_template,show_progress=True)
        
        
def delete_file(file):
    os.remove(file)
    
def add_suffix(path,suffix):
    
    return path.rsplit('.',1)[0] +f'.{suffix}'


import megfile
from data_fetcher import dump_json,load_json


if __name__ == "__main__":
        
    meta_file = 's3://weisiyuan-sh/datasets/CelebV-Text-clip.json'
    outpath = '/data/users/weisiyuan/tmp/'.rstrip("/")
    
    meta = dict()
    
    exist_meta_file = megfile.smart_exists(meta_file)
    if exist_meta_file:
        meta = load_json(meta_file)
    for fpath,fbytes in tqdm(
        fetch_video_from_tars(
            's3://weisiyuan-sh/datasets/CelebV-Text/'
        )
    ):  
        if exist_meta_file and fpath in meta:
            continue
        
        fbytes = fbytes.read()
        if len(fbytes) == 0:
            print('skip')
            continue

        # fpath = generate_hash_from_paths(fpath)
        with tempfile.NamedTemporaryFile(prefix=generate_hash_from_paths(fpath),
                                         suffix='.mp4',
                                         mode = 'wb',
                                         dir = DEFAULT_TEMPFILE_DIR,
                                         delete=True) as tmpvideo:
            tmpvideo.write(fbytes)
        
            try:
                            # clip_outpath = os.path.join(outpath,fpath)
            # os.makedirs(clip_outpath,exist_ok=True)
                scene_list = split_video_into_scenes(tmpvideo.name,outf_template=f'debug/$VIDEO_NAME-$SCENE_NUMBER-$START_FRAME-$END_FRAME.mp4')
                if len(scene_list) == 0:
                    scene_list = []
                else:
                    fps = scene_list[0][0].framerate
                    frame_num_level_scene_list = [
                        
                        (clip[0].frame_num,clip[1].frame_num)
                        for clip in scene_list
                    ]
                meta[fpath] = frame_num_level_scene_list
            except:
                continue
    
    dump_json(meta,meta_file)