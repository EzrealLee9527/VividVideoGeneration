import tempfile
import cv2
import matplotlib.pyplot as plt
from face_detection.insight_face_model import FaceAnalysis
from utils import volces_get_worker_cnt_worldsize
import os
import sys
from tqdm import tqdm
import traceback
import sys
import numpy as np
from functools import lru_cache
from data_fetcher import fetch_videos,generate_hash_from_paths,DEFAULT_TEMPFILE_DIR
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector,AdaptiveDetector,ThresholdDetector
from scenedetect.video_splitter import split_video_ffmpeg
from scenedetect.frame_timecode import FrameTimecode
from craft_text_detector import (
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
)

DEBUG = os.environ.get('DEBUG')
CONTENT_DET_TH = int(os.environ.get('CONTENT_DET_TH',20))

@lru_cache()
def get_text_det_models():
    refine_net = load_refinenet_model(cuda=True, weight_path="/data/cache/craft/craft_refiner_CTW1500.pth")
    craft_net = load_craftnet_model(cuda=True, weight_path="/data/cache/craft/craft_mlt_25k.pth")
    return refine_net,craft_net

def get_text_ratio(frame,refine_net,craft_net):
    # print(frame.shape)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # image = Image.fromarray(image)
    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        cuda=True,
        long_size=1280
    )
    if len(prediction_result["boxes"]):
        text_region_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        text_region_mask = cv2.fillPoly(
            text_region_mask, 
            prediction_result["boxes"].astype(np.int32), 
            1
        )
        text_region_ratio = text_region_mask.sum()/(text_region_mask.shape[0]*text_region_mask.shape[1])
    else:
        text_region_ratio = 0

    return text_region_ratio
def frametimecode2dict(code):
    
    return code.framerate, code.frame_num

def split_video_into_scenes(video_path,outf_template, threshold=27.0):
# Open our video, create a scene manager, and add a detector.
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=CONTENT_DET_TH,min_scene_len=15))
    scene_manager.add_detector(
        ThresholdDetector(min_scene_len=15))
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

import cv2
import megfile
from data_fetcher import dump_json


def sync_file(src,trg):
    megfile.smart_copy(src,trg)
    
def extract_frame(cap,resize_shape):
    
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = 0
    frame_list = []
    while True:
        # Read a new frame
        ret, frame = cap.read()
        
        # If we got frames, show and/or save them.
        if ret:
            if resize_shape is not None:
                frame = cv2.resize(frame,resize_shape)
            frame_list.append(frame)
            # Display the frame (for testing purposes)
        else:
            # No more frames are available; end the loop
            break
    return frame_list,fps

def check_video_per_sec_frame(cap,face_ana):
    
    if not cap.isOpened():
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    frame_num = 0
    # frame_list = []
    per_sec_frame_valid_flag_list = []
    per_sec_frame_face_num_list = []
    shape = None
    while True:
        # Read a new frame
        ret, frame = cap.read()
        
        # If we got frames, show and/or save them.
        if ret:
            if shape is None: 
                shape = frame.shape
            if frame_num%fps == 0:
            # frame_list.append(frame)
                res = face_ana.get(frame, max_num = 100)
                flag = check_frame(res,frame.shape[:2][::-1],(1,3))
                per_sec_frame_valid_flag_list.append(flag)
                per_sec_frame_face_num_list.append(len(res))
            frame_num +=1
        else:
            # No more frames are available; end the loop
            break
    return per_sec_frame_face_num_list,per_sec_frame_valid_flag_list,fps,frame_num,shape


def check_frame(res,shape, num_bbox_range = (1,3)):
    # shape: w,h
    # bbox contain coordinates: x1,y1,x2,y2
    n_face = len(res)
    if not (n_face >= num_bbox_range[0] and n_face <= num_bbox_range[1]):
        return False
    
    
    def is_valid(box):
        w,h = shape
        x1,y1,x2,y2 = box
        center = ((x1+x2)//2,(y1+y2)//2)
        size = (x2-x1)*(y2-y1)
        
        center_range = [(w//4,3*w//4),(h//4,3*h//4)] 
        size_range = (w//5*h//5,w*h)
        
        
        if size> size_range[0] and size < size_range[1] and \
            center[0]>center_range[0][0] and center[0]<center_range[0][1] and \
            center[1]>center_range[1][0] and center[1]<center_range[1][1]:
                return True
        else:
            return False
        
    containing_valid_face = any([
        is_valid(re['bbox'])
        for re in res
    ])
    return containing_valid_face
        
        
def write_video(frames,video_name,fps):
    
    fps = 1
    frames = [cv2.resize(f,(112,112)) for f in frames][::fps]
    frame_height, frame_width = frames[0].shape[:2]

    # 定义视频编码器和创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 也可以使用其他编码器，如'MP4V', 'MJPG'等
    video_writer = cv2.VideoWriter(video_name, fourcc, fps, (frame_width, frame_height))

    print('start write')
    # 将每一帧图像写入视频
    for frame in tqdm(frames):
        video_writer.write(frame)

    # 释放VideoWriter对象
    video_writer.release()
    
    
import subprocess

def ffmpeg_split_video(input_video_path,output_video_path,start_frame,end_frame,fps):
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps

    # 构建ffmpeg命令
    ffmpeg_command = [
        '/usr/bin/ffmpeg',
        '-v','quiet',
        '-ss', str(start_time),  # 开始时间
        '-i', input_video_path,  # 输入视频文件
        '-t', str(duration),     # 视频时长
        # '-c', 'copy',            # 使用"copy"参数快速裁剪而不重新编码
        
        '-c:v', 'libx264',             # 视频编解码器
            '-crf', '23',                  # 输出视频质量控制参数
        # '-filter:v', f'fps=fps=1',          # 设置新的帧率
        '-s', f'200x200',       # 设置新的分辨率
        output_video_path        # 输出视频文件
    ]

    # 执行ffmpeg命令
    subprocess.run(ffmpeg_command)
    
import json

def extract_key_frame_info(video_file):
    # 计算起点和终点的时间戳（秒）
    
    # 使用ffmpeg获取关键帧列表
    command = [
        'ffprobe',
        '-loglevel', 'error',  # 隐藏非错误信息
        '-select_streams', 'v:0',  # 选择第一个视频流
        '-show_entries', 'frame=pkt_pts_time,pict_type',  # 显示时间戳和帧类型
        '-of', 'json',  # 输出格式为 JSON
        video_file
    ]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise Exception('ffprobe error: ' + result.stderr.decode())
    
    frames_info = json.loads(result.stdout)
    return frames_info


def find_cloest_key_frame(frames_info, frame_start,frame_end,fps):
    
    time_start,time_end = frame_start / fps, frame_end / fps
    closest_start_key_frame_time = None
    closest_end_key_frame_time = None
    
    # 遍历所有帧，查找关键帧
    for frame in frames_info.get('frames', []):
        if frame.get('pict_type') == 'I':
            key_frame_time = float(frame['pkt_pts_time'])
            
            # 寻找最接近起始时间的关键帧
            if closest_start_key_frame_time is None or abs(key_frame_time - time_start) < abs(closest_start_key_frame_time - time_start):
                closest_start_key_frame_time = key_frame_time
            
            # 寻找最接近结束时间的关键帧
            if closest_end_key_frame_time is None or abs(key_frame_time - time_end) < abs(closest_end_key_frame_time - time_end):
                closest_end_key_frame_time = key_frame_time
    
    # 计算帧数
    closest_start_key_frame_count = round(closest_start_key_frame_time * fps)
    closest_end_key_frame_count = round(closest_end_key_frame_time * fps)
    
    return closest_start_key_frame_count, closest_end_key_frame_count


def extract_frame(input_file, frame_offset,shape):
    # FFmpeg命令模板，其中设置-frame_pos参数抓取指定位置的帧
    command = [
        'ffmpeg',
        '-i', input_file,            # 输入文件路径
        '-vf', 'select=gte(n\,{})'.format(frame_offset),  # 设置frame_offset过滤器
        '-vframes', '1',             # 指定只输出一个帧
        '-f', 'image2pipe',          # 告诉FFmpeg通过管道输出图像
        '-pix_fmt', 'bgr24',         # 设置图像格式为bgr24 (BGR色彩通道)
        '-vcodec', 'rawvideo',       # 输出未压缩的原始视频帧
        'pipe:1'                     # 将输出发送到标准输出
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()

    frame = np.frombuffer(out, np.uint8).reshape(*shape)

    return frame

@lru_cache()
def get_face_ana():
    face_ana = FaceAnalysis(allowed_modules= [
        # 'landmark_3d_68',
        'detection',
        # 'genderage',
        # 'recognition'
        ])
    face_ana.prepare(ctx_id=0, det_size=(640, 640))
    return face_ana

def filter_single_video(tmpfile):
    text_region_ratio_thresold = 0.07 # from SVD appendix
    face_ana = get_face_ana()
    refine_net,craft_net = get_text_det_models()
    try:
        # load all frames leads to OOM
        cap = cv2.VideoCapture(tmpfile)
        per_sec_frame_face_num_list,per_sec_frame_valid_flag_list ,fps,frame_num,shape = check_video_per_sec_frame(cap,face_ana)
        cap.release()
        
        if not any(per_sec_frame_face_num_list):
            return None
        frame_num_per_sec = int(fps)
    
        scene_list = split_video_into_scenes(tmpfile,None)
        # print(f'scene_list : {len(scene_list)}')
        if scene_list:
            frame_num_level_scene_list = [ (clip[0].frame_num,clip[1].frame_num) for clip in scene_list]
        else:
            frame_num_level_scene_list = [[0,frame_num-1]]
        valid_frame_num_level_scene_list = []

        
        clip_meta_list = []
        keyframe_info = extract_key_frame_info(tmpfile)
        for frame_st,frame_ed in frame_num_level_scene_list:
            
            # find the middle time, round it in the second level
            mid_sec_idx = int(round(  (frame_st +frame_ed) //2 / fps ))
            if not (mid_sec_idx > frame_st/fps and mid_sec_idx < frame_ed/fps):
                continue
            elif per_sec_frame_valid_flag_list[mid_sec_idx]:
                # faces bboxes check pass
                
                mid_sec_frame = extract_frame(tmpfile, int(mid_sec_idx * fps),shape )
                text_area_ratio = get_text_ratio(mid_sec_frame,refine_net,craft_net)
                
                if text_area_ratio>text_region_ratio_thresold:
                    continue
                
                n_face = per_sec_frame_face_num_list[mid_sec_idx]
                clip_meta_list.append(
                    dict(n_face = n_face,text_area_ratio = text_area_ratio)
                )
                
                fixed_frame_st,fixed_frame_ed = find_cloest_key_frame(
                    keyframe_info, frame_st,frame_ed,fps
                )
                valid_frame_num_level_scene_list.append((fixed_frame_st,fixed_frame_ed))

                if DEBUG:
                    ffmpeg_split_video(
                        tmpfile,
                        os.path.join('debug',f'{fname}-{frame_st}-{frame_ed}.mp4'),
                        fixed_frame_st,
                        fixed_frame_ed,
                        fps
                    )
    
        
        return {
            'clips':valid_frame_num_level_scene_list,
            'fps':fps,
            'total_frames':frame_num,
            'ori_shape':shape[:2],
            'clip_meta_list':clip_meta_list
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc(file=sys.stderr)
        return None
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=' svd data curation')
    parser.add_argument(
        "-i",
        "--indir",
        type=str,
        default="s3://nlp-data-map/video/hd-vila-100/hdvila_100m/download_videos"
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="hdvila_100m_meta"
    )
    parser.add_argument(
        "--from_tar",
        
        action='store_true',
        default=False,
    )
    args = parser.parse_args()
    
    
    worke_cnt, worldsize = volces_get_worker_cnt_worldsize()
    
    refine_net, craft_net = get_text_det_models()
    face_ana = FaceAnalysis(allowed_modules= [
        # 'landmark_3d_68',
        'detection',
        # 'genderage',
        # 'recognition'
        ])
    face_ana.prepare(ctx_id=0, det_size=(640, 640))
    rootdir = args.indir
    meta_file = f's3://weisiyuan-sh/datasets/{args.name}/worker{worke_cnt}.json'
    tmpdir = '/data/users/weisiyuan/tmp/'.rstrip("/")
    text_region_ratio_thresold = 0.07 # from SVD appendix
    
    
    # decide the mode of fetching video files
    filetypes = ['mp4','mkv']
    if not args.from_tar:
        fetcher = fetch_videos(rootdir,filetypes)
    else:
        from data_fetcher import fetch_video_from_tars
        fetcher = fetch_video_from_tars(rootdir,filetypes)
    meta = dict()
    for fidx,item in tqdm(enumerate(fetcher)):
        
        
        if fidx % worldsize != worke_cnt:
            continue
        
        if not args.from_tar:
            fpath = item
            tmpfile = os.path.join(tmpdir, f'{generate_hash_from_paths(fpath)}.mp4')
            sync_file(fpath,tmpfile)
        else:
            fpath,fbytes = item
            tmpfile = os.path.join(tmpdir, f'{generate_hash_from_paths(fpath)}.mp4')
            with open(tmpfile,'wb') as fio:
                fio.write(fbytes)
        try:
            
            
            fname = os.path.basename(fpath).rsplit('.',1)[0]
            # load all frames leads to OOM
            cap = cv2.VideoCapture(tmpfile)
            per_sec_frame_face_num_list,per_sec_frame_valid_flag_list ,fps,frame_num,shape = check_video_per_sec_frame(cap,face_ana)
            cap.release()
            
            if not any(per_sec_frame_face_num_list):
                os.remove(tmpfile)
                continue
            frame_num_per_sec = int(fps)
            
            
            scene_list = split_video_into_scenes(tmpfile,None)
            # print(f'scene_list : {len(scene_list)}')
            if scene_list:
                frame_num_level_scene_list = [ (clip[0].frame_num,clip[1].frame_num) for clip in scene_list]
            else:
                frame_num_level_scene_list = [[0,frame_num-1]]
            valid_frame_num_level_scene_list = []

            
            clip_meta_list = []
            keyframe_info = extract_key_frame_info(tmpfile)
            for frame_st,frame_ed in frame_num_level_scene_list:
                
                # find the middle time, round it in the second level
                mid_sec_idx = int(round(  (frame_st +frame_ed) //2 / fps ))
                if not (mid_sec_idx > frame_st/fps and mid_sec_idx < frame_ed/fps):
                    continue
                elif per_sec_frame_valid_flag_list[mid_sec_idx]:
                    # faces bboxes check pass
                    
                    mid_sec_frame = extract_frame(tmpfile, int(mid_sec_idx * fps),shape )
                    text_area_ratio = get_text_ratio(mid_sec_frame,refine_net,craft_net)
                    
                    if text_area_ratio>text_region_ratio_thresold:
                        continue
                    
                    n_face = per_sec_frame_face_num_list[mid_sec_idx]
                    clip_meta_list.append(
                        dict(n_face = n_face,text_area_ratio = text_area_ratio)
                    )
                    
                    fixed_frame_st,fixed_frame_ed = find_cloest_key_frame(
                        keyframe_info, frame_st,frame_ed,fps
                    )
                    valid_frame_num_level_scene_list.append((fixed_frame_st,fixed_frame_ed))

                    if DEBUG:
                        ffmpeg_split_video(
                            tmpfile,
                            os.path.join('debug',f'{fname}-{frame_st}-{frame_ed}.mp4'),
                            fixed_frame_st,
                            fixed_frame_ed,
                            fps
                        )
        
            
            meta[fpath] = {
                'clips':valid_frame_num_level_scene_list,
                'fps':fps,
                'total_frames':frame_num,
                'ori_shape':shape[:2],
                'clip_meta_list':clip_meta_list
            }
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc(file=sys.stderr)
        os.remove(tmpfile)
        
    if not DEBUG:
        dump_json(meta,meta_file)