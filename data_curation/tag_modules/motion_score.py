import numpy as np
import cv2
import megfile
import jsonlines
from tqdm import tqdm
import os


def get_global_motion_score(video_path, optflow_fps=1, optflow_shortest_px=16)-> float:
    cap = cv2.VideoCapture(video_path)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    assert optflow_fps <= fps, "计算光流的帧数不能高于原始视频的帧数"
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # ==> 总帧数

    if frame_height>frame_width:
        optflow_map_width = optflow_shortest_px
        optflow_map_height = int((optflow_shortest_px/frame_width)*frame_height)
    else:
        optflow_map_height = optflow_shortest_px
        optflow_map_width = int((optflow_shortest_px/frame_height)*frame_width)

    optflow_inp_frames = []
    optflow_out_mags = []
    for idx in range(0, frame_num-1, fps//optflow_fps):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        for _ in range(2):
            ret, frame = cap.read()
            if not ret:
                raise ValueError

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            optflow_inp_frames.append(frame_gray)
        
        flow = cv2.calcOpticalFlowFarneback(
            optflow_inp_frames[0], 
            optflow_inp_frames[1], 
            None, 
            0.5, # pyr_scale: 金字塔缩放的长度，如果为0.5，则是经典的金字塔图像，即下一层是前一层的一半；
            3,   # levels: 包含初始图像的层级， levels=1则没有创建额外的层，只使用原始的图像
            15,  # 平均窗口尺寸，值越大，会增加算法的鲁棒性可以应对快速的运动
            3,   # 迭代次数
            5,   # 每个像素的多项式展开，值越大，越平滑，一般设置为5或7
            1.1, # 标准差，如果 poly_n 为5， 则为1.1， 如果 poly_n 为 7， 则设置为 1.5
            0    # OPTFLOW_USE_INITIAL_FLOW， OPTFLOW_FARNEBACK_GAUSSIAN
        )

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mag_ds = cv2.resize(mag, dsize=(optflow_map_width, optflow_map_height),)
        optflow_out_mags.append(mag_ds)

    return float(np.array(optflow_out_mags).mean())


def worker(video_s3path_list, out_jsonl_dir, cache_dir="/dev/shm/")-> float:

    megfile.smart_makedirs(cache_dir, exist_ok=True)    


    ret_dic = {}
    for video_s3path in tqdm(video_s3path_list):
        video_name = video_s3path.split('/')[-1]
        video_localpath = megfile.smart_path_join(cache_dir, video_name)
        megfile.smart_copy(video_s3path, video_localpath)
        
        try:
            motion_score = get_global_motion_score(video_localpath)
            motion_score = float(motion_score)
        except Exception as e:
            print(f"{e}, {video_s3path}, skip it ")
            motion_score = -1

        cache_jsonl_path = megfile.smart_path_join(cache_dir, f"{video_name}.motionscore.jsonl")
        with jsonlines.open(cache_jsonl_path, mode="w") as file_jsonl:
            file_jsonl.write({
                "motion_score":motion_score
            })
        out_jsonl_path = megfile.smart_path_join(out_jsonl_dir, f"{video_name}.motionscore.jsonl")
        megfile.smart_move(cache_jsonl_path, out_jsonl_path)
        os.remove(video_localpath)
        
        ret_dic[video_s3path] = motion_score
    return ret_dic


if __name__ == "__main__":    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default="s3://a-collections-sh/pexels/video")
    parser.add_argument('--out_dir', type=str, default="s3://a-collections-sh/pexels/clips/motionscore")
    parser.add_argument('--start_idx', type=int)
    parser.add_argument('--end_idx', type=int)
    args = parser.parse_args()    

    mp4path_list = list(megfile.smart_glob(megfile.smart_path_join(args.video_dir,"*.mp4")))[args.start_idx: args.end_idx]
    cache_dir = "/data/users/jingminhao/data/tempfile/pexels-clips-motionscore"
    megfile.smart_makedirs(cache_dir, exist_ok=True)
    worker(mp4path_list, args.out_dir, cache_dir=cache_dir)
    print("done .")    