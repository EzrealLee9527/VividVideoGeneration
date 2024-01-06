import torch
from pytorchvideo.data.video import VideoPathHandler
import torch
from transformers import Blip2Processor
from video_blip.model import VideoBlipVisionModel, VideoBlipForConditionalGeneration, process
import torch
import megfile
import jsonlines
from tqdm import tqdm
import os
import numpy as np
import cv2
import os


def get_caption_videoblip(video_path, model, processor, device)-> float:

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # ==> 总帧数
    
    # 最多只处理100帧, 大约占用20G的显存
    if frame_num <= 100:
        used_frame_indices = range(0, frame_num)
    else:
        used_frame_indices = np.linspace(0, frame_num, 100, endpoint=False,).astype(np.int32) 
        used_frame_indices = np.unique(used_frame_indices)
        used_frame_indices = used_frame_indices.tolist()

    frames = []
    for idx in used_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame)
        frames.append(frame)
    # thwc_to_cthw
    frames = torch.stack(frames).permute(3, 0, 1, 2).to(torch.float32)
    
    inputs = process(processor, video=frames, text=None).to(device) # 尺寸会缩放到224x224
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=100)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    return generated_text


def worker(video_s3path_list, out_jsonl_dir, cache_dir="/dev/shm/")-> float:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained = "kpyu/video-blip-opt-2.7b-ego4d"
    processor = Blip2Processor.from_pretrained(pretrained)
    model = VideoBlipForConditionalGeneration.from_pretrained(pretrained).to(device)
    # pretrained = "kpyu/video-blip-flan-t5-xl-ego4d"
    # processor = Blip2Processor.from_pretrained(pretrained)
    # model = VideoBlipVisionModel.from_pretrained(pretrained).to(device)

    megfile.smart_makedirs(cache_dir, exist_ok=True)    

    ret_dic = {}
    for video_s3path in tqdm(video_s3path_list):
        video_name = video_s3path.split('/')[-1]
        video_localpath = megfile.smart_path_join(cache_dir, video_name)
        megfile.smart_copy(video_s3path, video_localpath)
        
        try:
            caption_videoblip = get_caption_videoblip(video_localpath, model, processor, device)
        except Exception as e:
            print(f"{e}, {video_s3path}, skip it ")
            caption_videoblip = ""
        
        cache_jsonl_path = megfile.smart_path_join(cache_dir, f"{video_name}.videoblip.jsonl")
        with jsonlines.open(cache_jsonl_path, mode="w") as file_jsonl:
            file_jsonl.write({
                "caption":caption_videoblip
            })
        out_jsonl_path = megfile.smart_path_join(out_jsonl_dir, f"{video_name}.videoblip.jsonl")
        megfile.smart_move(cache_jsonl_path, out_jsonl_path)
        os.remove(video_localpath)
        
        ret_dic[video_s3path] = caption_videoblip
    
    return ret_dic

if __name__ == "__main__":    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default="s3://a-collections-sh/pexels/video")
    parser.add_argument('--out_dir', type=str, default="s3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/SVD/DataCuration/pexels/videoblip")
    parser.add_argument('--start_idx', type=int)
    parser.add_argument('--end_idx', type=int)
    args = parser.parse_args()    

    mp4path_list = list(megfile.smart_glob(megfile.smart_path_join(args.video_dir,"*.mp4")))[args.start_idx: args.end_idx]
    cache_dir = "/data/users/jingminhao/data/tempfile/pexels-clips-caption-videoblip"
    megfile.smart_makedirs(cache_dir, exist_ok=True)
    worker(mp4path_list, args.out_dir, cache_dir=cache_dir)
    print("done .")    