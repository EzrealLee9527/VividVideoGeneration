from PIL import Image
import cv2
import torch
import megfile
import jsonlines
from tqdm import tqdm
import os
import numpy as np
from craft_text_detector import (
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
)


def text_detection(video_path, refine_net, craft_net):
    cap = cv2.VideoCapture(video_path)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # ==> 总帧数
    if frame_num <= 3:
        return -1 

    text_region_ratio_list = []
    for idx in [0, frame_num//2, frame_num-1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break 
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
        text_region_ratio_list.append(text_region_ratio)
    
    return float(np.array(text_region_ratio_list).max())


def worker(video_s3path_list, out_jsonl_dir, cache_dir="/dev/shm/")-> float:

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # load models
    refine_net = load_refinenet_model(cuda=True, weight_path="/data/users/jingminhao/.cache/craft/craft_refiner_CTW1500.pth")
    craft_net = load_craftnet_model(cuda=True, weight_path="/data/users/jingminhao/.cache/craft/craft_mlt_25k.pth")

    megfile.smart_makedirs(cache_dir, exist_ok=True)    

    ret_dic = {}
    for video_s3path in tqdm(video_s3path_list):
        video_name = video_s3path.split('/')[-1]
        video_localpath = megfile.smart_path_join(cache_dir, video_name)
        megfile.smart_copy(video_s3path, video_localpath)
        try:
            text_region_ratio = text_detection(video_localpath, refine_net, craft_net)
        except Exception as e:
            print(f"{e}, {video_s3path}, skip it ")
            text_region_ratio = -1
        
        cache_jsonl_path = megfile.smart_path_join(cache_dir, f"{video_name}.text_detection.jsonl")
        with jsonlines.open(cache_jsonl_path, mode="w") as file_jsonl:
            file_jsonl.write({
                "text_region_ratio":text_region_ratio
            })
        out_jsonl_path = megfile.smart_path_join(out_jsonl_dir, f"{video_name}.text_detection.jsonl")
        megfile.smart_move(cache_jsonl_path, out_jsonl_path)
        os.remove(video_localpath)
        ret_dic[video_s3path] = text_region_ratio
    return ret_dic


if __name__ == "__main__":    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default="s3://a-collections-sh/pexels/video")
    parser.add_argument('--out_dir', type=str, default="s3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/SVD/DataCuration/pexels/text_detection")
    parser.add_argument('--start_idx', type=int)
    parser.add_argument('--end_idx', type=int)
    args = parser.parse_args()    

    mp4path_list = list(megfile.smart_glob(megfile.smart_path_join(args.video_dir,"*.mp4")))[args.start_idx: args.end_idx]
    cache_dir = "/data/users/jingminhao/data/tempfile/pexels-clips-text-detection"
    megfile.smart_makedirs(cache_dir, exist_ok=True)
    worker(mp4path_list, args.out_dir, cache_dir=cache_dir)
    print("done .")    