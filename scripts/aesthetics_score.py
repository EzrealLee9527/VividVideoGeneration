import torch
from PIL import Image
from transformers import CLIPProcessor
from aesthetics_predictor import AestheticsPredictorV1
import cv2
import torch
import megfile
import jsonlines
from tqdm import tqdm
import os
import numpy as np


def get_aesthetic_score(video_path, model, processor, device)-> float:
    cap = cv2.VideoCapture(video_path)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # ==> 总帧数
    if frame_num <= 3:
        return -1

    aesthetic_score_list = []
    for idx in [0, frame_num//2, frame_num-1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break 
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        inputs = processor(images=image, return_tensors="pt")
        
        model = model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad(): # or `torch.inference_model` in torch 1.9+
            outputs = model(**inputs)
        aesthetic_score = outputs.logits.item()
        aesthetic_score_list.append(aesthetic_score)

    return np.array(aesthetic_score_list).mean()


def worker(video_s3path_list, out_jsonl_dir, cache_dir="/dev/shm/")-> float:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained = "shunk031/aesthetics-predictor-v1-vit-large-patch14"
    model = AestheticsPredictorV1.from_pretrained(pretrained)
    processor = CLIPProcessor.from_pretrained(pretrained)


    # model = VideoBlipVisionModel.from_pretrained(pretrained).to(device)

    for video_s3path in tqdm(video_s3path_list):
        video_name = video_s3path.split('/')[-1]
        video_localpath = megfile.smart_path_join(cache_dir, video_name)
        megfile.smart_copy(video_s3path, video_localpath)
        
        try:
            aesthetic_score = get_aesthetic_score(video_localpath, model, processor, device)
        except Exception as e:
            print(f"{e}, {video_s3path}, skip it ")
            aesthetic_score = -1
        
        cache_jsonl_path = megfile.smart_path_join(cache_dir, f"{video_name}.aesthetic.jsonl")
        with jsonlines.open(cache_jsonl_path, mode="w") as file_jsonl:
            file_jsonl.write({
                "video_clip":video_s3path,
                "aesthetic_score":aesthetic_score
            })
        out_jsonl_path = megfile.smart_path_join(out_jsonl_dir, f"{video_name}.aesthetic.jsonl")
        megfile.smart_move(cache_jsonl_path, out_jsonl_path)
        os.remove(video_localpath)


if __name__ == "__main__":    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default="s3://a-collections-sh/pexels/video")
    parser.add_argument('--out_dir', type=str, default="s3://a-collections-sh/pexels/clips/aesthetic")
    parser.add_argument('--start_idx', type=int)
    parser.add_argument('--end_idx', type=int)
    args = parser.parse_args()    

    mp4path_list = list(megfile.smart_glob(megfile.smart_path_join(args.video_dir,"*.mp4")))[args.start_idx: args.end_idx]
    cache_dir = "/data/users/jingminhao/data/tempfile/pexels-clips-aesthetic"
    megfile.smart_makedirs(cache_dir, exist_ok=True)
    worker(mp4path_list, args.out_dir, cache_dir=cache_dir)
    print("done .")    