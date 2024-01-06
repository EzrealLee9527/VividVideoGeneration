import cv2
from PIL import Image
import open_clip
import torch
import megfile
import jsonlines
from tqdm import tqdm
import os
import numpy as np


def get_text_image_similarities(video_path, text, model, transform, tokenizer)-> float:
    cap = cv2.VideoCapture(video_path)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # ==> 总帧数
    if frame_num <= 3:
        return -1
    text = tokenizer([text])

    similarities = []
    for idx in [0, frame_num//2, frame_num-1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break 
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transform(image).unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image = image.to(torch.float16).to("cuda")
            text = text.to("cuda")

            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (image_features @ text_features.T).item()
            similarities.append(text_probs)

    return np.array(similarities).mean()


# 输入首帧、中间帧以及末尾帧, 返回人脸的数量(三者最大值)
def worker(video_s3path_list, text_dic, out_jsonl_dir, cache_dir="/dev/shm/")-> float:

    model, _, transform = open_clip.create_model_and_transforms(
        model_name="ViT-L-14",
        pretrained="datacomp_xl_s13b_b90k", # 会自动从HuggingFace下载对应的模型
        device="cuda",
        precision="fp16",

        # pretrained="/data/users/jingminhao/.cache/huggingface/hub/models--laion--mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/blobs/f22c34acef2b7a5d1ed28982a21077de651363eaaebcf34a3f10676e17837cb8"
    )
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    megfile.smart_makedirs(cache_dir, exist_ok=True)    

    ret_dic = {}
    for video_s3path in tqdm(video_s3path_list):
        video_name = video_s3path.split('/')[-1]
        video_localpath = megfile.smart_path_join(cache_dir, video_name)
        megfile.smart_copy(video_s3path, video_localpath)
        
        text = text_dic[video_s3path]
       
        if text != "":
            try:
                similarity = get_text_image_similarities(video_localpath, text, model, transform, tokenizer)
            except Exception as e:
                print(f"{e}, {video_s3path}, skip it ")
                similarity = -1
        else:
            similarity = -1
        
        cache_jsonl_path = megfile.smart_path_join(cache_dir, f"{video_name}.similarity.jsonl")
        with jsonlines.open(cache_jsonl_path, mode="w") as file_jsonl:
            file_jsonl.write({
                "similarity":similarity
            })
        out_jsonl_path = megfile.smart_path_join(out_jsonl_dir, f"{video_name}.similarity.jsonl")
        megfile.smart_move(cache_jsonl_path, out_jsonl_path)
        os.remove(video_localpath)

        ret_dic[video_s3path] = similarity
    
    return ret_dic

if __name__ == "__main__":    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default="s3://a-collections-sh/pexels/video")
    parser.add_argument('--text_jsonl', type=str, default="./results/pexels_clips_videoblip.jsonl")
    parser.add_argument('--out_dir', type=str, default="s3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/SVD/DataCuration/pexels/videoblip-similarity")
    parser.add_argument('--start_idx', type=int)
    parser.add_argument('--end_idx', type=int)
    args = parser.parse_args()    

    mp4path_list = list(megfile.smart_glob(megfile.smart_path_join(args.video_dir,"*.mp4")))[args.start_idx: args.end_idx]
    
    text_dic = {}
    with open(args.text_jsonl, "r+") as f:
        for item in tqdm(jsonlines.Reader(f)):
            video_path = item["video_clip"]
            text_dic[video_path] = item["caption_videoblip"]

    cache_dir = f"/data/users/jingminhao/data/tempfile/similarity-{args.text_jsonl.split('/')[-1].split('.')[0]}"
    megfile.smart_makedirs(cache_dir, exist_ok=True)
    worker(mp4path_list, text_dic, args.out_dir, cache_dir=cache_dir)
    print("done .")    