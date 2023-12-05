import cv2
from PIL import Image
import open_clip
import torch
import megfile
import jsonlines
from tqdm import tqdm
import os


def get_caption_coca(video_path, coca_net, transform)-> float:
    cap = cv2.VideoCapture(video_path)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # ==> 总帧数
    mid_frame_idx = frame_num//2

    # model, _, transform = open_clip.create_model_and_transforms(
    #     model_name="coca_ViT-L-14",
    #     pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    # )

    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
    ret, midframe = cap.read()
    if not ret:
        raise IndexError

    im = Image.fromarray(midframe[...,::-1])
    im = transform(im).unsqueeze(0)
    # print(im.dtype)
    with torch.no_grad(), torch.cuda.amp.autocast():
        im = im.to(torch.float16).to("cuda")
        generated = coca_net.generate(im)

    # print(open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", ""))
    return open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")


# 输入首帧、中间帧以及末尾帧, 返回人脸的数量(三者最大值)
def worker(video_s3path_list, out_jsonl_path, cache_dir="/dev/shm/")-> float:

    model, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained="mscoco_finetuned_laion2B-s13B-b90k", # 会自动从HuggingFace下载对应的模型
        device="cuda",
        precision="fp16",

        # pretrained="/data/users/jingminhao/.cache/huggingface/hub/models--laion--mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/blobs/f22c34acef2b7a5d1ed28982a21077de651363eaaebcf34a3f10676e17837cb8"
    )

    with jsonlines.open(out_jsonl_path,mode="w") as file_jsonl:

        for video_s3path in tqdm(video_s3path_list):
            video_name = video_s3path.split('/')[-1]
            video_localpath = megfile.smart_path_join(cache_dir, video_name)
            megfile.smart_copy(video_s3path, video_localpath)
        
            caption_coca = get_caption_coca(video_localpath, model, transform)
            
            file_jsonl.write({
                "video_clip":video_s3path,
                "caption_coca":caption_coca
            })

            os.remove(video_localpath)


if __name__ == "__main__":    
    mp4path_list = list(megfile.smart_glob("s3://a-collections-sh/pexels/video/*.mp4"))
    megfile.smart_makedirs(f"/dev/shm/pexels-clips-caption-coca", exist_ok=True)
    worker(mp4path_list, "./pexels_clips_caption_coca.jsonl", cache_dir="/dev/shm/pexels-clips-caption-coca")
    print("done .")    