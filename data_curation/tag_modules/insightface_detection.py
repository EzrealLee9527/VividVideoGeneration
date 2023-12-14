import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import cv2
import megfile
import jsonlines
from tqdm import tqdm
import os


def get_faces(videopath, app):
    cap = cv2.VideoCapture(videopath)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # ==> 总帧数
    if frame_num <= 3:
        return -1

    face_cnts = -1
    for idx in [0, frame_num//2, frame_num-1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        faces     = app.get(np.array(frame))
        face_cnts = max(len(faces), face_cnts)

    return face_cnts


# 输入首帧、中间帧以及末尾帧, 返回人脸的数量(三者最大值)
def worker(video_s3path_list, out_jsonl_path, cache_dir="/dev/shm/")-> float:
    
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    with jsonlines.open(out_jsonl_path,mode="w") as file_jsonl:

        for video_s3path in tqdm(video_s3path_list):
            video_name = video_s3path.split('/')[-1]
            video_localpath = megfile.smart_path_join(cache_dir, video_name)
            megfile.smart_copy(video_s3path, video_localpath)
        
            face_cnts = get_faces(video_localpath, app)
            
            file_jsonl.write({
                "video_clip":video_s3path,
                "face_cnts":face_cnts
            })

            os.remove(video_localpath)

# if __name__ == "__main__":

#     app = FaceAnalysis(providers=['CPUExecutionProvider'])
#     # app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#     app.prepare(ctx_id=0, det_size=(640, 640))


#     face_cnts = get_faces("split-scenes/4Wf6yPJPZPo-Scene-003.mp4", app)
#     print(f"{face_cnts}")


if __name__ == "__main__":    
    mp4path_list = list(megfile.smart_glob("s3://a-collections-sh/pexels/video/*.mp4"))
    megfile.smart_makedirs(f"/dev/shm/pexels-clips-insightface", exist_ok=True)
    worker(mp4path_list, "./pexels_clips_insightface.jsonl", cache_dir="/dev/shm/pexels-clips-insightface")
    print("done .")