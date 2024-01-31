import tarfile
import os
import json
import megfile
from tqdm import tqdm
def has_duplicate_members(tar_path):
    with megfile.smart_open(tar_path, "rb") as fio:
        with tarfile.open(fileobj=fio, mode = "r") as tar:
            members = {}
            for member in tar.getmembers():
                if member.name in members:
                    print(f"Duplicate member found: {member.name} in {tar_path}")
                    return True
                members[member.name] = True
        return False

def read_all_src_videos(tar_path):
    
    src_videos = set()
    with megfile.smart_open(tar_path, "rb") as fio:
        with tarfile.open(fileobj=fio, mode = "r") as tar:
            members = {}
            for member in tar.getmembers():
                if member.name.endswith('.json'):
                    json_data = tar.extractfile(member).read().decode('utf-8')
            
                    # 解析JSON数据
                    try:
                        json_obj = json.loads(json_data)
                        src_videos.add(json_obj['src_video'])
                    except json.JSONDecodeError as e:
                        print("Error decoding JSON:", e)
                    break  #
                    
        return src_videos

# 替换成你的tar文件路径
all_src_videos = []
for tar_file_path in tqdm(megfile.smart_glob(
    os.path.join("s3://ljj/Datasets/Videos/processed/movies_20240117_aesthetics5/",'*.tar')
)):

    src_videos = read_all_src_videos(tar_file_path)
    all_src_videos += list(src_videos)
    
all_src_videos = list(set(all_src_videos))
print(all_src_videos)

with open('./logs/movies_20240117_aesthetics5.processed_src_videos.txt','w') as fio:
    fio.writelines([
        l + '\n'
        for l in all_src_videos
    ])
