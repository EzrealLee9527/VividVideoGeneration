import tarfile
# from split_file_reader import SplitFileReader
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import json
import megfile
from joblib import Memory
cachedir = './my_cache'  # 这是缓存将被保存的本地目录
__MEMORY = Memory(cachedir, verbose=0)


def dump_json(obj,path):
    with megfile.smart_open(path,'w') as fio:
        fio.write(json.dumps(obj,indent=2))
        
def load_json(path):
    with megfile.smart_open(path,'r') as fio:
        return json.load(fio)
# tarfile.TarFile.copybufsize = 1024 * 1024 * 2

DEFAULT_TEMPFILE_DIR='/data/users/weisiyuan/tmp/'
FFHQ_ROOT_DIR = '/data/users/weisiyuan/dataset'
# FFHQ_ROOT_DIR = 's3://weisiyuan-sh/datasets'



# NOTE: extremly slow
# def fetch_ffhq_data(verbose = False,parallel = 0,use_pigz = True):
#     filepaths = [
        
#         str(
#             os.path.join(FFHQ_ROOT_DIR,"FFHQ",f"FFHQ.tar.gz.{i:04}")
#         )
#         for i in range(10)
#     ]
#     print(filepaths)

#     if not use_pigz:
#         sfr = SplitFileReader(filepaths,stream_only = True)
#     else:
#         sfr = open(os.path.join(FFHQ_ROOT_DIR,"FFHQ",f"FFHQ.tar.gz"),'rb')
#     with sfr:
#         if use_pigz:
#             import subprocess
#             process = subprocess.Popen(
#                 ['pigz', '-dc'],
#                 stdin=sfr,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE)
#             fio = process.stdout
#             mode = 'r|'
#         else:
#             fio = sfr
#             mode = 'r|gz'
#         print(f"reading mode : {mode}")
#         with tarfile.open(fileobj=fio, mode=mode) as tf:
#             for member in tf:
#                 if member is None:
#                     break
#                 elif member.isfile():
#                         if verbose:
#                             print("File found")
#                        # TODO: slow while extrating from s3
#                         img = Image.open(tf.extractfile(member.name)).convert("RGB")
#                         print("Image Readed")
#                         yield np.array(img),member.name

import pickle
import megfile
def load_pickle(path):
    with megfile.smart_open(path,'rb') as fio:
        return pickle.loads(fio.read())

def fetch_ffhq_data_from_pkl(worldsize = 1,worker_cnt = 0):
    from megfile import smart_glob
    
    pkl_files = smart_glob(
        os.path.join('s3://weisiyuan-sh/datasets/20231004_FFHQ_3dMM','*.pkl')
    )
    
    cnt = 0
    for file in pkl_files:
        
        if cnt % worldsize == worker_cnt:
            yield load_pickle(file)['img'],os.path.basename(file)
        else:
            pass
        cnt +=1
        
def fetch_ffhq_data(worldsize = 1,worker_cnt = 0):
    from glob import glob
    rootdir = '/data/users/weisiyuan/dataset/FFHQ/images1024x1024'
    
    cnt = 0
    for file in glob(os.path.join(rootdir,'*.png')):
        
        if cnt % worldsize == worker_cnt:
            yield np.array(Image.open(file).convert("RGB")),os.path.basename(file)
        else:
            pass
        cnt +=1
        
        
def fetch_celebA_data(worldsize = 1,worker_cnt = 0):
    from glob import glob
    import zipfile
    from megfile import smart_open
    rootdir = 's3://weisiyuan-sh/datasets/CelebA/img_align_celeba.zip'
    identity_file = 's3://weisiyuan-sh/datasets/CelebA/identity_CelebA.txt'
    attr_file = 's3://weisiyuan-sh/datasets/CelebA/list_attr_celeba.txt'
    
    with smart_open(identity_file,'r') as rf:
        identity_infos = rf.readlines()
        identity_infos = [
            line.strip('\n').split(" ")
            for line in identity_infos
        ]
    identity_infos = {
        fname : id_num
        for fname,id_num in identity_infos
    }
        
    with smart_open(attr_file,'r') as rf:
        attr_infos = rf.readlines()
        attr_infos = [
            line.strip('\n').split(" ")
            for line in identity_infos
        ]
    
    
    
    cnt = 0
    with smart_open(rootdir,'rb') as rf:
        with zipfile.ZipFile(rf) as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                else:
                    with zf.open(member) as zf_member:
                        img = np.array(Image.open(zf_member).convert("RGB"))
                        fname = member.rsplit("/")[-1]
                        id_num = identity_infos[fname]

                        if cnt % worldsize == worker_cnt:
                            yield img,f'{id_num}/{fname}'
                        else:
                            pass
                        
                        cnt +=1
    


def fetch_IMDB_data():
    from glob import glob
    path = '/data/users/weisiyuan/dataset/IMDB/imdb_tar'
    tar_files = glob(os.path.join(path,'*.tar'))
    
    def parson_pid(p):
        
        fname = os.path.basename(p)
        pid = fname.split("_")[0]
        other_info = fname.lstrip(pid)
        return pid,other_info
    
    tar_files = [
        '/data/users/weisiyuan/dataset/IMDB/imdb_tar/imdb_all.tar'
    ]
    for tar_f in tar_files:
        print(tar_f)
        with tarfile.open(tar_f,) as tf:
            
            members = [mem for mem in tf.getmembers() if mem.isfile()]
            file_list = sorted(members, key=lambda x: parson_pid(x.name)[0])
            print(f'Members  : {len(file_list)}')
            same_pid_samples = []
            last_pid = None
            for member in file_list:
                
                if not member.isfile():
                    continue
                if member.name.endswith(".jpg") or member.name.endswith(".jpeg")  or member.name.endswith(".png"):
                    try:
                        img = Image.open(tf.extractfile(member.name)).convert("RGB")
                    except:
                        # broken image files
                        continue
                    if img.height * img.width < 224 * 224:
                        continue
                    
                    pid,other_info = parson_pid(member.name)
                    outfname = f'{pid}/{other_info}'
                    
                    if last_pid is None or last_pid == pid:
                        pass
                    else:
                        
                        yield last_pid,same_pid_samples
                        same_pid_samples = []
                    last_pid = pid
                    same_pid_samples.append((np.array(img), outfname))
                    
                    
                    
def fetch_self_collected_data():
    from glob import glob
    from collections import defaultdict
    from megfile import smart_open    
    tar_files = [
        # 's3://weisyuan-sh/datasets/neo/neo_fq_face.tar/neo_hq_face.tar',
        '/data/users/weisiyuan/dataset/neo_hq_face.tar'
    ]
    for tar_f in tar_files:
        print(tar_f)

        with tarfile.open(tar_f,) as tf:
            
            members = [mem for mem in tf.getmembers() if mem.isfile()]
            
            date2members = defaultdict(list)
            for mem in members:
                date = mem.name.split("/")[-2]
                date2members[date].append(mem)
            date2members = {
                
                k:sorted(v, key = lambda x:x.name)
                for k,v in date2members.items()
            }
            for date,members in date2members.items():
                
                date_samples = [date,[]]
                for member in members:
                    if not member.isfile():
                        continue
                    if member.name.endswith(".jpg") or member.name.endswith(".jpeg")  or member.name.endswith(".png"):
                        img = np.array(Image.open(tf.extractfile(member.name)).convert("RGB"))
                    date_samples[1].append((img,member.name))
                yield date_samples
                
import json
def read_jsonl_file(path):
    
    reader = megfile.smart_open(path, 'r')
    
    lines = reader.readlines()
    reader.close()
    return [
        json.loads(l.strip('\n')) for l in lines
    ]

from collections import defaultdict
def read_remote_img(p):
    with megfile.smart_open(p, 'rb') as rf:
        return Image.open(rf).convert("RGB")
    
    
def fetch_miaoji_duitang():
    metas = read_jsonl_file('jmh_miaoji_duitang_20231030.jsonl')
    pid2files = defaultdict(list)
    for m in metas:
        for k,v in m.items():
            pid2files[k].append(v)
    print(f'person num : {len(pid2files)}')
    for pid, files in pid2files.items():
        pid_samples = []
        for f in files:
            try:
                img = np.array(read_remote_img(f))
            except:
                continue
            outf = f'{pid}/{os.path.basename(f)}'
            pid_samples.append([img,outf])
        
        if pid_samples:
            yield pid,pid_samples
            
            


         
def fetch_videos(rootdir,filetypes = ['mp4']):
    
    video_files = []
    
    cache_glob_func = __MEMORY.cache(megfile.smart_glob)
    

    for ftype in filetypes:
        video_files += cache_glob_func(
            os.path.join(rootdir,f'*.{ftype}')
        )
    return video_files
    
# default celebV
def fetch_video_from_tars(rootdir = 's3://weisiyuan-sh/datasets/CelebV-Text/', filetypes = ['mp4']):
    #  return video path and bytes
    from glob import glob
    from collections import defaultdict
    from megfile import smart_open    
    tar_files = megfile.smart_glob(
        os.path.join(rootdir,'*.tar')
    )
    for tar_f in tar_files:
        
        
        with megfile.smart_open(tar_f,'rb') as fio:

            with tarfile.open(fileobj=fio,mode = 'r') as tf:
                
                members = [mem for mem in tf.getmembers() if mem.isfile()]
                

                for member in members:
                    if not member.isfile():
                        continue
                    
                    flag=False
                    for ftype in filetypes:
                        if member.name.endswith(ftype):
                            flag = True
                            break
                    
                    if not flag:
                        continue
                    
                    yield os.path.join(tar_f, member.name), tf.extractfile(member.name).read()

import hashlib
import os

def generate_hash_from_paths(file_path,current_script_path = os.path.abspath(__file__)):
    # Get the current executing script's directory
    
    # Combine both paths
    combined_path = file_path + current_script_path
    
    # Create a SHA256 hash object
    hash_object = hashlib.sha224()
    
    # Update the hash object with the combined path encoded to bytes
    hash_object.update(combined_path.encode('utf-8'))
    
    # Get the hexadecimal representation of the hash
    hex_digest = hash_object.hexdigest()
    
    return hex_digest




        
if __name__ == "__main__":
    
    
    video_files = fetch_videos(
        's3://nlp-data-map/video/hd-vila-100/hdvila_100m/download_videos',filetypes=['mp4']
    )
    print(len(video_files))
            