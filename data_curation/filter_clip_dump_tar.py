import os
import tarfile
import json
import megfile
from collections import OrderedDict
import functools
from copy import deepcopy
from data_fetcher import generate_hash_from_paths
from contextlib import contextmanager
from tag_modules.motion_score import get_global_motion_score
from utils import volces_get_worker_cnt_worldsize
from filter_human_videos_and_detect_clip import filter_single_video
WORKER_CNT,WORLDSIZE = volces_get_worker_cnt_worldsize()
TMP_DIR = '/data/users/weisiyuan/tmp'
SCRIPT_PATH = os.path.abspath(__file__)
@contextmanager
def sync_oss_file2local_tmp_dir(remote_file,local_file = None):
    if local_file is None:
        file_type = remote_file.rsplit(".",1)[-1]
        fhash = generate_hash_from_paths(remote_file,SCRIPT_PATH)
        local_file = os.path.join(TMP_DIR,f'{fhash}.{file_type}')
    megfile.smart_copy(remote_file,local_file)
    try:
        yield local_file
    finally:
        os.remove(local_file)

class SmartTarfile():
    
    def __init__(self,path,mode = 'r') -> None:

        self.fio = megfile.smart_open(path,f'{mode}b')
        self.tarfile_obj = tarfile.open(fileobj=self.fio, mode = mode)
        
    def add(self,*args,**kwargs):
        self.tarfile_obj.add(*args, **kwargs)
        
    def addfile(self,*args,**kwargs):
        self.tarfile_obj.addfile(*args, **kwargs)
        
    def close(self):
        self.tarfile_obj.close()
        self.fio.close()


        
class AutoSplitTarWriter:
    def __init__(self, rootdir, split_size,prefix = ''):
        self.rootdir = rootdir
        self.split_size = split_size
        self.file_count = 0
        self.archive_count = 1
        self.prefix = prefix

        megfile.smart_makedirs(rootdir,exist_ok=True)

        self.current_tar_path = self._get_new_archive_path()
        self.tarfile_obj = SmartTarfile(self.current_tar_path,mode = 'w')
        

    def _get_new_archive_path(self):
        return os.path.join(self.rootdir, f"archive{self.prefix}_{self.archive_count}.tar")

    def add_file(self,arcname, filepath = '',file_obj = None):
        if filepath and not os.path.isfile(filepath):
            raise FileNotFoundError(f"File {filepath} does not exist.")

        if self.file_count >= self.split_size:
            self._finish_current_archive()
            self._start_new_archive()

        if file_obj is None:
            self.tarfile_obj.add(filepath, arcname)
        else:
            tar_info=tarfile.TarInfo(arcname)
            tar_info.size = len(file_obj)
            self.tarfile_obj.addfile(tar_info,file_obj)
            
        self.file_count += 1

    def _finish_current_archive(self):
        self.tarfile_obj.close()
        print(f"Finished archive: {self.current_tar_path}")

    def _start_new_archive(self):
        self.archive_count += 1
        self.current_tar_path = self._get_new_archive_path()
        self.tarfile_obj = SmartTarfile(self.current_tar_path,mode = 'w')
        self.file_count = 0

    def finish(self):
        """ Call this method after all files are added to ensure the last tarfile is closed properly """
        self._finish_current_archive()


from data_fetcher import load_json, dump_json
from tqdm import tqdm
import subprocess


def ffmpeg_crop_video(input_file, start_time, duration,output_file = None, output_format='mp4',trg_shape = None):
    # 构建ffmpeg命令用于裁剪视频
    command = [
        'ffmpeg','-y',
        '-v','quiet',
        '-ss', str(start_time),  # 裁剪开始时间
        '-t', str(duration),     # 裁剪持续时间
        '-i', input_file,        # 输入文件
        '-c:v', 'copy',            # 复制编码格式以减少编码时间
        '-an',                   # 禁止音频输出
        # '-s', f'200x200',       # 设置新的分辨率
        
        '-f', output_format,     # 输出格式
                         # 将输出发送到标准输出
    ]
    
    if trg_shape is not None:
        h,w = trg_shape
        command.append('-s')
        command.append(f'{w}x{h}')
    
    if output_file is None:
        command.append(
            "-movflags",
        )
        command.append("frag_keyframe+empty_moov",)#将MOOV atom放在文件的尾部并避免寻找（seeking）
        command.append('pipe:1')
        # 运行ffmpeg命令并捕获输出
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError('ffmpeg command failed: ' + error.decode())
        
        return output  # 返回字节流
    else:
        command.append(output_file)
        subprocess.run(command)
        return output_file
    
# clip_gen
def get_read_tarfile(tar_f):
    return SmartTarfile(tar_f,mode = 'r')


_TAR_OBJ_WRAPPER_CACHE = {}
@contextmanager
def load_tar_member(tarpath_membername):
    file_type = tarpath_membername.rsplit('.',1)[-1]
    tar_f, member_name = tarpath_membername.split(".tar/")
    tar_f = f'{tar_f}.tar'
    
    tar_obj_wrapper = _TAR_OBJ_WRAPPER_CACHE.get(tar_f,get_read_tarfile(tar_f))
    
    member_bytes = tar_obj_wrapper.tarfile_obj.extractfile(member_name).read()

    fhash = generate_hash_from_paths(tarpath_membername,SCRIPT_PATH)
    local_file = os.path.join(TMP_DIR,f'{WORKER_CNT}-{WORLDSIZE}-{fhash}.{file_type}')
    with megfile.smart_open(local_file,'wb') as tmpfio:
        tmpfio.write( member_bytes)

    if tar_f not in _TAR_OBJ_WRAPPER_CACHE:
        for k ,v in _TAR_OBJ_WRAPPER_CACHE.items():
            v.close()
        _TAR_OBJ_WRAPPER_CACHE.clear()
        _TAR_OBJ_WRAPPER_CACHE[tar_f] = tar_obj_wrapper
    try:
        yield local_file
    finally:
        os.remove(local_file)

@contextmanager
def null_file_handler(path):
    yield path

def is_tarmember_path(path):
    return  '.tar' in path and not megfile.smart_exists(path)

@contextmanager
def smart_file_handler(path):

    if is_tarmember_path(path):
        with load_tar_member(path) as ctx:
            yield ctx
    else:
        with sync_oss_file2local_tmp_dir(path) as ctx:
            yield ctx
        

def get_resize_shape(shape,args):
    
    shape = shape[:2]
    min_short_side_size = args.min_short_side_size
    if min(shape) >min_short_side_size:
        
        scale = min_short_side_size /min(shape)
        shape = [ int(i * scale) for i in shape]
    return shape

def main(args):
    
    worker_cnt,worldsize = volces_get_worker_cnt_worldsize()
    print(f'{worker_cnt}/{worldsize}'.center(100,'-'))
    
    min_duration = 1
    max_duration = 10
    max_face_num = 1
    max_text_area_ratio = 0.07
    
    args.min_short_side_size = 720
    seg_over_long_clip_stride = max_duration // 2 # 当clip过长时，默认使用max_duration一半作为stride继续分割
    inpath = args.inpath
    outdir = args.outdir
    tar_split_size = 100
    
    tar_writer = AutoSplitTarWriter(outdir,int(tar_split_size *2),prefix=f'{worker_cnt}_{worldsize}') # considering the video file and meta json file
    
    
    # decide the mode of fetching video files
    filetypes = ['mp4','mkv']
    if not args.from_tar:
        from data_fetcher import fetch_videos
        fetcher = fetch_videos(inpath,filetypes)
    else:
        from data_fetcher import fetch_video_from_tars
        fetcher = fetch_video_from_tars(inpath,filetypes)
    meta = dict()
    
    

    total_clip_count = 0
    for video_idx,video_file in tqdm(enumerate(fetcher)):
    # for video_idx,(video_file, clips_metas) in tqdm(enumerate(video_meta.items())):
        
        job_idx = video_idx  
        if job_idx % worldsize != worker_cnt:
            continue
        # print(f'{job_idx}-{video_idx}')
        video_fhash = generate_hash_from_paths(video_file,SCRIPT_PATH)
        # process the video files in video tar files & vanilla video files
        with smart_file_handler(video_file) as local_video_file:
            clips_metas = filter_single_video(local_video_file)
            if clips_metas is None:
                #  no valid clips containing human faces
                continue
            video_fname = os.path.basename(video_file).rsplit('.',1)[0]
            frame_offset_list = clips_metas['clips']
            clip_meta_list = clips_metas['clip_meta_list']
            video_meta = {
                k:clips_metas[k] for k in ['fps','total_frames','ori_shape']
            }
            ori_shape = video_meta['ori_shape']
            def clip_gen():
            
                # the clip meta is from the processing of last data curation step
                for _,(frame_offsets, clip_meta) in enumerate(zip(frame_offset_list, clip_meta_list)):
                    
                    # skip videos with multi face
                    if clip_meta['n_face'] > max_face_num:
                        pass
                    # skip videos with too much text areas
                    elif clip_meta['text_area_ratio'] > max_text_area_ratio:
                        pass
                    else:
                        frame_st,frame_ed = frame_offsets
                        duration = (frame_ed - frame_st) / clips_metas['fps']
                        
                        if duration < min_duration:
                            # drop the over-short clip
                            continue
                        elif duration > max_duration:
                            frame_stride= int(clips_metas['fps'] * seg_over_long_clip_stride)
                            for frame_idx in range(frame_st,frame_ed+1,frame_stride):
                                
                                
                                start_time = frame_idx/clips_metas['fps']
                                end_time = min(frame_idx +frame_stride,frame_ed) / clips_metas['fps']
                                
                                item_meta = dict()
                                item_meta.update(
                                    src_video = video_file,
                                    src_video_info = video_meta,
                                    is_sec_seg = True,
                                    src_clip_meta = clip_meta, 
                                    start_time = start_time,
                                    duration = end_time - start_time,    
                                )
                                if item_meta['duration'] < min_duration:
                                    continue
                                yield local_video_file, item_meta
                            
                        else:
                            start_time = frame_st/clips_metas['fps']
                            end_time = frame_ed / clips_metas['fps']
                                
                            item_meta = dict()
                            item_meta.update(
                                src_video = video_file,
                                src_video_info = video_meta,
                                is_sec_seg = False,
                                src_clip_meta = clip_meta, 
                                start_time = start_time,
                                duration = end_time - start_time,    
                            )
                            yield local_video_file, item_meta
                    
            for clip_idx ,(local_video_file, item_meta) in enumerate(clip_gen()):
                
                
                member_name = f'{video_idx}-{clip_idx}-{video_fname}'
                tmp_clip_file = os.path.join(
                    TMP_DIR,f'{member_name}.mp4'
                )
                tmp_meta_file = os.path.join(
                    TMP_DIR,f'{member_name}.json'
                )
                
                trg_shape = get_resize_shape(ori_shape,args)
                
                _ = ffmpeg_crop_video(
                    local_video_file,item_meta['start_time'],item_meta['duration'],output_file = tmp_clip_file,trg_shape = trg_shape
                ) #TODO: resize 
                
                # NOTE: add extra tags
                # motion_score = get_global_motion_score(tmp_clip_file)
                
                dump_json(item_meta, tmp_meta_file)
                
                tar_writer.add_file(
                    f'{member_name}.mp4',filepath = tmp_clip_file
                )
                tar_writer.add_file(
                    f'{member_name}.json', filepath = tmp_meta_file
                )
                os.remove(tmp_clip_file)
                os.remove(tmp_meta_file)

                
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=' from video meta json to the video file and meta file in the clip level')
    parser.add_argument(
        "-i",
        "--inpath",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=int,
        default= 0,
        help=' the resume count in the tar file level'
    )
    parser.add_argument(
        "--from_tar",
        action='store_true',
        default=False,
    )
    
    
    args = parser.parse_args()
    
    main(args)