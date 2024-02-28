from megfile import smart_open, smart_exists, smart_sync, smart_remove, smart_glob
import os

src_tos_dir = 's3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/wds/*'
local_save_dir = '/data/wds/'
target_oss_dir = 's3://public-datasets/Datasets/Videos/processed/'
for per_src_tos_dir in smart_glob(src_tos_dir):
    local_save_path = os.path.join(local_save_dir, per_src_tos_dir.split('/')[-1])
    sp_remote_save_path = target_oss_dir + per_src_tos_dir.split('/')[-1]

    command = f'aws --profile hs --endpoint-url=https://tos-s3-cn-shanghai.volces.com s3 sync {per_src_tos_dir} {local_save_path}'
    print(command)
    os.system(command)
    
    command = f'aws --endpoint-url=http://oss.i.brainpp.cn:80 s3 sync {local_save_path} {sp_remote_save_path}'
    print(command)
    os.system(command)

    smart_remove(local_save_path)
    print('\n')