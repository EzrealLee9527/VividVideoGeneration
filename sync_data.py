from megfile import smart_open, smart_exists, smart_sync, smart_remove, smart_glob
import os

hs_tos_dir = 's3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/wds/*'
local_save_dir = '/data/wds/'
sp_oss_dir = 's3://ljj/Datasets/Videos/processed/'
os.system(f'cp ~/.aws/credentials_hs ~/.aws/credentials')
for per_hs_tos_dir in smart_glob(hs_tos_dir):
    local_save_path = os.path.join(local_save_dir, per_hs_tos_dir.split('/')[-1])
    sp_remote_save_path = sp_oss_dir + per_hs_tos_dir.split('/')[-1]
    # print('per_hs_tos_dir', per_hs_tos_dir)
    # print('local_save_path', local_save_path)
    # print('sp_remote_save_path', sp_remote_save_path)
    command = f'aws --endpoint-url=https://tos-s3-cn-shanghai.volces.com s3 sync {per_hs_tos_dir} {local_save_path}'
    print(command)
    os.system(command)

    command = f'cp ~/.aws/credentials_sp ~/.aws/credentials'
    print(command)
    os.system(command)
    
    command = f'aws --endpoint-url=http://oss.i.shaipower.com:80 s3 sync {local_save_path} {sp_remote_save_path}'
    print(command)
    os.system(command)
    
    command = f'cp ~/.aws/credentials_hs ~/.aws/credentials'
    print(command)
    os.system(command)
    smart_remove(local_save_path)
    print('\n')