import tarfile
import os
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

# 替换成你的tar文件路径
for tar_file_path in megfile.smart_glob(
    os.path.join("s3://weisiyuan-sh/datasets/CelebV_webdataset_20231211/",'*.tar')
):

    if has_duplicate_members(tar_file_path):
        print('There are duplicate members in the tar file.')
    else:
        print('No duplicate members found in the tar file.')
