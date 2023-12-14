## overview

目前分为两步: 
1. 粗筛长视频数据，生成对应的meta
2. 分片生成tar包训练数据

## steps



1. 粗筛长视频数据，生成对应的meta, 包括: face num筛选,pySceneDetecion + ffmpeg关键帧修正　＋　ocr检测

```
WORLD＝... RANK=... # 建议bash传入相关环境变量切片并行处理或使用火山云replica worker设置，脚本会自动读取相关ddp变量切片处理
DEBUG=1 # debug mode会dump video clips在本地
python3 filter_human_videos_and_detect_clip.py 
    -i {长视频文件夹地址，支持通配符匹配，支持mp4格式与mp4 tar格式,eg:s3://weisiyuan-sh/datasets/CelebV-Text/,s3://weisiyuan-sh/datasets/xiaohongshu/*/vedio/} \
    -n {数据集名称，会将长视频粗筛后的meta放置在 f's3://weisiyuan-sh/datasets/{args.name}/worker{worke_cnt}.json'}
    --from_tar {是否从tar包中读取视频}

```
例如：

a. 更改volc yaml中的相关参数，设置replica workers，在脚本中切片并行处理任务
```
Framework: "PyTorchDDP"
TaskRoleSpecs:
    - RoleName: "worker"
      RoleReplicas: 18
      Flavor: "ml.gni2.3xlarge" # A10
```
b. 处理源数据长视频
```

＃处理celebV数据
$PYTHON_CMD filter_human_videos_and_detect_clip.py -i s3://weisiyuan-sh/datasets/CelebV-Text/ -n CelebV --from_tar > logs/celebv/${HOSTNAME}.txt  2>&1 &
＃处理hdvila数据
CONTENT_DET_TH=10 $PYTHON_CMD filter_human_videos_and_detect_clip.py -i s3://nlp-data-map/video/hd-vila-100/hdvila_100m/download_videos -n hdvila_100m_20231209 > logs/hdvila_100m/${HOSTNAME}.txt  2>&1 &
＃处理小红书数据
$PYTHON_CMD filter_human_videos_and_detect_clip.py -i "s3://weisiyuan-sh/datasets/xiaohongshu/*/vedio/" -n xiaohongshu_20231209 > logs/xiaohongshu/${HOSTNAME}.txt  2>&1 &

```
注: 建议重定向log,火山云平台replica worker的日志输出有问题会跟不上实际输出
2. 根据长视频粗筛选后的meta信息处理为tar包