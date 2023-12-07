# Vivid Video Generation

# News
**2023.12.07**
- 添加了生成MetaInfo的Pipeline,包括生成Caption(CoCa,VideoBLIP), 计算MotionScore(Farneback法计算稠密光流), 计算Text-Image相似度(CoCa/VideoBLIP), 计算美学得分(CLIP+MLP), 计算文字覆盖率(Craft)等流程; 
**2023.12.05**
- 环境配置文件上传
- 把各个步骤的脚本上传
- 在pexels数据集上测试过:  [worklog](https://wiki.megvii-inc.com/pages/viewpage.action?pageId=519812610)


# 环境配置
- conda配置文件：svd_env.yaml
- 文字检测会用到的Craft存在环境不适配问题, 需要修改一些代码:
  - [ImportError: cannot import name 'model_urls' from 'torchvision.models.vgg'](https://github.com/clovaai/CRAFT-pytorch/issues/191)
  - [ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (19,) + inhomogeneous part.](https://blog.csdn.net/m0_53127772/article/details/132492224)
- 其余问题应该不大, 爬坑记录:  [worklog](https://wiki.megvii-inc.com/pages/viewpage.action?pageId=518190042)
- 模型文件：目前都存放在火山云的tos上
  - insightface: s3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/SVD/DataCuration/models/.insightface
  - Caption/Aesthetic/Similarity:  s3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/SVD/DataCuration/models/huggingface/hub/
  - TextDetection:  s3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/SVD/DataCuration/models/huggingface/hub/craft

# 各个脚本调用
### 分镜检测
目前仅支持最简单的使用方式,还未考虑fade-in/fade-out的情况.

`python3 scripts/cur_detection.py`

### 调用InsightFace检测人脸
目前仅支持onnx的CPU版本eval, gpu环境有冲突, 有必要的话可以专门配置一下insightface的gpu环境:

`python3 scripts/insightface_detection.py`

### MotionScore
- #### OpenCV Farneback:
  `python3 scripts/motion_score.py`

### Caption
- #### CoCa
  `python3 scripts/caption_coca.py`
- #### VideoBLIP
  `python3 scripts/caption_videoblip.py`
- #### LLM：
  coming soon

### AestheticScore
`python3 scripts/aesthetics_score.py`

### Text-Image Similarity
`python3 scripts/text_image_similarities.py`

### 检测文字
`python3 scripts/text_detection.py`