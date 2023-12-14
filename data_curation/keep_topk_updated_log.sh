#!/bin/bash

# 设置文件夹路径
log_folder="logs/hdvila_100m"

# 要保留的最新日志文件数量，即Top K
top_k=18

# 进入日志文件所在的目录
cd "$log_folder"

# 找到并删除更新时间在Top K之外的所有日志文件
find . -maxdepth 1 -type f -name '*.txt' | sort -r | tail -n +18 | xargs rm

# -maxdepth 1 确保只列出当前目录中的文件
# -type f 表示查找文件，不包括目录
# -name '*.log' 查找所有扩展名为.log的文件
# sort -r 对结果进行逆序排序（最新的文件会排在前面）
# tail -n +X 从第X行开始输出，即跳过前面的文件
# xargs rm 将tail的输出传给rm命令执行删除操作
