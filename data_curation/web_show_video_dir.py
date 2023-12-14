import streamlit as st
import os
import argparse


parser = argparse.ArgumentParser(
        description="none"
    )
parser.add_argument("--dir", type=str, default=None)
args = parser.parse_args()
# 设置页面标题
st.title('视频文件播放器')

# 读取当前目录下的所有mp4文件
current_dir = args.dir
video_files = [f for f in os.listdir(current_dir) if f.endswith('.mp4')]

# 在侧边栏显示文件列表供用户选择
st.sidebar.header('视频列表')
selected_video = st.sidebar.selectbox('选择一个视频文件', video_files)

# 如果选定了某个视频文件，则在主区域中显示和播放该视频
if selected_video is not None:
    st.video(os.path.join(current_dir, selected_video))
