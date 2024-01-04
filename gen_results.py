import os
from glob import glob
from moviepy.editor import VideoFileClip

sample_base_dir = './samples/*'
for sample_res_dir in glob(sample_base_dir):
    input_path = os.path.join(sample_res_dir,'videos/2_2.mp4')
    print(input_path)    
    fname = sample_res_dir.split('/')[-1]
    output_path = f'results_240103_simple/{fname}.gif'
    print(output_path)

    try:
        command = f'ffmpeg -i {input_path} -filter:v crop=528:528:1056:0 -c:a copy {output_path}'
        os.system(command)
        # clip = VideoFileClip(input_path)

        # # 可以选择裁剪视频到你想要的某段
        # # clip = clip.subclip(start_time, end_time)

        # clip.write_gif(output_path)
    except:
        continue
