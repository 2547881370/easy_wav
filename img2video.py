'''
原作者地址：https://github.com/anothermartz/Easy-Wav2Lip
眠NEON-优化版，B站地址：https://space.bilibili.com/386198029 \Wechat:dzemcn
'''

import cv2
import subprocess
import os

ffmpeg_path = 'ffmpeg.exe'
ffprobe_path = 'ffprobe.exe'

def get_audio_duration(audio_file_path):
    cmd = f'"{ffprobe_path}" -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{audio_file_path}"'
    audio_duration = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
    return float(audio_duration)

def create_video_from_image(image_path, audio_path, output_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    if height % 2 != 0:
        height -= 1  
    if width % 2 != 0:
        width -= 1 
    image = cv2.resize(image, (width, height))
    temp_image_path = 'temp_image.png'
    cv2.imwrite(temp_image_path, image)
    audio_duration = get_audio_duration(audio_path)
    cmd = (
        f'"{ffmpeg_path}" -loop 1 -framerate 25 -i "{temp_image_path}" '
        f'-i "{audio_path}" -c:v mpeg4 -t {audio_duration} '
        f'-pix_fmt yuv420p -c:a aac -shortest -strict experimental -y "{output_path}"'
    )
    subprocess.call(cmd, shell=True)
    os.remove(temp_image_path)


def check_and_convert_image_to_video(in_folder, media_path):
    allowed_image_extensions = ['.jpeg', '.jpg', '.png', '.bmp']
    allowed_video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    file_name, file_extension = os.path.splitext(os.path.basename(media_path))  # 获取文件的基础名称和扩展名
    file_extension = file_extension.lower()
    
    if file_extension in allowed_video_extensions:
        if os.path.exists(media_path):
            print("已确认文件为视频格式，跳过图片转换视频过程。")
            return media_path
        else:
            raise ValueError("视频文件不存在")
    
    elif file_extension in allowed_image_extensions:
        audio_file = next((f for f in os.listdir(in_folder) if f.endswith(('.mp3', '.wav', '.aac'))), None)
        if audio_file is None:
            raise ValueError("未找到音频文件")
        audio_path = os.path.join(in_folder, audio_file)
        output_video_path = os.path.join(in_folder, f"{file_name}.mp4")
        create_video_from_image(media_path, audio_path, output_video_path)
        print("图片已转换为视频。")
        return output_video_path
    
    else:
        raise ValueError("文件既不是支持的视频格式也不是图片格式。")