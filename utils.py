'''
原作者地址：https://github.com/anothermartz/Easy-Wav2Lip
眠NEON-优化版，B站地址：https://space.bilibili.com/386198029 \Wechat:dzemcn
'''

import configparser
import os

def update_config(file_path, section, key, new_value):
    config = configparser.ConfigParser()
    config.read(file_path)
    config.set(section, key, new_value)
    with open(file_path, 'w') as configfile:
        config.write(configfile)      

def find_media_file(directory, extensions):
    for file_name in os.listdir(directory):
        if any(file_name.lower().endswith(ext) for ext in extensions):
            return os.path.join(directory, file_name)
    return None  

def find_image_or_video(in_folder):
    allowed_extensions = ['.jpeg', '.jpg', '.png', '.bmp', '.mp4', '.avi', '.mov', '.mkv']
    return find_media_file(in_folder, allowed_extensions)

def find_audio(in_folder):
    allowed_extensions = ['.mp3', '.wav', '.aac']
    return find_media_file(in_folder, allowed_extensions)