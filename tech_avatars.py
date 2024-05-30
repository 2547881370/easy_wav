import datetime
import random
import socket
import subprocess
import asyncio
import json
import pygame
import requests
import torch
import numpy as np
import math
import os
import pickle
import cv2
import websockets
import audio
from batch_face import RetinaFace
from functools import partial
from tqdm import tqdm
from easy_functions import load_model
from moviepy.editor import VideoFileClip
import threading
import time
import vlc
import queue
from flask import Flask, request, jsonify
from enhance import upscale
from enhance import load_sr
from PIL import Image

# from video_player import VideoPlayer


# 加载基础环境变量
class base_tech_avatars_init:
    def __init__(self):
            self.device='cuda'
            self.ffmpeg_path = 'ffmpeg.exe'
            self.mel_step_size = 16
            self.model = None
            self.detector = None
            self.detector_model = None

            with open(os.path.join('checkpoints','predictor.pkl'), 'rb') as f:
                self.predictor = pickle.load(f)

            with open(os.path.join('checkpoints','mouth_detector.pkl'), 'rb') as f:
                self.mouth_detector = pickle.load(f)
                
            self.do_load('G:\\BaiduNetdiskDownload\\Easy-Wav2Lip-mian-0229\\checkpoints\\Wav2Lip.pth')
                
    def do_load(self,checkpoint_path):
            self.model = load_model(checkpoint_path)
            self.detector = RetinaFace(gpu_id=0, model_path="checkpoints/mobilenet.pth", network="mobilenet")
            self.detector_model = self.detector.model

new_base_tech_avatars_init = base_tech_avatars_init()

# 视频预处理
class VideoPreprocessing:
    def __init__(self, video_path):
        self.video_path = video_path

    def extract_faces(self):
          video_stream = cv2.VideoCapture(self.video_path)
          # 拿到fps
          fps = video_stream.get(cv2.CAP_PROP_FPS)

          full_frames = []
          while 1:
              still_reading, frame = video_stream.read()
              if not still_reading:
                  video_stream.release()
                  break
              y1, y2, x1, x2 = [0, -1, 0, -1]
              if x2 == -1: x2 = frame.shape[1]
              if y2 == -1: y2 = frame.shape[0]
              frame = frame[y1:y2, x1:x2]
              full_frames.append(frame)
        
          # 人脸数据
          results_array = self.face_detect(full_frames)
          
          timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
          os.makedirs(f'./cache/{timestamp}/', exist_ok=True)
          
          # 保存人脸数据
          for idx, face_data in enumerate(results_array):
            file_path = f'./cache/{timestamp}/face_{idx}.npy'
            np.save(file_path, face_data)
            
          # 保存图片帧
          for idx, frame_data in enumerate(full_frames):
            file_path = f'./cache/{timestamp}/frame_{idx}.npy'
            np.save(file_path, frame_data)

          return fps
              
    def face_detect(self,images):
            results = []
            pady1, pady2, padx1, padx2 = [0, 10, 20, 0]
            from tqdm import tqdm
            tqdm = partial(tqdm, position=0, leave=True)

            for image, rect in tqdm(zip(images, self.face_rect(images)), total=len(images), desc="detecting face in every frame", ncols=100):
                if rect is None:
                    cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
                    raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

                y1 = max(0, rect[1] - pady1)
                y2 = min(image.shape[0], rect[3] + pady2)
                x1 = max(0, rect[0] - padx1)
                x2 = min(image.shape[1], rect[2] + padx2)

                results.append([x1, y1, x2, y2])

            boxes = np.array(results)
            boxes = self.get_smoothened_boxes(boxes, T=5)
            results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

            # 将 results 转换为 numpy 数组
            results_array = np.array(results, dtype=object)

            return results_array
              
    def face_rect(self,images):
        face_batch_size = 8
        num_batches = math.ceil(len(images) / face_batch_size)
        prev_ret = None
        for i in range(num_batches):
            batch = images[i * face_batch_size: (i + 1) * face_batch_size]
            # return faces list of all images
            all_faces = new_base_tech_avatars_init.detector(batch)  
            for faces in all_faces:
                if faces:
                    box, landmarks, score = faces[0]
                    prev_ret = tuple(map(int, box))
                yield prev_ret
                
    def get_smoothened_boxes(self,boxes, T):
            for i in range(len(boxes)):
                if i + T > len(boxes):
                    window = boxes[len(boxes) - T:]
                else:
                    window = boxes[i : i + T]
                boxes[i] = np.mean(window, axis=0)
            return boxes
 
 
# newVideoPreprocessing = VideoPreprocessing('./temp/2.mp4')
# fps = newVideoPreprocessing.extract_faces()

# 数字人处理进度维护
class VideoPreprocessor:
    def __init__(self, dirName):
        self.dirName = dirName
        self.frames = 0
        self.faces = 0
        self.current_index = 0
        self.frame_first = None
        self.direction = 1  # 定义方向变量，初始值为1，表示正向
        self.load_frames()
        self.load_faces()
        # 缓存池最大容量
        self.max_cache_size= 6000
        # self.frames // 10
        if(self.frames < 2000) :
            self.max_cache_size = self.frames
        else :
             self.max_cache_size = self.frames // 10
        self.cache_pool = {
            # 人脸数据
            'faces_to_use' : [],
            # 视频帧
            'frames_to_use' : []
        }
        self.refill_thread = threading.Thread(target=self.random_refill_cache)
        self.refill_thread.start()
        
        self.current_index_list = []
    
    # 正序，反向 缓存到缓存池
    def refill_cache(self):
        while True:
            time.sleep(0.01)
            while len(self.cache_pool['faces_to_use']) < self.max_cache_size:
                for _ in range(self.max_cache_size - len(self.cache_pool['faces_to_use'])):
                    # 判断索引是否超出范围
                    if self.current_index >= self.frames:
                        self.current_index = self.frames - 1
                        self.direction = -1  # 当索引超出范围时，改变方向为逆向
                    elif self.current_index < 0:
                        self.current_index = 0
                        self.direction = 1  # 当索引超出范围时，改变方向为正向
                        
                    # 读取人脸数据
                    loaded_face_results_array = np.load(f'./cache/{self.dirName}/face_{self.current_index}.npy', allow_pickle=True)
                    # 读取视频帧数据
                    loaded_frame_results_array = np.load(f'./cache/{self.dirName}/frame_{self.current_index}.npy')
                    self.cache_pool['faces_to_use'].append(loaded_face_results_array)
                    self.cache_pool['frames_to_use'].append(loaded_frame_results_array)
                    
                    # 根据当前方向更新索引
                    self.current_index += self.direction
    # 随机读取 缓存到缓存池
    def random_refill_cache(self):
        while True:
            time.sleep(0.01)
            while len(self.cache_pool['faces_to_use']) < self.max_cache_size:
                num_frames = self.max_cache_size - len(self.cache_pool['faces_to_use'])
                self.direction *= -1
                reverse_frames = random.randint(num_frames // 5, num_frames)  # 随机倒放的帧数
                
                for _ in range(num_frames):
                    self.current_index_list.append(self.current_index)
                    # 读取人脸数据
                    loaded_face_results_array = np.load(f'./cache/{self.dirName}/face_{self.current_index}.npy', allow_pickle=True)
                    # 读取视频帧数据
                    loaded_frame_results_array = np.load(f'./cache/{self.dirName}/frame_{self.current_index}.npy')
                    self.cache_pool['faces_to_use'].append(loaded_face_results_array)
                    self.cache_pool['frames_to_use'].append(loaded_frame_results_array)
                    
                    if( _ == reverse_frames):
                        self.direction *= -1
                        
                    if self.current_index >= self.frames - 1 and self.direction == 1:
                        self.direction = -1
                    elif self.current_index <= 0 and self.direction == -1:
                        self.direction = 1
                    # 根据当前方向更新索引
                    self.current_index += self.direction
            

    def load_frames(self):
        # 列出指定文件夹中以'frame_'为前缀的.npy文件数量
        frame_files = [f for f in os.listdir(f'./cache/{self.dirName}/') if f.startswith('frame_')]
        self.frames = len(frame_files)
        self.frame_first = np.load(f'./cache/{self.dirName}/frame_0.npy')

    def load_faces(self):
        # 列出指定文件夹中以'face_'为前缀的.npy文件数量
        face_files = [f for f in os.listdir(f'./cache/{self.dirName}/') if f.startswith('face_')]
        self.faces = len(face_files)
    
    # 读取缓存池中的人脸数据和视频帧数据
    def get_next_frames_and_faces(self, num_frames):
        faces_to_use = []
        frames_to_use = []
        for _ in range(num_frames):
            faces_to_use.append(self.cache_pool['faces_to_use'].pop(0))
            frames_to_use.append(self.cache_pool['frames_to_use'].pop(0))
        return faces_to_use, frames_to_use
    
    # 随机顺序
    def _get_next_frames_and_faces(self, num_frames):
        faces_to_use = []
        frames_to_use = []
        
        self.direction *= -1
        reverse_frames = random.randint(num_frames / 5, num_frames)  # 随机倒放的帧数
        
        for _ in range(num_frames):
            # 读取人脸数据
            loaded_face_results_array = np.load(f'./cache/{self.dirName}/face_{self.current_index}.npy', allow_pickle=True)
            # 读取视频帧数据
            loaded_frame_results_array = np.load(f'./cache/{self.dirName}/frame_{self.current_index}.npy')
            faces_to_use.append(loaded_face_results_array)
            frames_to_use.append(loaded_frame_results_array)
            
            if( _ == reverse_frames):
                self.direction *= -1
                
            if self.current_index >= self.frames - 1 and self.direction == 1:
                self.direction = -1
            elif self.current_index <= 0 and self.direction == -1:
                self.direction = 1
            self.current_index += self.direction
            
        return faces_to_use, frames_to_use


# 数字人合成
class DigitalHumanSynthesizer:
    def __init__(self, video_preprocessor, batch_size=1,quality = 'Fast'):
        self.video_preprocessor = video_preprocessor
        self.batch_size = batch_size
        self.kernel = self.last_mask = self.x = self.y = self.w = self.h = None
        # 数字人处理速度 'Fast' 'Enhanced'
        self.quality = quality
        
    def create_tracked_mask(self,img, original_img): 
        # Convert color space from BGR to RGB if necessary 
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img) 
        cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB, original_img) 
        
        # Detect face 
        faces = new_base_tech_avatars_init.mouth_detector(img) 
        if len(faces) == 0: 
            if self.last_mask is not None: 
                self.last_mask = cv2.resize(self.last_mask, (img.shape[1], img.shape[0]))
                mask = self.last_mask  # use the last successful mask 
            else: 
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img) 
                return img, None 
        else: 
            face = faces[0] 
            shape = new_base_tech_avatars_init.predictor(img, face) 
        
            # Get points for mouth 
            mouth_points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)]) 

            # Calculate bounding box dimensions
            self.x, self.y, self.w, self.h = cv2.boundingRect(mouth_points)

            # Set kernel size as a fraction of bounding box size
            kernel_size = int(max(self.w, self.h) * 2.5)
            #if kernel_size % 2 == 0:  # Ensure kernel size is odd
                #kernel_size += 1

            # Create kernel
            self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Create binary mask for mouth 
            mask = np.zeros(img.shape[:2], dtype=np.uint8) 
            cv2.fillConvexPoly(mask, mouth_points, 255)

            self.last_mask = mask  # Update last_mask with the new mask
        
        # Dilate the mask
        dilated_mask = cv2.dilate(mask, self.kernel)

        # Calculate distance transform of dilated mask
        dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)

        # Normalize distance transform
        cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

        # Convert normalized distance transform to binary mask and convert it to uint8
        _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
        masked_diff = masked_diff.astype(np.uint8)
        
        #make sure blur is an odd number
        blur = 5
        if blur % 2 == 0:
            blur += 1
        # Set blur size as a fraction of bounding box size
        blur = int(max(self.w, self.h) * blur)  # 10% of bounding box size
        if blur % 2 == 0:  # Ensure blur size is odd
            blur += 1
        masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)

        # Convert numpy arrays to PIL Images
        input1 = Image.fromarray(img)
        input2 = Image.fromarray(original_img)

        # Convert mask to single channel where pixel values are from the alpha channel of the current mask
        mask = Image.fromarray(masked_diff)

        # Ensure images are the same size
        assert input1.size == input2.size == mask.size

        # Paste input1 onto input2 using the mask
        input2.paste(input1, (0,0), mask)

        # Convert the final PIL Image back to a numpy array
        input2 = np.array(input2)

        #input2 = cv2.cvtColor(input2, cv2.COLOR_BGR2RGB)
        cv2.cvtColor(input2, cv2.COLOR_BGR2RGB, input2)
        
        return input2, mask

    def generate_digital_human(self, audio_file,fps = 25):
        self.kernel = self.last_mask = self.x = self.y = self.w = self.h = None
        
        start_time = time.time()
        # 从音频文件中获取帧数
        audio_frames = AudioUtils.get_audio_frames(audio_file,fps)
        print(f"get_audio_frames 代码块执行时间为: {time.time() - start_time} 秒")
        
        start_time = time.time()
        # 获取需要的人脸帧数据
        faces_to_use,frames_to_use = self.video_preprocessor.get_next_frames_and_faces(len(audio_frames))
        faces_to_use,frames_to_use = self.local_shuffle(faces_to_use,frames_to_use, window_size=15, swap_count=1)
        print(f"get_next_frames_and_faces 代码块执行时间为: {time.time() - start_time} 秒")
        
        gen = self.datagen(frames_to_use, faces_to_use,audio_frames)
        # 高清修复
        if self.quality == 'Enhanced':
            run_params = load_sr()
        # 合成数字人
        for i, (img_batch, mel_batch, frames, coords, audio_chunk) in enumerate(tqdm(
            gen,
            total=int(np.ceil(float(len(audio_frames))/self.batch_size)),
            desc="Processing Wav2Lip",ncols=100
        )):
            if i == 0:

                frame_h, frame_w = self.video_preprocessor.frame_first.shape[:-1]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
                out = cv2.VideoWriter('temp/result.mp4', fourcc, fps, (frame_w, frame_h))
            
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to('cuda')
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to('cuda')

            with torch.no_grad():
                pred = new_base_tech_avatars_init.model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c

                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                
                # 高清修复
                if self.quality == 'Enhanced':
                    cf = f[y1:y2, x1:x2]
                    p = upscale(p, run_params)
                    for i in range(len(frames)):
                        p, last_mask = self.create_tracked_mask(p, cf)
                
                f[y1:y2, x1:x2] = p
                out.write(f)
                
        out.release()
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            subprocess.check_call([
            "ffmpeg.exe", "-y", "-loglevel", "error",
            "-i", 'temp/result.mp4',
            "-c:v", "h264_nvenc",
            "-an",
            f'out/result_{timestamp}.mp4' ,
        ])
        except subprocess.CalledProcessError as e:
            print("FFmpeg command failed with error:", e)
        file_path =  f'out/result_{timestamp}.mp4'   
        return file_path
            
    def datagen(self, frames, face_det_results, mels):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        static = False
        img_size = 96

        for i, m in enumerate(mels):
            idx = 0 if static else i%len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (img_size, img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, img_size//2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                # 获取当前帧对应的音频数据
                audio_chunk = m
                
                yield img_batch, mel_batch, frame_batch, coords_batch, audio_chunk
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    
    # 抽帧随机换位置
    def local_shuffle(self,faces_to_use,frames_to_use, window_size, swap_count):
        """
            在一个局部窗口内随机打乱数组的元素位置。
            
            参数:
            faces_to_use (list): 人脸数据数组
            frames_to_use (list): 人脸数据对应的视频帧数组
            window_size (int): 局部窗口的大小
            swap_count (int): 每个窗口内随机替换位置的数量
            
            返回:
            list: 打乱后的数组
        """
        frames_copy = frames_to_use.copy()
        faces_copy = faces_to_use.copy()
        for start in range(0, len(frames_to_use), window_size):
            end = min(start + window_size, len(frames_to_use))
            for _ in range(swap_count):
                i = random.randint(start, end - 1)
                j = random.randint(start, end - 1)
                frames_copy[i], frames_copy[j] = frames_copy[j], frames_copy[i]
                faces_copy[i], faces_copy[j] = faces_copy[j], faces_copy[i]
        return faces_copy,frames_copy
   
# 音频工具类
class AudioUtils:
    
    @staticmethod
    def get_audio_frames(audio_file,fps = 25):
            wav = audio.load_wav(audio_file, 16000)
            mel = audio.melspectrogram(wav)
            
            mel_chunks = []
            mel_idx_multiplier = 80./fps
            i = 0
            while 1:
                start_idx = int(i * mel_idx_multiplier)
                if start_idx + new_base_tech_avatars_init.mel_step_size > len(mel[0]):
                    mel_chunks.append(mel[:, len(mel[0]) - new_base_tech_avatars_init.mel_step_size:])
                    break
                mel_chunks.append(mel[:, start_idx : start_idx + new_base_tech_avatars_init.mel_step_size])
                i += 1
            return mel_chunks

wav_path_files = []
app = Flask(__name__) 
# 与音频程序进行通信   
@app.route('/enqueue_video', methods=['POST'])
def enqueue_video():
    file_path = request.form.get('file_path')
    if not file_path:
        return jsonify({'error': 'Missing file_path parameter'}), 400
    
    wav_path_files.append(file_path)
    return jsonify({'message': 'Video enqueued successfully'})

def write_json_file(file_path, new_file_path, new_oldest_video):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []

    if new_file_path and new_oldest_video:
        new_data = {'file_path': new_file_path, 'oldest_video': new_oldest_video}
        data.append(new_data)

        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
    else:
        print("file_path和oldest_video不能为空")

# 主程序        
class VideoReader:
    def __init__(self, directory,preprocessor,quality = 'Fast'):
        # 音频文件前缀
        self.directory = directory
        # 初始化模型
        video_preprocessor = VideoPreprocessor(preprocessor)
        # 初始化数字人合成
        self.digital_human_synthesizer = DigitalHumanSynthesizer(video_preprocessor,1,quality)
    
    def list_video_files(self):
        if len(wav_path_files) > 0:
            return wav_path_files.pop(0)
        else :
            return None
        
    def send_video_data(self,file_path, oldest_video):
        write_json_file('video_data.json', file_path, oldest_video)

    def read_next_video(self):
        video_file = self.list_video_files()
        if video_file:
            video_file = video_file
            # 这里可以加上视频文件的读取逻辑
            print(f"Reading {video_file}")
            file_path = self.digital_human_synthesizer.generate_digital_human(video_file, 20)
            self.send_video_data(
                file_path,
                video_file
            )

def read_next_video():
    while True:
        video_reader.read_next_video()
# 测试
if __name__ == "__main__":
    video_reader = VideoReader(r"G:\project\utils\UnmannedSystem\text_splice_to_audioV2",'20240515_225257')
    server_thread = threading.Thread(target=read_next_video)
    server_thread.daemon = True
    server_thread.start()
    app.run(threaded=True)
    
    # newVideoPreprocessing = VideoPreprocessing('./temp/3-3-3-720p.mp4')
    # fps = newVideoPreprocessing.extract_faces()