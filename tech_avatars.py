import datetime
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
 
 
# newVideoPreprocessing = VideoPreprocessing('G:\\BaiduNetdiskDownload\\Easy-Wav2Lip-main-0229\\temp\\demo3.mp4')
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

    def load_frames(self):
        # 列出指定文件夹中以'frame_'为前缀的.npy文件数量
        frame_files = [f for f in os.listdir(f'./cache/{self.dirName}/') if f.startswith('frame_')]
        self.frames = len(frame_files)
        self.frame_first = np.load(f'./cache/{self.dirName}/frame_0.npy')

    def load_faces(self):
        # 列出指定文件夹中以'face_'为前缀的.npy文件数量
        face_files = [f for f in os.listdir(f'./cache/{self.dirName}/') if f.startswith('face_')]
        self.faces = len(face_files)
    
    def get_next_frames_and_faces(self, num_frames):
        faces_to_use = []
        frames_to_use = []
        for _ in range(num_frames):
            # 判断索引是否超出范围
            if self.current_index >= self.frames:
                self.current_index = self.frames - 1
                self.direction = -1  # 当索引超出范围时，改变方向为逆向
            elif self.current_index < 0:
                self.current_index = 0
                self.direction = 1  # 当索引超出范围时，改变方向为正向
            # print(f'当前索引 ： {self.current_index}；当前视频帧数 ：{self.frames}；当前人脸帧数：{self.faces}')
            # 读取人脸数据
            loaded_face_results_array = np.load(f'./cache/{self.dirName}/face_{self.current_index}.npy', allow_pickle=True)
            # 读取视频帧数据
            loaded_frame_results_array = np.load(f'./cache/{self.dirName}/frame_{self.current_index}.npy')
            faces_to_use.append(loaded_face_results_array)
            frames_to_use.append(loaded_frame_results_array)
            # 根据当前方向更新索引
            self.current_index += self.direction
        return faces_to_use, frames_to_use

# 数字人合成
class DigitalHumanSynthesizer:
    def __init__(self, video_preprocessor, batch_size=1):
        self.video_preprocessor = video_preprocessor
        self.batch_size = batch_size

    def generate_digital_human(self, audio_file,fps = 25):
        # 从音频文件中获取帧数
        audio_frames = AudioUtils.get_audio_frames(audio_file,fps)
        
        # 获取需要的人脸帧数据
        faces_to_use,frames_to_use = self.video_preprocessor.get_next_frames_and_faces(len(audio_frames))
        
        gen = self.datagen(frames_to_use, faces_to_use,audio_frames)
        
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
                f[y1:y2, x1:x2] = p
                out.write(f)
                # 显示每一帧图像
                # cv2.imshow('Processed Frame', f)
                # cv2.waitKey(1)  # 等待1毫秒以确保图像显示在窗口中
                
        out.release()
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            subprocess.check_call([
            "ffmpeg.exe", "-y", "-loglevel", "error",
            "-i", 'temp/result.mp4',
            "-i", audio_file,
            "-c:v", "h264_nvenc",
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


# 专门用于订阅soket
class VideoDataSubscriber:
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

    def send_video_data(self, file_path, oldest_video):
        self.socket.sendall(f"{file_path},{oldest_video}".encode())
        print("Data sent to server")

    def close_connection(self):
        self.socket.close()

class StreamMediaClient:
    def __init__(self, server_url):
        self.server_url = server_url

    def upload_file(self, file_path):
        url = self.server_url + '/enqueue_video'
        response = requests.post(url, data={'video_path': file_path})
        return response.text
    
    def send_upload_file(self, file_path,oldest_video):
        url = self.server_url + '/enqueue_video'
        response = requests.post(url, data={'file_path': file_path,'oldest_video': oldest_video})
        return response.text

    def stream_media(self):
        url = self.server_url + '/stream'
        return requests.get(url, stream=True).content


wav_path_files = []
app = Flask(__name__)    
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
    def __init__(self, directory):
        self.directory = directory
        video_preprocessor = VideoPreprocessor('20240418_235942')
        self.digital_human_synthesizer = DigitalHumanSynthesizer(video_preprocessor)
        self.streamMediaClient = StreamMediaClient('http://127.0.0.1:9999')     
        # pygame.init()
        # self.screen = pygame.display.set_mode((800, 500))
        # self.player = VideoPlayer()   
        
        # init_video_thread = threading.Thread(target=self.video_init)
        # init_video_thread.daemon = True
        # init_video_thread.start()
    
    def video_init(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.player.playing = False
                    pygame.quit()
                    return

            if not self.player.playing and self.player.audio_paths:
                self.player.play(self.screen)
        
    def list_video_files(self):
        if len(wav_path_files) > 0:
            return wav_path_files.pop(0)
        else :
            return None
        
    def run_video(self,file_path, oldest_video):
        self.player.add_video(file_path, oldest_video)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.player.playing = False
                pygame.quit()
                return

        if not self.player.playing and self.player.audio_paths:
            self.player.play(self.screen)
    
    def send_video_data(self,file_path, oldest_video):
        # self.videoDataSubscriber.send_video_data(file_path, oldest_video)
        # video_thread = threading.Thread(target=self.run_video,args=(file_path, oldest_video,))
        # video_thread.daemon = True
        # video_thread.start()
        # self.streamMediaClient.send_upload_file(file_path, oldest_video)
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
                
            # 读取完成后可以删除文件
            # os.remove(os.path.join(self.directory, oldest_video))
            # os.remove(file_path)

def read_next_video():
    while True:
        video_reader.read_next_video()
# 测试
if __name__ == "__main__":
    video_reader = VideoReader(r"G:\project\utils\UnmannedSystem\text_splice_to_audioV2")
    server_thread = threading.Thread(target=read_next_video)
    server_thread.daemon = True
    server_thread.start()
    app.run(threaded=True)