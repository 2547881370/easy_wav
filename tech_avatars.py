import datetime
import subprocess
import asyncio
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
                
            self.do_load('G:\\BaiduNetdiskDownload\\Easy-Wav2Lip-眠-0229\\checkpoints\\Wav2Lip.pth')
                
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
 
 
# newVideoPreprocessing = VideoPreprocessing('G:\\BaiduNetdiskDownload\\Easy-Wav2Lip-眠-0229\\temp\\demo.mp4')
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
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
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
                # # 显示每一帧图像
                # cv2.imshow('Processed Frame', f)
                # cv2.waitKey(1)  # 等待1毫秒以确保图像显示在窗口中
                
                # TODO 这里进行播放音频帧
                
        # cv2.destroyAllWindows()  # 在结束时关闭OpenCV窗口
        out.release()
        
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
        
# ws服务
class WebSocketServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server = None
        video_preprocessor = VideoPreprocessor('20240420_220826')
        self.digital_human_synthesizer = DigitalHumanSynthesizer(video_preprocessor)
        self.videoPlayer = VideoPlayer()
        self.videoPlayer.start_player_thread()

    async def handle_message(self, message):
        # 在这里处理序列化后的消息
        print(message)
        # self.videoPlayer.add_to_playlist(message)

    async def echo(self, websocket, path):
        async for message in websocket:
            # 处理收到的消息
            await self.handle_message(message)

    async def start_server(self):
        # 创建 WebSocket 服务器,监听指定的主机和端口
        self.server = await websockets.serve(self.echo, self.host, self.port)

        # 输出服务器地址信息
        print(f"WebSocket 服务器运行在 {self.server.sockets[0].getsockname()}")

        # 保持服务器运行
        await self.server.wait_closed()

    def run(self):
        # 运行主函数
        asyncio.run(self.start_server())
        

class StreamMediaClient:
    def __init__(self, server_url):
        self.server_url = server_url

    def upload_file(self, file_path):
        url = self.server_url + '/enqueue_video'
        response = requests.post(url, data={'video_path': file_path})
        return response.text

    def stream_media(self):
        url = self.server_url + '/stream'
        return requests.get(url, stream=True).content
    
class VideoReader:
    def __init__(self, directory):
        self.directory = directory
        video_preprocessor = VideoPreprocessor('20240420_220826')
        self.digital_human_synthesizer = DigitalHumanSynthesizer(video_preprocessor)
        # 在单独的线程中执行创建窗口的函数
        self.player = VideoPlayer()
        window_thread = threading.Thread(target=create_window, args=(self.player,))
        window_thread.start()

        # 在单独的线程中执行播放视频
        video_thread = threading.Thread(target=self.player.play_videos)
        video_thread.start()
        # self.client = StreamMediaClient('http://127.0.0.1:5000')
        

    def list_video_files(self):
        # 获取目录下所有文件
        files = os.listdir(self.directory)
        # 过滤出视频文件
        video_files = [f for f in files if f.endswith('.wav')]
        # 按文件创建时间从老到新排序
        video_files.sort(key=lambda x: os.path.getctime(os.path.join(self.directory, x)))
        return video_files

    def read_next_video(self):
        video_files = self.list_video_files()
        if video_files:
            # 读取最老的视频文件
            oldest_video = video_files[0]
            with open(os.path.join(self.directory, oldest_video), 'rb') as f:
                # 这里可以加上视频文件的读取逻辑
                print(f"Reading {oldest_video}")
                file_path = self.digital_human_synthesizer.generate_digital_human(os.path.join(self.directory, oldest_video), 30)
                # self.player.add_video(file_path)
                #response = self.client.upload_file(os.path.abspath(file_path))
            # 读取完成后可以删除文件
            # os.remove(os.path.join(self.directory, oldest_video))
            # os.remove(file_path)
        else:
            print("No video files found.")
            

# 创建一个显示窗口的函数
def create_window(player):
    player.player_window.set_hwnd(0)  # 使用默认窗口
    player.player_window.play()
    
    

class VideoPlayer:
    def __init__(self):
        self.instance = vlc.Instance('--no-xlib --quiet')
        self.player = self.instance.media_player_new()
        self.media_list = self.instance.media_list_new([])
        self.media_list_player = self.instance.media_list_player_new()
        self.media_list_player.set_media_player(self.player)
        self.queue = queue.Queue()
        self.preloaded_filepath = None
        self.playing = False
        self.lock = threading.Lock()

        # Create a window to display the video
        self.player_window = self.instance.media_player_new()

    def play_video(self, filepath):
        media = self.instance.media_new(filepath)
        self.media_list.add_media(media)
        self.media_list_player.set_media_list(self.media_list)
        self.media_list_player.set_media_player(self.player_window)  
        self.media_list_player.play()

        while True:
            time.sleep(0.1)
            if not self.player_window.is_playing():
                break

    def preload_video(self):
        while True:
            if not self.queue.empty():
                self.preloaded_filepath = self.queue.get()
                media = self.instance.media_new(self.preloaded_filepath)
                self.media_list.add_media(media)
                self.media_list_player.set_media_list(self.media_list)
                self.media_list_player.stop()
            else:
                time.sleep(1)

    def add_video(self, filepath):
        self.queue.put(filepath)

    def play_videos(self):
        while True:
            if not self.queue.empty():
                filepath = self.queue.get()
                threading.Thread(target=self.play_video, args=(filepath,)).start()
            else:
                time.sleep(1)
# 测试
if __name__ == "__main__":
    video_reader = VideoReader(r"G:\project\utils\UnmannedSystem\text_splice_to_audioV2\output\create\生成音频")
    while True:
        video_reader.read_next_video()

# video_preprocessor = VideoPreprocessor('20240420_220826')
# digital_human_synthesizer = DigitalHumanSynthesizer(video_preprocessor)
# digital_human_synthesizer.generate_digital_human('G:\\BaiduNetdiskDownload\\Easy-Wav2Lip-眠-0229\\temp\\test1.wav',30)
# digital_human_synthesizer.generate_digital_human('G:\\BaiduNetdiskDownload\\Easy-Wav2Lip-眠-0229\\temp\\test2.wav',30)

# ws_server = WebSocketServer("127.0.0.1", 9999)
# ws_server.run()