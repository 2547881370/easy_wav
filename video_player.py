import json
import threading
import time
import cv2
import numpy as np
import pygame
import os
from flask import Flask, request, jsonify

class VideoPlayer:
    def __init__(self):
        # 音频路径作为键，视频帧作为值
        self.video_frames = {}
        # 音频路径列表
        self.audio_paths = []
        # 是否正在播放
        self.playing = False
        # 音频路径作为键，视频路径作为值
        self.path = {}

    def preload_video(self, video_path, audio_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        self.video_frames[audio_path] = frames
        self.audio_paths.append(audio_path)
        self.path[audio_path] = video_path

    def play_video(self, screen):
        pygame.mixer.init()
        pygame.mixer.set_num_channels(1)
        clock = pygame.time.Clock()
        while self.audio_paths:
            audio_path = self.audio_paths.pop(0)
            frames = self.video_frames[audio_path]
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            self.playing = True
            for frame in frames:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.playing = False
                        pygame.quit()
                        return
                if not self.playing:
                    return
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 0)
                
                # 逆时针旋转90°
                frame = np.rot90(frame, k=-1)
                # 缩小为原来的三分之一大小
                frame = cv2.resize(frame, (int(frame.shape[1] * 0.8), int(frame.shape[0] * 0.8)))
                
                # 计算垂直居中的位置
                screen_width, screen_height = screen.get_size()
                frame_width, frame_height, _ = frame.shape
                x_offset = (screen_width - frame_width) // 2
                y_offset = (screen_height - frame_height) // 2
                
                frame = pygame.surfarray.make_surface(frame)
                screen.blit(frame, (x_offset, y_offset))
                pygame.display.flip()
                clock.tick(20)
            pygame.mixer.music.stop()
            del self.video_frames[audio_path]
            
            file_timestamp = time.time()
            threading.Thread(target=self.handle_file_deletion, args=(audio_path, file_timestamp)).start()
            
        self.playing = False

    def handle_file_deletion(self, file_path, timestamp):
        while True:
            current_time = time.time()
            if current_time - timestamp > 10:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    if os.path.exists(self.path[file_path]):
                        os.remove(self.path[file_path])
                except OSError as e:
                    print(f"Error deleting file: {str(e)}")
                    # Handle the error, such as waiting and retrying
                    time.sleep(1)
                    continue
                break
            time.sleep(1)


def read_and_remove_first_object(json_file='video_data.json'):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            if not data or not isinstance(data, list):
                return None

            first_object = data.pop(0)
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=4)

            return first_object
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {str(e)}")
        return None

# Test the video player
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((720 * 0.8, 1280 * 0.8))
    player = VideoPlayer()

    # audio_thread = threading.Thread(target=player.play_audio)
    # audio_thread.start()

    while True:
        read_result = read_and_remove_first_object()
        if read_result:
            player.preload_video(read_result.get('file_path'), read_result.get('oldest_video'))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                player.playing = False
                pygame.quit()
                break

        if not player.playing and player.audio_paths:
            video_thread = threading.Thread(target=player.play_video, args=(screen,))
            video_thread.start()
