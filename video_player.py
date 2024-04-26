import json
import threading
import time
import cv2
import pygame
import socket
import os
from flask import Flask, request, jsonify

class VideoPlayer:
    def __init__(self):
        self.video_frames = {}
        self.audio_paths = []
        self.playing = False
        self.path = {}

    def add_video(self, video_path, audio_path):
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

    def play(self, screen):
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
                frame = pygame.surfarray.make_surface(frame)
                screen.blit(frame, (0, 0))
                pygame.display.flip()
                clock.tick(20)
            pygame.mixer.music.stop()
            del self.video_frames[audio_path]
            # os.remove(audio_path)
            # os.remove(self.path[audio_path])
        self.playing = False

def read_and_remove_first_object(json_file = 'video_data.json'):
    with open(json_file, 'r') as f:
        data = json.load(f)
        if not data:
            return None
        
        first_object = data.pop(0)
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)
        
        return first_object
            
            
# 测试
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((800, 500))
    player = VideoPlayer()

    while True:
        read_result = read_and_remove_first_object()
        if read_result:
                player.preload_video(
                    read_result.get('file_path'),read_result.get('oldest_video')
                )
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                player.playing = False
                pygame.quit()
                break

        if not player.playing and player.audio_paths:
            player.play(screen)
    


    
