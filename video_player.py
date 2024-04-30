import json
import threading
import time
import cv2
import pygame
import os
from flask import Flask, request, jsonify

class VideoPlayer:
    def __init__(self):
        self.video_frames = {}
        self.audio_paths = []
        self.playing = False
        self.path = {}
        self.files_to_delete = []
        self.audio_queue = []
        self.audio_subscribers = []

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
                frame = pygame.surfarray.make_surface(frame)
                screen.blit(frame, (0, 0))
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

    def play_audio(self):
        while True:
            if self.audio_queue:
                audio_path = self.audio_queue.pop(0)
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(1)
            else:
                time.sleep(1)

    def subscribe_audio(self, subscriber):
        self.audio_subscribers.append(subscriber)

    def publish_audio(self, audio_path):
        for subscriber in self.audio_subscribers:
            subscriber.receive_audio(audio_path)

class AudioPlayer:
    def __init__(self, video_player):
        self.video_player = video_player

    def receive_audio(self, audio_path):
        self.video_player.audio_queue.append(audio_path)

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
    screen = pygame.display.set_mode((1980, 800))
    player = VideoPlayer()
    audio_player = AudioPlayer(player)

    audio_thread = threading.Thread(target=player.play_audio)
    audio_thread.start()

    while True:
        read_result = read_and_remove_first_object()
        if read_result:
            player.preload_video(read_result.get('file_path'), read_result.get('oldest_video'))
            player.publish_audio(read_result.get('oldest_video'))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                player.playing = False
                pygame.quit()
                break

        if not player.playing and player.audio_paths:
            video_thread = threading.Thread(target=player.play_video, args=(screen,))
            video_thread.start()
