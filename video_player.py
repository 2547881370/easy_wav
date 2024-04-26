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

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 500))
    player = VideoPlayer()

    HOST = '127.0.0.1'  # 服务器主机地址
    PORT = 65432        # 服务器端口号

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()

        print(f"Server listening on {HOST}:{PORT}")

        while True:
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                data = conn.recv(1024)
                if not data:
                    break

                file_path, oldest_video = data.decode().split(",")
                print(file_path, oldest_video)
                player.add_video(file_path, oldest_video)
            
            print("执行")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    player.playing = False
                    pygame.quit()
                    return

            if not player.playing and player.audio_paths:
                player.play(screen)

app = Flask(__name__)    
@app.route('/enqueue_video', methods=['POST'])
def enqueue_video():
    file_path = request.form.get('file_path')
    oldest_video = request.form.get('oldest_video')
    if not file_path:
        return jsonify({'error': 'Missing file_path parameter'}), 400
    
    read_next_video(file_path,oldest_video)
    
    return jsonify({'message': 'Video enqueued successfully'})


def read_next_video(file_path, oldest_video):
        player.add_video(file_path, oldest_video)
                
        print("执行",file_path, oldest_video)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                player.playing = False
                pygame.quit()
                return

        if not player.playing and player.audio_paths:
            player.play(screen)
        
# 测试
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((800, 500))
    player = VideoPlayer()
    
    app.run(threaded=True,port=9999)
    


    
