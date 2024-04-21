import os
from moviepy.editor import VideoFileClip
import threading
import time

class VideoPlayer:
    def __init__(self, onNoPlaylist = None , onPlayVideo=None, beforeCallback=None):
        self.playing = False
        self.playlist = []
        self.timer_interval = 1.0  # 定时器间隔(秒)
        self.onNoPlaylist = onNoPlaylist  # 播放队列是空的时候
        self.onPlayVideo = onPlayVideo  # 当前正在播放的视频
        self.onBeforeCallback = beforeCallback  # 播放前的回调

    def play_video(self, video_path):
        if self.onPlayVideo:
            self.onPlayVideo(video_path)

        try:
            clip = VideoFileClip(video_path)
            clip.preview()
            os.remove(video_path)
        except Exception as e:
            time.sleep(0.3)
            self.play_video(video_path)
            print(f"播放器出错: {e}")

    def play_next_video(self):
        self.playing = True
        if self.playlist:
            video_path = self.playlist.pop(0)
            if video_path is None:
                if self.onNoPlaylis != None:
                    self.onNoPlaylist()
            else:
                if self.onBeforeCallback:
                    threading.Thread(target=self.onBeforeCallback).start()  # Execute callback asynchronously
                self.play_video(video_path)
        else:
            if self.onNoPlaylis != None:
                self.onNoPlaylist()

    def timer_callback(self):
        while True:
            # 在这里添加你的定时器逻辑
            # print("定时器回调 - 调整视频效果")
            time.sleep(self.timer_interval)

    def start_timer_thread(self):
        timer_thread = threading.Thread(target=self.timer_callback)
        timer_thread.daemon = True
        timer_thread.start()

    def start_player_thread(self):
        player_thread = threading.Thread(target=self.player_thread)
        player_thread.start()

    def player_thread(self):
        while True:
            if not self.playing:
                self.play_next_video()
                self.playing = False

    def add_to_playlist(self, video_path):
        self.playlist.append(video_path)

    def add_insert_playlist(self, video_path):
        self.playlist = [video_path] + self.playlist




def onPlayVideo(video_path):
    print(f"正在播放：{video_path}")

# 创建一个VideoPlayer实例
videoPlayer = VideoPlayer(None, onPlayVideo)

# 添加视频到播放列表
videoPlayer.add_to_playlist("temp/demo.mp4")
videoPlayer.add_to_playlist("temp/demo.mp4")

# 开始播放
videoPlayer.start_player_thread()

# 等待视频播放完成
while True:
    time.sleep(1)