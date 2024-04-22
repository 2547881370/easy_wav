# from flask import Flask, Response, request
# from flask.templating import render_template
# import requests
# from flask_socketio import SocketIO
# import subprocess
# import queue

# app = Flask(__name__)
# socketio = SocketIO(app)

# q = queue.Queue()

# def generate_video():
#     while True:
#         video_path = q.get()
#         cmd = ['ffmpeg.exe', '-i', video_path, '-f', 'mp4', '-movflags', 'frag_keyframe+empty_moov', '-']
#         p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         while True:
#             data = p.stdout.read(1024)
#             if not data:
#                 break
#             yield data
#         q.task_done()

# @app.route('/')
# def index():
#     return render_template('./index.html')

# @socketio.on('connect')
# def handle_connect():
#     socketio.emit('video_stream', generate_video())

# @app.route('/enqueue_video', methods=['POST'])
# def enqueue_video():
#     video_path = request.form.get('video_path')
#     if video_path:
#         q.put(video_path)
#         return "Video enqueued successfully", 200
#     else:
#         return "No video path provided", 400

# if __name__ == '__main__':
#     socketio.run(app, host='0.0.0.0', port='5000')



import tkinter as tk
import vlc
 
class VideoPlayer:
    def __init__(self, master):
        self.master = master
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()
 
        # 创建GUI界面
        self.create_widgets()
 
    def create_widgets(self):
        # 创建Canvas用于播放视频
        self.canvas = tk.Canvas(self.master, width=800, height=600)
        self.canvas.pack()
 
        # 添加按钮控制视频播放
        self.play_button = tk.Button(self.master, text="Play", command=self.play)
        self.play_button.pack(side=tk.LEFT)
        self.pause_button = tk.Button(self.master, text="Pause", command=self.pause)
        self.pause_button.pack(side=tk.LEFT)
        self.stop_button = tk.Button(self.master, text="Stop", command=self.stop)
        self.stop_button.pack(side=tk.LEFT)
 
        # 添加滑动条控制音量
        self.volume_scale = tk.Scale(self.master, from_=0, to=100, orient=tk.HORIZONTAL, command=self.set_volume)
        self.volume_scale.pack(side=tk.BOTTOM)
 
        # 加载视频文件
        self.media = self.instance.media_new("./temp/result_20240421_165753.mp4")
        self.player.set_media(self.media)
 
    def play(self):
        # 开始播放视频
        self.player.play()
        
    def pause(self):
        # 暂停播放视频
        self.player.pause()
 
    def stop(self):
        # 停止播放视频
        self.player.stop()
 
    def set_volume(self, volume):
        # 设置音量
        self.player.audio_set_volume(int(volume))
 
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Video Player")
    player = VideoPlayer(root)
    root.mainloop()