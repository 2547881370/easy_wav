import queue
import subprocess
import time
from flask import Flask, Response, request

app = Flask(__name__)

q = queue.Queue()

def generate_video():
    while True:
        video_path = q.get()
        cmd = ['ffmpeg.exe', '-i', video_path, '-f', 'mp4', '-movflags', 'frag_keyframe+empty_moov', '-']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while True:
            data = p.stdout.read(1024)
            if not data:
                break
            yield data
        q.task_done()

@app.route('/')
def index():
    return "Flask server"

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='video/mp4')

@app.route('/enqueue_video', methods=['POST'])
def enqueue_video():
    video_path = request.form.get('video_path')
    if video_path:
        q.put(video_path)
        return "Video enqueued successfully", 200
    else:
        return "No video path provided", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
