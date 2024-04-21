import cv2
from flask import Flask, Response
import queue
import threading
import time

app = Flask(__name__)

q = queue.Queue()

def feed():
    while True:
        if not q.empty():
            video_path = q.get()
            video = cv2.VideoCapture(video_path)
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                frame = cv2.imencode('.jpg', frame)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            video.release()
        else:
            time.sleep(1)

@app.route('/video_feed')
def video_feed():
    return Response(feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def enqueue_video():
    while True:
        video_path = input("Enter the path of a mp4 video: ")
        q.put(video_path)

if __name__ == '__main__':
    t = threading.Thread(target=enqueue_video)
    t.start()
    app.run(host='0.0.0.0', port='5000')
