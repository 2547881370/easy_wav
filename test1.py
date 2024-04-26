import socket
import time

import requests



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
        
# videoDataSubscriber = VideoDataSubscriber()
# videoDataSubscriber.send_video_data(
#     'out/result_20240423_142253.mp4',
#     'G:\\project\\utils\\UnmannedSystem\\text_splice_to_audioV2\\output\\create\\生成音频\\20240426_180217.wav'
# )


videoDataSubscriber = StreamMediaClient('http://127.0.0.1:9999')
videoDataSubscriber.send_upload_file(
    'out/result_20240423_142253.mp4',
    'G:\\project\\utils\\UnmannedSystem\\text_splice_to_audioV2\\output\\create\\生成音频\\20240426_180217.wav'
)

while True:
    time.sleep(1000)