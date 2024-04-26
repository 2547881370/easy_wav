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
        
videoDataSubscriber = VideoDataSubscriber()
videoDataSubscriber.send_video_data(
    'out/result_20240426_224347.mp4',
    'G:\\project\\utils\\UnmannedSystem\\text_splice_to_audioV2\\output\\create\\\u751f\u6210\u97f3\u9891\\\u654f\u654f\u808c\u5305\u5305\u808c\u5907\u5b55\u548c\u54fa\u4e73\u671f\uff0c\u5988\u5988\u4eec\u5c0f\u670b\u53cb\u90fd\u662f\u53ef\u4ee5\u7528\u7684\uff0c\u53ea\u8981\u5c31\u5269\u6700\u540e\u7684\u4e24\u5355\u4e86\uff0c\u4e5f\u5e2e\u4f60\u4e0b\u5348\u5b89\u6392\u4e00\u4e0b\uff0c\u4f18\u5148\u52a0\u6025\u554a\u3002_create_localAudioMode.wav'
)
time.sleep(3)
videoDataSubscriber.send_video_data(
    'out/result_20240426_224352.mp4',
    'G:\\project\\utils\\UnmannedSystem\\text_splice_to_audioV2\\output\\create\\\u751f\u6210\u97f3\u9891\\\u5c31\u7684\u4e24\u5f39\u56fa\u5b58\u4e86\uff0c\u4f60\u62b9\u8138\u5316\u5986\u524d\u9694\u79bb\uff0c\u9632\u6652\u4e09\u6548\u5408\u4e00\uff0c\u6211\u7684\u9632\u6652\u662f\u53ef\u4ee5\u5f53\u5986\u524d\u7684\u3002_create_localAudioMode.wav'
)
time.sleep(3)
videoDataSubscriber.send_video_data(
    'out/result_20240426_224356.mp4',
    'G:\\project\\utils\\UnmannedSystem\\text_splice_to_audioV2\\output\\create\\\u751f\u6210\u97f3\u9891\\\u51cf\u5c11\u8102\u548c\u8d1f\u5f39\u7684\u7532\u5b50\u538b\u597d\uff0c\u6211\u4eec\u6765\u770b\u4e00\u4e0b\u4ef7\u683c\uff0c\u65e5\u5e38\u8d2d\u4e70\u5b83\u55ef\u3002_create_localAudioMode.wav'
)

# videoDataSubscriber = StreamMediaClient('http://127.0.0.1:9999')
# videoDataSubscriber.send_upload_file(
#     'out/result_20240423_142253.mp4',
#     'G:\\project\\utils\\UnmannedSystem\\text_splice_to_audioV2\\output\\create\\生成音频\\20240426_180217.wav'
# )

while True:
    time.sleep(1000)