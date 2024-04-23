import random
import subprocess
import queue
import threading
import os
import time

class RTMPPusher:
    def __init__(self, rtmp_address):
        self.rtmp_address = rtmp_address
        self.queue = queue.Queue()
        self.is_pushing = False
        self.lock = threading.Lock()
        self.push_thread = threading.Thread(target=self._push_loop)
        self.push_thread.daemon = True
        self.push_thread.start()

    def add_to_queue(self, mp4_url):
        self.queue.put(mp4_url)

    def _push_loop(self):
        while True:
            if not self.queue.empty():
                with self.lock:
                    if not self.is_pushing:
                        self.is_pushing = True
                        mp4_url = self.queue.get()
                        self._push_mp4(mp4_url)
                        self.is_pushing = False
                        # 删除已推流的 MP4 地址和文件
                        # self._delete_mp4_file(mp4_url)

    def _push_mp4(self, mp4_url):
        # 使用当前时间戳作为流名称
        stream_name = f"stream_{int(time.time())}"
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', mp4_url,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-f', 'flv',
            f"{self.rtmp_address}/{stream_name}"  # 在 RTMP 地址后面添加流名称
        ]
        subprocess.Popen(ffmpeg_cmd)  # 使用 Popen 异步执行命令

    def _delete_mp4_file(self, mp4_url):
        try:
            os.remove(mp4_url)
            print(f"Deleted file: {mp4_url}")
        except Exception as e:
            print(f"Error deleting file: {mp4_url}, {e}")

# if __name__ == "__main__":
#     rtmp_address = 'rtmp://192.168.1.27:1935'
#     pusher = RTMPPusher(rtmp_address)
#     pusher.add_to_queue('./temp/result.mp4')
#     # time.sleep(1)
#     pusher.add_to_queue('./temp/result.mp4')
#     # time.sleep(3)
#     pusher.add_to_queue('./temp/result.mp4')
#     # time.sleep(4)
    
#     while True:
#         time.sleep(10)


class Streamer:
    def __init__(self, rtmp_url):
        self.rtmp_url = rtmp_url
        self.queue = queue.Queue()
        self.is_streaming = False

    def add_mp4(self, mp4_path):
        self.queue.put(mp4_path)

    def stream(self):
        while True:
            mp4_path = self.queue.get()
            self.is_streaming = True
            self._stream_mp4(mp4_path)
            self.is_streaming = False

    def _stream_mp4(self, mp4_path):
        command = ['ffmpeg.exe',
                   '-re',
                   '-i', mp4_path,
                   '-c', 'copy',
                   '-f', 'flv',
                   self.rtmp_url]
        subprocess.call(command)

    def start(self):
        threading.Thread(target=self.stream).start()

# 使用方法
# streamer = Streamer('rtmp://192.168.1.27:1935')
# streamer.start()

# # 在某个地方添加mp4文件
# streamer.add_mp4('./out/result_20240423_142253.mp4')
# streamer.add_mp4('./out/result_20240423_155445.mp4')
# streamer.add_mp4('./out/result_20240423_155452.mp4')
# streamer.add_mp4('./out/result_20240423_155458.mp4')


# 使用cv2读取显示视频
 
# 引入math
import math
# 引入opencv
import cv2
from ffpyplayer.player import MediaPlayer
# opencv获取本地视频
 

# Global variables
video_queue = queue.Queue()
current_video_path = None
next_video_path = None
playback_finished = threading.Event()
video_window_name = "Video Player"

def play_video(video_path, audio_play=True):
    global current_video_path, next_video_path, playback_finished
    cap = cv2.VideoCapture(video_path)
    if audio_play:
        player = MediaPlayer(video_path)
    isopen = cap.isOpened()
    if not isopen:
        print("Err: Video is failure. Exiting ...")
        return
    
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    wait = int(1000 / fps) if fps else 1
    read_frame = 0

    # Create video window if not already open
    cv2.namedWindow(video_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(video_window_name, frame_width, frame_height)

    while isopen:
        ret, frame = cap.read()
        if not ret:
            if read_frame < total_frame:
                print("Err: Can't receive frame. Exiting ...")
            else:
                print("Info: Stream is End")
            break

        read_frame += 1
        cv2.putText(frame, "[{}/{}]".format(str(read_frame), str(int(total_frame))), (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 9), 2)
        dst = cv2.resize(frame, (1920//2, 1080//2), interpolation=cv2.INTER_CUBIC)
        timecode_h = int(read_frame / fps / 60 / 60)
        timecode_m = int(read_frame / fps / 60)
        timecode_s = read_frame / fps % 60
        s = math.modf(timecode_s)
        timecode_s = int(timecode_s)
        timecode_f = int(s[0] * fps)
        print("{:0>2d}:{:0>2d}:{:0>2d}.{:0>2d}".format(timecode_h, timecode_m, timecode_s, timecode_f))

        cv2.imshow(video_window_name, dst)
        wk = cv2.waitKey(wait)
        keycode = wk & 0xff

        if keycode == ord(" "):
            cv2.waitKey(0)

        if keycode == ord('q'):
            print("Info: By user Cancal ...")
            break

        # Preload the next video while the current one is playing
        if read_frame == total_frame // 2 and next_video_path:
            cap.release()
            cv2.destroyAllWindows()
            play_video(next_video_path)
            break

    cap.release()
    cv2.destroyAllWindows()
    playback_finished.set()

def video_player():
    global current_video_path, next_video_path, playback_finished
    while True:
        if not video_queue.empty():
            if current_video_path:
                playback_finished.wait()  # Wait for the previous video to finish
                playback_finished.clear()
            current_video_path = video_queue.get()
            if not video_queue.empty():
                next_video_path = video_queue.queue[0]  # Preload the next video
            else:
                next_video_path = None
            play_video(current_video_path)
        else:
            if current_video_path:
                playback_finished.wait()  # Wait for the last video to finish
                current_video_path = None
                next_video_path = None
                print("Info: All videos played.")
                break
if __name__ == "__main__":
    # Start the video player thread
    player_thread = threading.Thread(target=video_player)
    player_thread.start()

    # Add video paths to the queue
    video_queue.put("./out/result_20240423_142253.mp4")
    video_queue.put("./out/result_20240423_155445.mp4")
    video_queue.put("./out/result_20240423_155452.mp4")
    video_queue.put("./out/result_20240423_155458.mp4")

    # Wait for the player thread to finish
    player_thread.join()
# import os
# import subprocess

# # 设置RTMP地址
# rtmp_address = "rtmp://127.0.0.1:1935/live/hls"



# # 获取out文件夹中所有视频文件路径
# out_folder = "out"
# video_files = [os.path.join(out_folder, f) for f in os.listdir(out_folder) if os.path.isfile(os.path.join(out_folder, f))]

# # 按照文件创建时间排序
# video_files.sort(key=os.path.getctime)

# # 循环推流视频文件到RTMP地址
# for video_file in video_files:
#     # 使用subprocess调用ffmpeg命令推流视频到RTMP地址
#     # subprocess.run(['ffmpeg', '-re', '-i', video_file, '-c', 'copy', '-f', 'flv', rtmp_address])
#     subprocess.run([
#         "ffmpeg", "-re", "-i", video_file,
#         "-c:v", "h264_nvenc", "-preset", "fast", "-c:a", "aac",
#         "-f", "flv", rtmp_address
#     ])