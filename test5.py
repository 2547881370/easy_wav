import os
import subprocess
import threading
import queue
import vlc
import time
from datetime import datetime

class VideoConverter(threading.Thread):
    def __init__(self, mp4_queue, m3u8_queue):
        super().__init__()
        self.mp4_queue = mp4_queue
        self.m3u8_queue = m3u8_queue

    def run(self):
        while True:
            try:
                input_file = self.mp4_queue.get(timeout=1)
                output_dir = 'static/hls'
                output_file = os.path.join(output_dir, f"output_{datetime.now().strftime('%Y%m%d%H%M%S')}.m3u8")
                self.convert_to_hls(input_file, output_file)
                self.m3u8_queue.put(output_file)
                self.mp4_queue.task_done()
            except queue.Empty:
                pass

    def convert_to_hls(self, input_file, output_file):
        try:
            subprocess.run(['G:\\project\\utils\\UnmannedSystem\\AI人脸替换工具V5.0完整包\\imageio_ffmpeg\\binaries\\ffmpeg-win64-v4.2.2.exe', '-i', input_file, '-hls_time', '10', '-hls_list_size', '0', '-c:v', 'copy', '-bsf:a', 'aac_adtstoasc', output_file], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("ffmpeg error:", e.stderr)
            raise e

class VideoPlayer(threading.Thread):
    def __init__(self, m3u8_queue):
        super().__init__()
        self.m3u8_queue = m3u8_queue
        self.instance = vlc.Instance('--no-xlib')
        self.player = self.instance.media_player_new()
        self.current_media = None

    def run(self):
        while True:
            try:
                m3u8_file = self.m3u8_queue.get(timeout=1)
                self.play(m3u8_file)
            except queue.Empty:
                pass

    def play(self, m3u8_file):
        media = self.instance.media_new(m3u8_file)
        self.player.set_media(media)
        self.player.play()
        self.current_media = media
        time.sleep(2)  # Allow some time for VLC to initialize before returning

    def stop(self):
        if self.current_media:
            self.player.stop()

def main():
    mp4_queue = queue.Queue()
    m3u8_queue = queue.Queue()

    # Add MP4 files to the queue (replace with your logic to add files)
    mp4_queue.put('./out/1.mp4')
    mp4_queue.put('./out/2.mp4')
    mp4_queue.put('./out/3.mp4')
    mp4_queue.put('./out/4.mp4')
    mp4_queue.put('./out/5.mp4')
    mp4_queue.put('./out/6.mp4')

    converter = VideoConverter(mp4_queue, m3u8_queue)
    converter.start()

    player = VideoPlayer(m3u8_queue)
    player.start()

    # Wait for the conversion and playback threads to finish
    mp4_queue.join()
    m3u8_queue.join()

    # Stop the player thread
    player.stop()

if __name__ == "__main__":
    main()
