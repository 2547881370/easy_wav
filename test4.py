import os, time
import vlc
import pyvirtualcam
from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
import subprocess
import queue
import threading

class LiveStream(object):
    def __init__(self, rtmpUrl='rtsp://127.0.0.1:8554/mystream'):
        self.video_queue = queue.Queue()
        # 自行设置
        self.rtmpUrl = rtmpUrl
        print("开启推流:",self.rtmpUrl)
        # Get video information
        self.fps = 30
        self.width = 1080
        self.height = 1920
    

    def run_push_stream(self):
        while True:
            audio_file =self.video_queue.get()
            # ffmpeg command
            # self.command = ['D:\\UnmannedSystem\\douyin_room_digithuman-main\\ffmpeglibx264\\ffmpeg\\bin\\ffmpeg.exe',
            #         '-i', audio_file,
            #         '-s 1080x1920 -pix_fmt yuvj420p -vcodec libx264 -c:a aac -movflags +faststart',
            #         '-f', 'flv',
            #         self.rtmpUrl]
            # ffmpeg command
            # self.command = ['G:/project/utils/UnmannedSystem/AI人脸替换工具V5.0完整包/imageio_ffmpeg/binaries/ffmpeg-win64-v4.2.2.exe',
            #         '-probesize 10M -analyzeduration 180M -r 30 -loop 1 -i 1.png',
            #         '-i', audio_file,
            #         '-s 1080x1920 -pix_fmt yuvj420p -vcodec libx264 -c:a aac -shortest -movflags +faststart',
            #         '-f', 'flv',
            #         self.rtmpUrl]
            self.command = ['G:/project/utils/UnmannedSystem/AI人脸替换工具V5.0完整包/imageio_ffmpeg/binaries/ffmpeg-win64-v4.2.2.exe',
                                '-re',
                                '-i', audio_file,
                                '-c copy',
                                '-f', 'rtsp',
                                self.rtmpUrl]
            # 管道配置
            command_str=" ".join(self.command)
            print("command:", command_str)
            self.p = subprocess.Popen(cwd=os.getcwd(), args=command_str, shell=False, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            output, error_output = self.p.communicate()
            print(f"执行成功:\n{str(output,'utf-8')}")
            
    def push_video(self,video_file):
        self.video_queue.put(video_file)        

    def start(self):
        threads = [
            threading.Thread(name='live_stream_thread', target=LiveStream.run_push_stream, args=(self,)),
        ]
        [thread.setDaemon(True) for thread in threads]
        [thread.start() for thread in threads]

class VLC_Player:
    '''
        args:设置 options
    '''
    def __init__(self, *args):
        if args:
            instance = vlc.Instance(*args)
            self.media = instance.media_player_new()
        else:
            self.media = vlc.MediaPlayer()
        self.add_callback(vlc.EventType.MediaPlayerTimeChanged, self.my_call_back)  

        # 流媒体视频
        self.live_stream = LiveStream()
        self.live_stream.start()
        
    # 设置待播放的url地址或本地文件路径，每次调用都会重新加载资源
    def set_uri(self, uri):
        self.media.set_mrl(uri)

    # 播放 成功返回0，失败返回-1
    def play(self, path=None):
        if path:
            self.set_uri(path)
            return self.media.play()
        else:
            return self.media.play()

    def video_process(self, audio_file):
        outFile = audio_file.split('.mp3')[0]+'_output.mp4'
        self.live_stream.push_video((audio_file))
        return outFile
    
    # 暂停
    def pause(self):
        self.media.pause()

    # 恢复
    def resume(self):
        self.media.set_pause(0)

    # 停止
    def stop(self):
        self.media.stop()

    # 释放资源
    def release(self):
        return self.media.release()

    # 是否正在播放
    def is_playing(self):
        return self.media.is_playing()

    # 已播放时间，返回毫秒值
    def get_time(self):
        return self.media.get_time()

    # 拖动指定的毫秒值处播放。成功返回0，失败返回-1 (需要注意，只有当前多媒体格式或流媒体协议支持才会生效)
    def set_time(self, ms):
        return self.media.get_time()

    # 音视频总长度，返回毫秒值
    def get_length(self):
        return self.media.get_length()

    # 获取当前音量（0~100）
    def get_volume(self):
        return self.media.audio_get_volume()

    # 设置音量（0~100）
    def set_volume(self, volume):
        return self.media.audio_set_volume(volume)

    # 返回当前状态：正在播放；暂停中；其他
    def get_state(self):
        state = self.media.get_state()
        if state == vlc.State.Playing:
            return 1
        elif state == vlc.State.Paused:
            return 0
        else:
            return -1

    # 当前播放进度情况。返回0.0~1.0之间的浮点数
    def get_position(self):
        return self.media.get_position()

    # 拖动当前进度，传入0.0~1.0之间的浮点数(需要注意，只有当前多媒体格式或流媒体协议支持才会生效)
    def set_position(self, float_val):
        return self.media.set_position(float_val)

    # 获取当前文件播放速率
    def get_rate(self):
        return self.media.get_rate()

    # 设置播放速率（如：1.2，表示加速1.2倍播放）
    def set_rate(self, rate):
        return self.media.set_rate(rate)

    # 设置宽高比率（如"16:9","4:3"）
    def set_ratio(self, ratio):
        self.media.video_set_scale(0)  # 必须设置为0，否则无法修改屏幕宽高
        self.media.video_set_aspect_ratio(ratio)

    # 注册监听器
    def add_callback(self, event_type, callback):
        self.media.event_manager().event_attach(event_type, callback)

    # 移除监听器
    def remove_callback(self, event_type, callback):
        self.media.event_manager().event_detach(event_type, callback)
        
    def my_call_back(self, event):
        #print("video call:", self.get_time())
        pass        

#Player.camera_source = pyvirtualcam.Camera(width=1080, height=1920, fps=30)
#print(f'Using virtual camera: {Player.camera_source.device}')


if "__main__" == __name__:
    player = VLC_Player()
    # 播放本地mp3
    player.video_process("G:/BaiduNetdiskDownload/Easy-Wav2Lip-mian-0229/temp/movie.mp4")
    player.video_process("G:/BaiduNetdiskDownload/Easy-Wav2Lip-mian-0229/temp/movie.mp4")
    player.video_process("G:/BaiduNetdiskDownload/Easy-Wav2Lip-mian-0229/temp/movie.mp4")
    player.video_process("G:/BaiduNetdiskDownload/Easy-Wav2Lip-mian-0229/temp/movie.mp4")
    player.video_process("G:/BaiduNetdiskDownload/Easy-Wav2Lip-mian-0229/temp/movie.mp4")
    input('please')
    
    