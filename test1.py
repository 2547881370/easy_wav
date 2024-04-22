import threading
import vlc
import queue
import time

class VideoPlayer:
    def __init__(self):
        self.instance = vlc.Instance('--no-xlib --quiet')
        self.player = self.instance.media_player_new()
        self.media_list = self.instance.media_list_new([])
        self.media_list_player = self.instance.media_list_player_new()
        self.media_list_player.set_media_player(self.player)
        self.queue = queue.Queue()
        self.preloaded_filepath = None
        self.playing = False
        self.lock = threading.Lock()

        # Create a window to display the video
        self.player_window = self.instance.media_player_new()

    def play_video(self, filepath):
        media = self.instance.media_new(filepath)
        self.media_list.add_media(media)
        self.media_list_player.set_media_list(self.media_list)
        self.media_list_player.set_media_player(self.player_window)  
        self.media_list_player.play()

        while True:
            time.sleep(0.1)
            if not self.player_window.is_playing():
                break

    def preload_video(self):
        while True:
            if not self.queue.empty():
                self.preloaded_filepath = self.queue.get()
                media = self.instance.media_new(self.preloaded_filepath)
                self.media_list.add_media(media)
                self.media_list_player.set_media_list(self.media_list)
                self.media_list_player.stop()
            else:
                time.sleep(1)

    def add_video(self, filepath):
        self.queue.put(filepath)

    def play_videos(self):
        while True:
            if not self.queue.empty():
                filepath = self.queue.get()
                threading.Thread(target=self.play_video, args=(filepath,)).start()
            else:
                time.sleep(1)

# Function to create the window for video display
def create_window(player):
    player.player_window.set_hwnd(0)  # Use the default window
    player.player_window.play()

# Main function
if __name__ == "__main__":
    player = VideoPlayer()

    # Start the window creation thread
    window_thread = threading.Thread(target=create_window, args=(player,))
    window_thread.start()

    # Start the video playback thread
    video_thread = threading.Thread(target=player.play_videos)
    video_thread.start()

    # Add videos to the queue
    player.add_video("./out/result_20240421_233436.mp4")
    time.sleep(10)
    player.add_video("./out/result_20240421_233440.mp4")
    time.sleep(3)
    player.add_video("./out/result_20240421_233443.mp4")
    time.sleep(3)
    player.add_video("./out/result_20240421_233447.mp4")
    time.sleep(3)
    player.add_video("./out/result_20240421_233451.mp4")
    time.sleep(3)
    player.add_video("./out/result_20240421_233455.mp4")

    # Main thread continues executing other tasks
    try:
        while True:
            time.sleep(1)
            print("Main thread is still running...")
    except KeyboardInterrupt:
        print("Main thread exits.")
