import time

class FPS_Clock:

    def __init__(self, fps):
        self.fps = fps
        self.wait_time_per_frame = 1/fps
        self.last_time = None

    def start_timer(self):
        self.last_time = time.time()

    def tick(self):

        current_time = time.time()
        elapsed = current_time - self.last_time
        sleep_time = self.wait_time_per_frame - elapsed
        sleep_time = sleep_time if sleep_time > 0 else 0
        self.last_time = current_time

        time.sleep(sleep_time)

