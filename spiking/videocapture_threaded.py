from threading import Thread
import cv2


class VideoCaptureThreaded:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.success, self.frame = self.cap.read()
        self.end = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.end:
            if not self.success:
                self.stop()
            else:
                self.success, self.frame = self.cap.read()

    def stop(self):
        self.end = True
        # self.cap.release()

