from threading import Thread
import cv2


class ImshowThreaded:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.end = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.end:
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.end = True

    def stop(self):
        self.end = True
