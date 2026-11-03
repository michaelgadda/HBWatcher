from src.video_stream.video_stream import VideoStream
from src.video_stream.frame import Frame
from pathlib import Path
import uuid
import cv2

class Camera(VideoStream):
    def __init__(self, capture_path: str, inference_persistence: int = 30):
        self._id = uuid.uuid4()
        self._cur_frame_number = None
        self._capture_path = capture_path
        self._record = False
        self._motion_detected = False
        self._motion_detected_frame = 0
        self._inference_persistence = inference_persistence
        self._bird_detected = False
        self._bird_detected_frame = 0
        

    @property
    def cur_frame(self):
        return self._cur_frame

    @property
    def prev_frame(self):
        return self._prev_frame

    @property
    def capture_path(self):
        return self._capture_path

    @property
    def record(self):
        return self._record

    @property
    def  motion_detected(self):
        return self._motion_detected

    @property
    def  motion_detected_frame(self):
        return self._motion_detected_frame

    @property
    def inference_persistence(self):
        return self._inference_persistence

    @property
    def bird_detected(self):
        return self._bird_detected
    
    @property
    def bird_detected_frame(self):
        return self._bird_detected_frame

    @property
    def id(self):
        return self._id

    def capture(self):
        path = Path(self._capture_path).resolve()
        cap = cv2.VideoCapture(str(path))
        while True:    
            ret, frame = self.cap.read()

            if not ret:
                continue

            yield Frame(self._cur_frame_number, frame)