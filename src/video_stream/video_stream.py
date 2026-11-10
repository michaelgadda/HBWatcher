from pathlib import Path
import uuid
from datetime import datetime
import cv2
from src.motion_detector.motion_detector_registry import MOTION_DETECTOR_REGISTRY, MotionDetectorEnum
from src.video_stream.frame import Frame
from src.data_classes import StreamCtx

class VideoStream:
    """Shared frame capture, motion detection, and recording for files and streams."""
    def __init__(self, params: StreamCtx):
        self._id = uuid.uuid4()
        self._motion_detector = None
        if params.motion_detector:
            self._motion_detector = MOTION_DETECTOR_REGISTRY[params.motion_detector]()
        self._cur_frame_number = -1
        self._capture_path = params.capture_path
        self._backend = params.backend
        self._record = False
        self._last_record_frame_num = None
        self._cur_record_opp = None
        self._motion_detected = False
        self._motion_detected_frame = 0
        self._inference_persistence = params.inference_persistence
        self._bird_detected = False
        self._bird_detected_frame = 0
        self._cur_frame = None
        self._prev_frame = None

    ################## Properties ##################

    @property
    def cur_record_opp(self):
        return self._cur_record_opp

    @property
    def last_record_frame_num(self):
        return self._last_record_frame_num

    @property
    def motion_detector(self):
        return self._motion_detector

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
    def motion_detected(self):
        return self._motion_detected

    @property
    def motion_detected_frame(self):
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

    ################## Methods ##################

    def record_frame(self):
        if self._cur_frame is None:
            raise RuntimeError("Capture a frame before recording it.")

        output_path = Path("./hb_records")
        output_path.mkdir(parents=True, exist_ok=True)
        if (
            self._cur_record_opp is None
            or self._last_record_frame_num is None
            or self._cur_frame_number - self._last_record_frame_num > 30
        ):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._cur_record_opp = output_path / f"hb_cap_{timestamp}"
            self._cur_record_opp.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = self._cur_record_opp / f"{timestamp}.jpg"
        success = cv2.imwrite(str(image_path), self._cur_frame.image)
        if success:
            self._last_record_frame_num = self._cur_frame_number
            print("Image saved successfully.")
        else:
            print("Failed to save image. Check if the directory path exists.")

    def capture(self):
        """Yield frames until the source ends or a read fails."""
        cap = cv2.VideoCapture(str(self._capture_path), self._backend)
        try:
            if not cap.isOpened():
                raise RuntimeError("Unable to open video source.")

            while True:
                ret, image = cap.read()
                if not ret:
                    break

                if self._motion_detected:
                    if self._motion_detected_frame >= self._inference_persistence:
                        self._motion_detected = False
                    else:
                        self._motion_detected_frame += 1

                self._cur_frame_number += 1
                frame = Frame(self._cur_frame_number, image)
                self._prev_frame = self._cur_frame
                self._cur_frame = frame

                if self._motion_detector is not None:
                    if self._motion_detector.detect_motion(frame.image):
                        self._motion_detected = True
                        self._motion_detected_frame = 0

                if self._record:
                    self.record_frame()

                yield frame
        finally:
            cap.release()
