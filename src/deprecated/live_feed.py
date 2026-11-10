import cv2
import os
from  src.constants import RTSP_URL, MIN_CONTOUR_AREA
from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
# https://learnopencv.com/moving-object-detection-with-opencv/


class LiveFeed:
    def __init__(self, motion_detection: bool = True, show_md: bool = False, prerecord: str | None = None, rotate: int | None = None):
        self.rotate = rotate
        if prerecord:
            path = Path(prerecord).resolve()
            print(path)
            self.cap = cv2.VideoCapture(str(path))
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            self.cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        
        if not self.cap.isOpened():
            print("An error has occured and the video has not been opened")
            exit(400)
        self.motion_detection = motion_detection
        self.show_motion_detection = show_md
        weights = hf_hub_download("Ultralytics/YOLO26", "yolo26s.pt", cache_dir="./cache")
        self.model = YOLO(weights)

    def generate_frames(self):
        if self.motion_detection: 
            bg_subtraction = cv2.createBackgroundSubtractorMOG2(varThreshold=200)
        counter = 0
        while True:
            
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.rotate is not None:
                print(f"rotated.. {self.rotate}")
                frame = cv2.rotate(frame, self.rotate)
            
            if self.motion_detection:
                fg_mask = bg_subtraction.apply(frame)
                retval, mask_thresh = cv2.threshold( fg_mask, 180, 255, cv2.THRESH_BINARY)
                
                # Convolution sliding window
                conv_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
                # Errosion = If not all items in conv_kern are 1 (up to a threshold) then we errode it to 0
                errosion_mask = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, conv_kern)
                errosion_mask = cv2.dilate(errosion_mask, conv_kern, iterations=1)
                contours, hierarchy = cv2.findContours(errosion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # frame_w_cntr = cv2.drawContours(frame, contours, -1, (255,0,0), 3)
                kept_contours = [cntr for cntr in contours if cv2.contourArea(cntr) > MIN_CONTOUR_AREA]
                if kept_contours:
                    counter += 1
                    print("Motion detected!" , counter)
                    preds = self.model.predict(frame, conf=.1, iou=.2)
                    class_dict = preds[0].names
                    box_classes = preds[0].boxes.cls

                    if preds[0].boxes is not None:
                        for index, box in enumerate(preds[0].boxes.xyxy.cpu().numpy()):
                            if 'bird' in class_dict[box_classes[index].item()]:
                                x1, y1, x2, y2 = map(int, box)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 10)
                
                # for cnt in kept_contours: 
                # x, y, w, h = cv2.boundingRect(cnt)
                # frame_out = cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0 ,200), 3)


            if self.show_motion_detection:
                ret, buffer = cv2.imencode(".jpg", errosion_mask)
            else:
                ret, buffer = cv2.imencode(".jpg", frame)

            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )







    
