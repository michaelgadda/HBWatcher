from src.motion_detector.motion_detector import MotionDetector
import cv2
from src.constants import MIN_CONTOUR_AREA

class Mog2MotionDetector(MotionDetector):

    def __init__(self):
        self.bg_subtractor =  cv2.createBackgroundSubtractorMOG2(varThreshold=200)

    def detect_motion(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        retval, mask_thresh = cv2.threshold( fg_mask, 180, 255, cv2.THRESH_BINARY)
        conv_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        errosion_mask = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, conv_kern)
        errosion_mask = cv2.dilate(errosion_mask, conv_kern, iterations=1)
        contours, hierarchy = cv2.findContours(errosion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kept_contours = [cntr for cntr in contours if cv2.contourArea(cntr) > MIN_CONTOUR_AREA]

        if kept_contours:
            return True
        return False
