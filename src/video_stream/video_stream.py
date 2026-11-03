from abc import ABC, abstractmethod

class VideoStream(ABC):


    @property
    @abstractmethod
    def cur_frame(self):
        pass

    @property
    @abstractmethod
    def prev_frame(self):
        pass

    @property
    @abstractmethod
    def capture_path(self):
        pass

    @property
    @abstractmethod
    def record(self):
        pass

    @property
    @abstractmethod
    def  motion_detected(self):
        pass

    @property
    @abstractmethod
    def  motion_detected_frame(self):
        pass

    @property
    @abstractmethod
    def inference_persistence(self):
        pass

    @property
    @abstractmethod
    def bird_detected(self):
        pass
    
    @property
    @abstractmethod
    def bird_detected_frame(self):
        pass

    @property
    @abstractmethod
    def id(self):
        pass


    @abstractmethod
    def capture(self):
        pass