from src.protocol.protocol_registry import PROTOCOL_REGISTRY, ProtocolEnum
from src.video_stream.video_stream import VideoStream
from src.data_classes import StreamCtx
from src.interface.webui.webui import WebUI


class Controller:
    def __init__(
        self,
        protocol: ProtocolEnum,
        stream_params: StreamCtx,
        web_interface: bool = False,
        *,
        ui_host: str = "127.0.0.1",
        ui_port: int = 5000,
    ):
        self.protocol = PROTOCOL_REGISTRY[protocol]()
        self.stream = VideoStream(stream_params)
        self.web_interface = web_interface
        self.ui = WebUI(ui_host, ui_port) if web_interface else None

    def start(self):
        frames = self.stream.capture()
        try:
            if self.ui is not None:
                self.ui.start()
            for frame in frames:
                predictions = None
                if self.stream.motion_detected():
                    predictions = self.protocol.predict(frame.image)
                    # TODO: If a bird is detected, set bird_detected in video stream (or move variable out into controller)
                if self.ui is not None:
                    self.ui.publish(frame.image, predictions)
        finally:
            try:
                frames.close()
            finally:
                if self.ui is not None:
                    self.ui.stop()
