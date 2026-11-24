from src.protocol.gen_protocol import GenericProtocol
from src.video_stream.video_stream import VideoStream
from src.data_classes import StreamCtx, ProtocolCtx, Classification
from src.interface.webui.webui import WebUI
from src.db import db_operations


class Controller:
    def __init__(
        self,
        stream_params: StreamCtx,
        protocol_params: ProtocolCtx,
        web_interface: bool = False,
        *,
        ui_host: str = "127.0.0.1",
        ui_port: int = 5000,
        db: bool = False
    ):
        self.protocol = GenericProtocol(protocol_params)
        self.stream = VideoStream(stream_params)
        self.web_interface = web_interface
        self.ui = WebUI(ui_host, ui_port) if web_interface else None
        self.db = db
        if db:
            self.current_clsf_types = db_operations.get_curr_classes()

    def start(self, type_of_interest: str = "bird"):
        frames = self.stream.capture()
        try:
            if self.ui is not None:
                self.ui.start()
            for frame in frames:
                predictions = None
                if self.stream.motion_detected or self.stream.bird_detected:
                    predictions = self.protocol.predict(frame.image)
                    if not predictions:
                        continue
                    tof = predictions.get(type_of_interest)
                    if tof:
                        tof: list[Classification]
                        self.stream.bird_identified()
                        if self.db:
                            for clsf in tof:
                                if clsf.classification not in self.current_clsf_types:
                                    db_operations.insert_new_clsf_type(clsf.classification)
                                db_operations.insert_bbox_detection(
                                    clsf.classification,
                                    self.stream._curr_image_file_opp,
                                    clsf.localization.bbox,
                                    confidence=clsf.confidence,
                                )

                if self.ui is not None:
                    self.ui.publish(frame.image, predictions)
        finally:
            try:
                frames.close()
            finally:
                if self.ui is not None:
                    self.ui.stop()
