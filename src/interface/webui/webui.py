"""Display controller-published frames without owning capture or inference."""
from __future__ import annotations
from collections.abc import Iterator, Sequence
from threading import Condition, Thread
from typing import TYPE_CHECKING
import cv2
import numpy as np
from flask import Flask, Response, render_template
from werkzeug.serving import make_server

if TYPE_CHECKING:
    from src.data_classes import Classification


class WebUI:
    def __init__(self, host: str = "127.0.0.1", port: int = 5000):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self._condition = Condition()
        self._jpeg = None
        self._version = 0
        self._stopped = False
        self._server = None
        self._thread = None
        self.app.add_url_rule("/", "index", self.index)
        self.app.add_url_rule("/video", "video", self.video)

    def index(self):
        return render_template("index.html")

    def video(self):
        return Response(
            self.generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    def start(self):
        """Start the HTTP server in a background thread and return immediately."""
        if self._server is not None:
            return
        server = make_server(self.host, self.port, self.app, threaded=True)
        with self._condition:
            self._jpeg = None
            self._version = 0
            self._stopped = False
        self.port = server.server_port
        self._server = server
        self._thread = Thread(target=server.serve_forever, daemon=True)
        self._thread.start()

    def publish(
        self,
        image: np.ndarray,
        predictions: list[Classification] | None = None,
    ):
        """Encode once and broadcast the latest frame to all viewers.

        Slow viewers skip older frames. Drawing does not modify the input image.
        """
        if predictions:
            image = image.copy()
            for _, cls_predictions in predictions.items():
                for prediction in cls_predictions:
                    bbox = prediction.localization.bbox
                    if bbox is None:
                        continue
                    x1, y1, x2, y2 = map(
                        int, (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
                    )
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        image, prediction.classification, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )
        success, buffer = cv2.imencode(".jpg", image)
        if not success:
            raise ValueError("Unable to encode frame as JPEG.")
        jpeg = buffer.tobytes()
        with self._condition:
            if self._stopped:
                return
            self._jpeg = jpeg
            self._version += 1
            self._condition.notify_all()

    def generate_frames(self) -> Iterator[bytes]:
        """Each viewer independently waits for the next published frame."""
        last_version = 0
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._version != last_version or self._stopped
                )
                if self._version == last_version:
                    return
                jpeg = self._jpeg
                last_version = self._version
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + jpeg + b"\r\n"
            )

    def stop(self):
        """Wake viewers and shut down the HTTP server."""
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join()
            self._server = None
            self._thread = None
