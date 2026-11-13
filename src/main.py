import argparse
import os
import cv2
import logging
from src.controller import Controller
from src.data_classes import StreamCtx, ProtocolCtx
from src.protocol.protocol_registry import ProtocolEnum
from src.motion_detector.motion_detector_registry import MotionDetectorEnum

LOGGER = logging.getLogger("Main")

def run():
    """Configure and run BirdWatcher from the command line."""
    parser = argparse.ArgumentParser(
        description="Run BirdWatcher with an optional web UI."
    )
    parser.add_argument(
        "--source", "--capture-path", "-prp", "--prerecord_path",
        default=os.environ.get("BIRD_WATCHER_RTSP_URL"),
        help="Video file or stream URL (defaults to BIRD_WATCHER_RTSP_URL).",
    )
    parser.add_argument(
        "--rotate", type=int, choices=[cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE],
        default=None, help="Flip a frame in CV2.",
    )
    parser.add_argument(
        "--iou", type=float,
        default=.5, help="Intersection-over-Union used for NMS.",
    )
    parser.add_argument(
        "--det_conf", type=float, 
        default=.25, help="Sets the minimum confidence required to keep a detection.",
    )
    parser.add_argument("--record", action = "store_true", default = False, 
                        help="Pass flag if you want to record bird detections.")
    parser.add_argument(
        "--protocol", type=ProtocolEnum, choices=list(ProtocolEnum),
        default=ProtocolEnum.YOLO_PROTOCOL, help="Prediction protocol.",
    )
    parser.add_argument(
        "--inference-persistence", type=int, default=30,
        help="Number of frames to retain the motion-detected state.",
    )
    parser.add_argument(
        "--motion-detector",
        choices=[detector.value for detector in MotionDetectorEnum] + ["none"],
        default=MotionDetectorEnum.MOG2.value,
        help="Motion detector; use 'none' to disable motion detection.",
    )
    parser.add_argument(
        "--backend", type=int, default=cv2.CAP_FFMPEG,
        help="OpenCV capture backend ID (0 selects automatically; 1900 is FFmpeg).",
    )
    parser.add_argument(
        "--web-interface", action=argparse.BooleanOptionalAction, default=True,
        help="Enable the web UI; use --no-web-interface for headless processing.",
    )
    parser.add_argument("--ui-host", "--host", dest="ui_host", default="127.0.0.1",
                        help="Web UI listening address.")
    parser.add_argument("--ui-port", "--port", dest="ui_port", type=int, default=5000,
                        help="Web UI listening port (0 selects an available port).")
    args = parser.parse_args()
    if not args.source:
        parser.error("Provide --source or set BIRD_WATCHER_RTSP_URL.")
    if args.inference_persistence < 0:
        parser.error("--inference-persistence must be nonnegative.")
    if not 0 <= args.ui_port <= 65535:
        parser.error("--ui-port must be between 0 and 65535.")

    if not isinstance(args.iou, (float, int)) or (args.iou > 1 or args.iou < 0) :
        LOGGER.error("IoU must be a float between 0.0 and 1")

    if not isinstance(args.det_conf, (float, int)) or (args.det_conf > 1 or args.det_conf < 0) :
        LOGGER.error("Detection confidence must be a float between 0.0 and 1")

    protocol_params = ProtocolCtx(args.iou, args.det_conf)

    stream_params = StreamCtx(
        capture_path=args.source,
        inference_persistence=args.inference_persistence,
        motion_detector=(
            None if args.motion_detector == "none"
            else MotionDetectorEnum(args.motion_detector)
        ),
        backend=args.backend,
        rotate= None if not args.rotate else args.rotate,
        record=args.record 
    )
    Controller(
        protocol=args.protocol,
        stream_params=stream_params,
        protocol_params=protocol_params,
        web_interface=args.web_interface,
        ui_host=args.ui_host,
        ui_port=args.ui_port,
    ).start()


if __name__ == "__main__":
    run()
