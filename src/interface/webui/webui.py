from flask import Flask, Response, render_template
from src.live_feed import LiveFeed
import argparse
from pathlib import Path

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/subtract_video")
def subtract_video():
    lv = LiveFeed(show_md=True, prerecord=app.config.get("prerecord_path"), rotate=app.config.get("rotate", None))
    return Response(
        lv.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/video")
def video():
    lv = LiveFeed(app.config.get("motion_detection"), app.config.get("show_motion_detection"), app.config.get("prerecord_path"), rotate=app.config.get("rotate", None))
    return Response(
        lv.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

def run(): 
    parser = argparse.ArgumentParser(description="Pass args for UI.")
    parser.add_argument("-md","--motion_detection", action="store_true", default=True)
    parser.add_argument("-show_md", "--show_motion_detection", action="store_true", default=False)
    parser.add_argument("-prp", "--prerecord_path", default=None )
    parser.add_argument("-rot", "--rotate", default=None)
    args = parser.parse_args()
    app.config['motion_detection'] = args.motion_detection
    app.config['show_motion_detection'] = args.show_motion_detection
    app.config['prerecord_path'] = args.prerecord_path
    try:
        app.config['rotate'] = int(args.rotate)
    except Exception:
        app.config['rotate'] = None
    app.run(debug=True)


if __name__ == "__main__":
    run()

    