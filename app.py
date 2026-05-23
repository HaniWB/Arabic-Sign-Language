import glob
import os
import shutil
import subprocess
import sys
from signLanguage.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLOV5_DIR = os.path.join(BASE_DIR, "yolov5")
RUNS_DIR = os.path.join(YOLOV5_DIR, "runs")
WEIGHTS_PATH = os.path.join(BASE_DIR, "exp14", "weights", "best.pt")
DATA_YAML = os.path.join(BASE_DIR, "Arabic sign language translator.v3i.yolov5pytorch", "data.yaml")

app = Flask(__name__)
CORS(app)
webcam_process = None


class ClientApp:
    def __init__(self):
        self.filename = os.path.join(BASE_DIR, "inputImage.jpg")


def start_webcam_if_needed():
    global webcam_process
    if webcam_process is not None and webcam_process.poll() is None:
        return False

    webcam_process = subprocess.Popen(
        [
            sys.executable,
            "detect.py",
            "--weights", WEIGHTS_PATH,
            "--data", DATA_YAML,
            "--img", "640",
            "--conf", "0.5",
            "--source", "0",
        ],
        cwd=YOLOV5_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)

        if os.path.exists(RUNS_DIR):
            shutil.rmtree(RUNS_DIR)

        subprocess.run(
            [
                sys.executable,
                "detect.py",
                "--weights", WEIGHTS_PATH,
                "--img", "640",
                "--conf", "0.5",
                "--source", clApp.filename,
            ],
            cwd=YOLOV5_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        exp_dirs = glob.glob(os.path.join(YOLOV5_DIR, "runs", "detect", "exp*"))
        if not exp_dirs:
            return Response("Detection produced no output", status=500)

        latest_exp = sorted(exp_dirs)[-1]
        result_image = os.path.join(latest_exp, os.path.basename(clApp.filename))
        opencodedbase64 = encodeImageIntoBase64(result_image)
        result = {"image": opencodedbase64.decode('utf-8')}

        shutil.rmtree(RUNS_DIR)

    except ValueError as val:
        print(val)
        return Response("Value not found inside json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"

    return jsonify(result)


@app.route("/live", methods=['GET'])
@cross_origin()
def predictLive():
    try:
        started = start_webcam_if_needed()
        return "Camera starting!!" if started else "Camera already running!!"

    except ValueError as val:
        print(val)
        return Response("Value not found inside json data")


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080)


