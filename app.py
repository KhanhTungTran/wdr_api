from flask import Flask, request, jsonify
from google.protobuf.message import Error
import time
from removal import WatermarkRemoval
from models.experimental import attempt_load
from utils.general  import set_logging
from utils.torch_utils import select_device
import requests, glob, os
from detect import detect


app = Flask(__name__)

@app.route("/predict/", methods=['GET', 'POST'])
def post():
    execution_time = time.time()
    try:
        content = request.json
        url_list = content["urls"]
        
        download_image_ipg(url_list, 'images_to_infer/')

        paths = detect(removal_model, detection_model, device, half)

        delete_images('images_to_infer/')

        return jsonify(label=paths, time=time.time() - execution_time), 200
    except Error as e:
        print(e)
        delete_images('images_to_infer/')
        return jsonify(label="Error"), 400


def download_image_ipg(urls, file_path):
    for i, url in enumerate(urls):
        full_path = file_path + format(i, '04d') + '.jpg'
        r = requests.get(url)
        with open(full_path, 'wb') as outfile:
            outfile.write(r.content)


def delete_images(path):
    files = glob.glob(path + '*')
    for f in files:
        os.remove(f)


if __name__ == '__main__':
    # Initialize
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA
    detection_model = attempt_load('models/detection/best.pt', map_location=device)  # load FP32 model
    if half:
        detection_model.half()  # to FP16
    removal_model = WatermarkRemoval()
    paths = detect(removal_model, detection_model, device, half)
    # app.run(debug = True, host = '0.0.0.0', port = 5002)
