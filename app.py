"""
Diabetic Retinopathy prototype - Flask back end.
Run:  python app.py      then open http://127.0.0.1:5000
"""
import base64
import os

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

import dr_utils as U

MODEL_PATH = os.environ.get('DR_MODEL', 'best_dr_model.pth')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model, ck = U.load_checkpoint(MODEL_PATH, DEVICE)          # architecture + weights + metadata
ARCH, SIZE, NAMES = ck['arch'], ck['img_size'], ck['class_names']

# Plain-language, non-diagnostic notes shown with each stage.
NOTES = {
    0: 'No signs of diabetic retinopathy detected. Routine screening should continue as advised.',
    1: 'Early changes such as microaneurysms may be present. Follow-up screening is usually recommended.',
    2: 'More widespread changes are likely. An eye-care specialist should review this eye.',
    3: 'Extensive changes are likely. Prompt specialist review is important.',
    4: 'Signs of advanced disease are likely. Urgent specialist review is important.',
}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024        # 10 MB upload limit


def to_data_url(rgb):
    ok, buf = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return 'data:image/png;base64,' + base64.b64encode(buf).decode()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/model')
def model_info():
    return jsonify(arch=ARCH, size=SIZE, classes=len(NAMES), device=DEVICE,
                   test_accuracy=ck.get('test_accuracy'), test_kappa=ck.get('test_kappa'))


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if file is None or file.filename == '':
        return jsonify(error='Choose a retinal image first.'), 400
    try:
        original = np.array(Image.open(file.stream).convert('RGB'))
    except Exception:
        return jsonify(error='That file could not be read as an image. Use a JPG or PNG.'), 400

    processed = U.preprocess_fundus(original, SIZE)                  # same pipeline as training
    x = U.EVAL_TF(processed).unsqueeze(0).to(DEVICE)
    cam, idx, probs = U.gradcam(model, ARCH, x)                      # prediction + explanation in one pass

    return jsonify(
        stage=idx, name=NAMES[idx], confidence=float(probs[idx]),
        probabilities=[{'name': n, 'p': float(p)} for n, p in zip(NAMES, probs)],
        note=NOTES[idx],
        enhanced=to_data_url(processed),
        heatmap=to_data_url(U.overlay_cam(processed, cam)),
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(error='The image is larger than 10 MB. Choose a smaller file.'), 413


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
