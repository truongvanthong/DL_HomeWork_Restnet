from flask import Flask, render_template, Response, request
import cv2
import tensorflow as tf
from mtcnn.detector import MtcnnDetector
import numpy as np
import base64

app = Flask(__name__)

# Singleton instances
detector = None
model = None

def get_detector():
    global detector
    if detector is None:
        detector = MtcnnDetector()
    return detector

def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model('resnet50_w4.model')
    return model

def detect_and_predict_mask(frame):
    h, w = frame.shape[:2]
    boxes, _ = get_detector().detect_faces(frame)
    for box in boxes:
        startX, startY, endX, endY = box[:4].astype('int')
        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(w - 1, endX), min(h - 1, endY)

        face = frame[startY:endY, startX:endX]
        face = cv2.resize(face, (224, 224))
        mask, withoutMask = get_model().predict(face.reshape(1, 224, 224, 3))[0]

        label = 'Mask' if mask > withoutMask else 'No Mask'
        color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    return frame

def generate_frames():
    vs = cv2.VideoCapture(0)
    while True:
        success, frame = vs.read()
        if not success:
            break
        frame = detect_and_predict_mask(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/load_image', methods=['GET', 'POST'])
def load_image():
    results = []
    if request.method == 'POST':
        uploaded_images = request.files.getlist('image')
        for uploaded_image in uploaded_images:
            if uploaded_image:
                npimg = np.fromstring(uploaded_image.read(), np.uint8)
                img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                result_img = detect_and_predict_mask(img)
                _, buffer = cv2.imencode('.jpg', result_img)
                encoded_image = buffer.tobytes()
                encoded_image_str = "data:image/jpeg;base64," + base64.b64encode(encoded_image).decode('utf-8')
                results.append(encoded_image_str)
    return render_template('load_image.html', results=results)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)