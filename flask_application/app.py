import numpy as np
import cv2
import imutils
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

LABEL = {0: "MASK OFF",
         1: "MASK ON MOUTH",
         2: "MASK OFF",
         3: "MASK ON"}

app = Flask(__name__)
camera = cv2.VideoCapture(0)
clf = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('mask_detector_aug3.model')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detect_mask(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def detect_mask(frame):
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(gray, 1.1, 4)
    faces_dict = {"faces_list": [], "faces_rect": []}

    for rect in faces:
        (x, y, w, h) = rect
        face_frame = frame[y:y + h, x:x + w]
        face_frame_prepared = preprocess_face_frame(face_frame)
        faces_dict["faces_list"].append(face_frame_prepared)
        faces_dict["faces_rect"].append(rect)

    if faces_dict["faces_list"]:
        faces_preprocessed = preprocess_input(np.array(faces_dict["faces_list"]))
        preds = model.predict(faces_preprocessed)

        for i, pred in enumerate(preds):
            label = LABEL[np.argmax(pred)]
            (x, y, w, h) = faces_dict["faces_rect"][i]
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return frame


def preprocess_face_frame(face_frame):
    face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)  # RGB
    face_frame_resized = cv2.resize(face_frame, (224, 224))  # resize for model input
    face_frame_array = img_to_array(face_frame_resized)  # convert to array
    return face_frame_array


if __name__ == '__main__':
    app.run(debug=True)
