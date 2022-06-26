# importing libraries
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os


# directory of which face detection model
directory = "Face_detector"
trained_model = "mask_detector.model"

# loading face detection model
model_weightsPath = os.path.sep.join(
    [directory, "res10_300x300_ssd_iter_140000.caffemodel"])
model_prototxtPath = os.path.sep.join([directory, "deploy.prototxt"])
net = cv2.dnn.readNet(model_prototxtPath, model_weightsPath)
model = load_model(trained_model)


class Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, image = self.video.read()
        orig = image.copy()
        (h, w) = image.shape[:2]

        # construct the blob
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # passing the blob
        net.setInput(blob)
        detections = net.forward()

        # looping over the detections
        for i in range(detections.shape[2]):
            # condifence
            # setting the confidence threshold value to 0.5
            confidence = detections[0, 0, i, 2]

            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensuring the box falls within the frame
                (startX, startY) = (max(0, startX), max(9, startY))
                (endX, endY) = (min(w-1, endX), min(h-1, endY))

                # face = image[startY:endY, startX:endX]
                # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = image[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                # predictions
                (mask, withoutMask) = model.predict(face)[0]

                # making the label and color for the 2 categories
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # displaying the text
                cv2.putText(image, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

        ret, jpg = cv2.imencode('.jpg', image)
        return jpg.tobytes()
