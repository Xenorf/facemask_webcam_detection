import numpy as np
import cv2
import tensorflow as tf

# Path to the file that recognize the position of the face.
face_cascade = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

# Loading the previously trained model
loaded_model = tf.keras.models.load_model("masked.h5")

# Infinite while loop to print the webcam flow until input a break command
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detection of all the faces in the grey picture
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(200, 200))
    toprint = "NOTHING"
    # Iterating on the faces in the picture
    for (x, y, w, h) in faces:
        # Determining the dimensions of the picture
        r = max(w, h) / 2
        centerx = x + w / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)
        # Assining the face to a variable to resize and normalize it
        faceimg = frame[ny:ny+nr, nx:nx+nr]
        resized = cv2.resize(faceimg, (128, 128))
        normalized = resized/255.0
        reshaped = np.reshape(normalized, (1, 128, 128, 3))
        reshaped = np.vstack([reshaped])
        # Doing the prediction with the trained model on the current face and applying a label and a color
        (withoutMask, mask) = loaded_model.predict(reshaped)[0]
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        # Formating the printing of the prediction on the picture
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Display the image
    cv2.imshow('frame', frame)

    # Break condition to stop the script
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
