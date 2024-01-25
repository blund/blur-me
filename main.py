import os
import cv2 as cv

def detectAndDisplay(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.equalizeHist(img_gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(img_gray)

    # Draw rectangles
    for i, (x, y, w, h) in enumerate(faces):
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face = frame[y:y + h, x:x + w]
        cv.imshow("Cropped Face", face)
        cv.imwrite(f'detected-faces/face{i}.jpg', face)
        
    cv.imshow('Capture - Face detection', img)


# Load cascades
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
eyes_cascade = cv.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

if not os.path.exists("detected-faces"):
    print('made \'detected-faces\' directory')
    os.makedirs("detected-faces")

# Load video
cap = cv.VideoCapture(0, cv.CAP_V4L2)

if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

while True:
    # Read frame
    ret, frame = cap.read()

    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    # Detect and display
    detectAndDisplay(frame)
    cv.imshow('Capture - Face detection', frame)

    if cv.waitKey(10) == 27:
        break

