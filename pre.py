import os
import cv2 as cv
from os import listdir
from os.path import join

# illumination check
# rotation (?)
# ????+
# color space transport

# https://www.tensorflow.org/tutorials/quickstart/beginner
# https://www.tensorflow.org/tutorials/images/cnn

target="detected-faces-test"
dataset="dataset/casia-webface"

def detectAndStore(img, folder_name, file_name):
    # Make sure our 'detecteded faces' folder exists, otherwise create it
          
    # Detect faces
    faces = face_cascade.detectMultiScale(img)

    # Draw rectangles
    for i, (x, y, w, h) in enumerate(faces):
        # cv.rectangle(img, (x-1, y-1), (x+w+1, y+h+1), (255, 0, 0), 2)
        scale = 5
        face = img[y-scale:y + h+scale, x-scale:x + w+scale]
        detected_face = face.copy()
        try:
            resized_face = cv.resize(detected_face, (64,64))
            resized_face = cv.cvtColor(resized_face, cv.COLOR_BGR2GRAY)
            resized_face = cv.equalizeHist(resized_face)
            print(folder_name + " " + file_name)
            cv.imwrite(f'{target}/{folder_name}/{file_name}-{i}.jpg', resized_face)
        except Exception as e:
            print(f"failed image {folder_name} {file_name}")
            return
    # cv.imshow('Capture - Face detection', img)

# Load cascades
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Make sure our 'detecteded faces' folder exists, otherwise create it
if not os.path.exists(target):
    print(f'made \'{target}\' directory')
    os.makedirs(target)

# Setup where to read images from
path = dataset
dirs = listdir(path)
dirs.sort()
dirs = dirs[:9000]

for dir in dirs:
    folder_name = dir

    folder_path = join(path, dir)
    if not os.path.exists(f"{target}/{folder_name}"):
        print(f"made \'{folder_name}\' directory")
        os.makedirs(f"{target}/{folder_name}")
    else:
        # Folder is already generated
        print(f"Skipping folder {folder_name}, already exists")
        continue
        
    files = listdir(folder_path)
     
    print(f"Processing: {folder_path}")
    for file in files:
        file_path = join(folder_path, file)
        file_name = os.path.splitext(file)[0]
        image = cv.imread(file_path)
        detectAndStore(image, folder_name, file_name)

