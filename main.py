import cv2
import numpy as np
import os
import face_recognition
from datetime import datetime

# Path to the directory containing images
path = 'ImageAttendance'

# Load images from the directory
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    if cl != '.DS_Store':
        curImage = cv2.imread(f'{path}/{cl}')
        images.append(curImage)
        classNames.append(os.path.splitext(cl)[0])


# Encode the images for face recognition
def findEncoding(toEncodeimages):
    encodeList = []
    for img in toEncodeimages:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncoding(images)

# Load the prerecorded video file
video_path = 'Anthem of Morocco 1.MOV'
cap = cv2.VideoCapture(video_path)

# Check if the video file was successfully opened
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dtString}')


# Main loop to process video frames
while True:
    # Read frame from the video file
    success, img = cap.read()

    # Check if frame is valid
    if not success:
        break

    # Resize the frame
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Perform face detection and recognition
    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    # Display the frame
    cv2.imshow('Video', img)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video file handle and close windows
cap.release()
cv2.destroyAllWindows()
