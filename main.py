import os
import cv2
import pickle
import face_recognition
import firebase_admin
import numpy as np
import cvzone
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime

#AUTHENTICATION TO ACCESS FIREBASE AND INITIALIZATION
cred = credentials.Certificate("serviceAccountKey.json")    #"serviceAccountKey.json" is the file provided by firebase which authenticates us to access the database
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://automated-attendance-sys-187b7-default-rtdb.firebaseio.com/",
    'storageBucket': "automated-attendance-sys-187b7.appspot.com"
})

bucket = storage.bucket()

#VIDEO CAPTURE OBJECT SETUP
cap = cv2.VideoCapture(0)   #Video capture object
cap.set(3, 640) #Width
cap.set(4, 480) #Height

imgBackground = cv2.imread('Resources/background.png')

#IMPORTING THE MODE IMAGES INTO A LIST
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)   #List of everything inside Resources/Modes
imgModeList = []    #Empty list of final images
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))  #Traversing every element in modePathList

#LOAD THE ENCODING FILE CREATED IN "EncodeGenerator.py"
print("Loading Encode File...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIDs = pickle.load(file)
file.close()
encodeListKnown, studentIDs = encodeListKnownWithIDs
print("Encode File Loaded!")

modeType = 0    #Initially mode image will be the 0th mode ("active")
counter = 0     #Counts frames to change modes accordingly
id = -1         #ID of the student whose face has been recognised
imgStudent = []

#DISPLAY
while True:
    success, img = cap.read()

    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25) #Scaling down the size of captured image to 1/4th original value (reduces computations)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)    #Ensuring captured image is in RGB format

    faceCurrFrame = face_recognition.face_locations(imgSmall) #Location of faces in captured image
    encodeCurrFrame = face_recognition.face_encodings(imgSmall, faceCurrFrame)  # Encoding of face in captured image

    imgBackground[162:162+480, 55:55+640] = img #Show the webcam frames superimposed onto imgBackground
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    #IF FACE IS DETECTED IN THE CAPTURED FRAME
    if faceCurrFrame:
        #COMPARING ENCODINGS OF FACES FROM IMAGES FOLDER AND REALTIME VIDEO CAPTURE
        for encodeFace, faceLocation in zip(encodeCurrFrame, faceCurrFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)   #List of encoding of all the faces, present in both the captured image and input image
            faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)  #List of distance of all the faces, present in both the captured image and input image

            matchIndex = np.argmin(faceDistance)    #Minimum of all the distances in recognized faces is considered

            #DRAWING RECTANGLE AROUND RECOGNISED FACE
            if matches[matchIndex]: #if True
                y1, x2, y2, x1 = faceLocation
                y1, x2, y2, x1 = 4*y1, 4*x2, 4*y2, 4*x1
                x1, y1, x2, y2 = 55+x1, 162+y1, 55+x2, 162+y2
                #bbox = 55+x1, 162+y1, x2-x1, y2-y1
                #imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                imgBackground = cv2.rectangle(imgBackground, (x1,y1), (x2,y2), (0, 255, 0), 3)

                id = studentIDs[matchIndex] #ID of recognized student

                if counter == 0:
                    #cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                    #cv2.imshow("Face Attendance", imgBackground)
                    #cv2.waitKey(1)
                    counter = 1     #Face has been recognised so, now counter is incremented account for next frame
                    modeType = 1    #Changing mode to "student info" mode from "active" mode

        if counter != 0:
            if counter == 1:
                #GETTING THE DATA FROM DATABASE
                studentInfo = db.reference(f'Students/{id}').get()

                #GETTING THE IMAGE FROM THE STORAGE IN DATABASE
                bucket = storage.bucket()
                blob = bucket.get_blob(f'Images/{id}.png')
                array = np.frombuffer(blob.download_as_bytes(), np.uint8)   #Downloading image of student from database as a string
                imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

                #UPDATE ATTENDANCE
                datetimeObject = datetime.strptime(studentInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now()-datetimeObject).total_seconds()    #Time difference (in seconds) between now and last attendance marked for the same student

                if secondsElapsed > 10:
                    ref = db.reference(f'Students/{id}')
                    studentInfo['total_attendance'] += 1
                    ref.child('total_attendance').set(studentInfo['total_attendance'])
                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                else:
                    modeType = 3
                    counter = 0
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

            if modeType != 3:

                if 10 < counter < 20:
                    modeType = 2

                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                if counter <= 10:
                    #DISPLAY THE DATA FROM DATABASE ON SCREEN
                    cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)   #Displays 'total_attendance' on the screen, fetched from the database
                    cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1) #Displays 'major' on the screen, fetched from the database
                    cv2.putText(imgBackground, str(id), (1006, 493), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1) #Displays 'id' on the screen
                    cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)    #Displays 'standing' on the screen, fetched from the database
                    cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1) #Displays 'year' on the screen, fetched from the database
                    cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)  #Display 'starting_year' on the screen, from the database

                    (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)    #Identifying width, height of 'name' fetched from database for given font (third returned parameter is insignificant to us)
                    offset = int((414-w)/2)  #Centering 'name' display
                    cv2.putText(imgBackground, str(studentInfo['name']), (808+offset, 445), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)  # Display 'name' on the screen, fetched from the database

                    imgBackground[175:175+216, 909:909+216] = imgStudent

                counter += 1

                if counter >= 20:
                    counter = 0
                    modeType = 0
                    studentInfo = []
                    imgStudent = []
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    else:
        modeType = 0
        counter = 0

    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)  #Displays the particular window for 'parameter' milliseconds