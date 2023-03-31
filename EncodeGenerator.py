import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://automated-attendance-sys-187b7-default-rtdb.firebaseio.com/",
    'storageBucket': "automated-attendance-sys-187b7.appspot.com"
})

#IMPORTING THE STUDENT IMAGES
folderPath = 'Images'
imgPathList = os.listdir(folderPath)
imgList = []
studentIDs = []

print("Uploading 'Images' Folder...")
for path in imgPathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))  #Creating an image element in imgList for every element in imgPathList
    #print(imgList)
    studentIDs.append(os.path.splitext(path)[0])    #List of student IDs from images
    #print(studentIDs)

    bucket = storage.bucket()   #Provides access to the only/default bucket available in firebase
    fileName = f'{folderPath}/{path}'   #Address(es) of 'Images' folder and the image(s) within both, the computer and the only/default bucket in firebase
    d = bucket.blob(fileName)   #Creates reference object 'd' to specific blob(file) within a folder and if file/folder or both are not present, it is created/added/appended
    d.upload_from_filename(fileName)    #Uploading images from 'fileName' in computer to 'fileName' in bucket using blob reference object 'd'
print("'Images' Folder Uploaded!")

def findEncodings(imagesList):
    encodeList = []

    for img in imgList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #Making sure image is in RGB format
        encode = face_recognition.face_encodings(img)[0]    #Obtaining image encodings
        #Will show 'list index out of range' in case face isn't recognised in even one of the images
        encodeList.append(encode)   #List of all image encodings

    return encodeList

print("\nEncoding Started...")
#print(imgList)
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIDs = [encodeListKnown, studentIDs]
print("Encoding Complete!")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIDs, file)
file.close()
print("File Saved!")