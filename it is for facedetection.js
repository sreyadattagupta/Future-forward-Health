import os
import cv2
import pickle
import face_recognition
import numpy as np
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://facedetectionrealtime-d062a-default-rtdb.firebaseio.com/",
    'storageBucket':"facedetectionrealtime-d062a.appspot.com"
})

bucket = storage.bucket()


cap = cv2.VideoCapture(1)
cap.set(2,640)
cap.set(3,720)

imgBackground = cv2.imread("Resources/gr1.png")
imgHeading = cv2.imread("Resources/m1.jpg")
imgTop = cv2.imread("Modes/m7.jpg")
imgInfo = cv2.imread("Modes/m8.jpg")

#Importing the modes images into a lists

folderModePath = 'Modes'
modePathList = os.listdir(folderModePath)
imgModeLists = []
for path in modePathList:
    imgModeLists.append(cv2.imread(os.path.join(folderModePath,path)))

#print(len(imgModeLists))
#print(modePathList)


#Load the encoding file
print("Loading Encoded File....")

file = open('EncodeFile.p','rb')
encodeListknownWithIds = pickle.load(file)
file.close()
encodeListknown,studentIds = encodeListknownWithIds
#print(studentIds)
print("Encoded File Loaded....")


modeType = 0
counter = 0
id = -1
imgStudent =[]



while True:
    success, img = cap.read()

    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    imgBackground[100:100+480,0:0+640] = img
    imgBackground[0:0+100,0:0+640] = imgHeading
    imgBackground[0:0+150,700:700+300] = imgTop
    #imgBackground[700:700+250, 230:230+200] = imgInfo
    imgBackground[190:190+150, 700:700+300] = imgModeLists[modeType]

    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurrFrame,faceCurFrame):
            matches = face_recognition.compare_faces(encodeListknown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListknown,encodeFace)
            #print("matches",matches)
            #print("faceDis", faceDis)


            matchIndex = np.argmin(faceDis)
            #print("Match Index",matchIndex)

            if matches[matchIndex]:
                #print("Known Face Detected")
                #print(studentIds[matchIndex])
                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 =y1*4,x2*4,y2*4,x1*4
                bbox = 0 + x1, 100+y1, x2 - x1, y2 - y1
                imgBackground = cvzone.cornerRect(imgBackground,bbox, rt=0)
                id = studentIds[matchIndex]
                if counter == 0:
                    cvzone.putTextRect(imgBackground,"Loading......",(300,350))
                    cv2.imshow("Face Attandance",imgBackground)
                    cv2.waitKey(1)
                    counter = 1
                    modeType = 1

        if counter != 0:

            if counter == 1:
                #Getting the data
                studentInfo = db.reference(f'Students/{id}').get()
                print(studentInfo)

                #Getting the image from the storage
                blob = bucket.get_blob(f'Images/{id}.jpg')
                array = np.frombuffer(blob.download_as_string(),np.uint8)
                imgStudent = cv2.imdecode(array,cv2.COLOR_BGRA2BGR)

                #Update the data of presence
                datetimeObject = datetime.strptime(studentInfo['last_seen'],
                                              "%Y-%m-%d  %H:%M:%S")
                secondElapsed = (datetime.now()-datetimeObject).total_seconds()
                print(secondElapsed)
                if secondElapsed >30:
                    ref = db.reference(f'Students/{id}')
                    studentInfo['total_attendance'] +=1
                    ref.child('total_attendance').set(studentInfo['total_attendance'])
                    ref.child('last_seen').set(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
                else:
                    modeType = 3
                    counter = 0
                    imgBackground[190:190 + 150, 700:700 + 300] = imgModeLists[modeType]


            if modeType !=3:


                if 10<counter<20:
                    modeType = 2

                    imgBackground[190:190 + 150, 700:700 + 300] = imgModeLists[modeType]


                if counter <=10:

                    cv2.putText(imgBackground,str(studentInfo['total_attendance']),(850,90),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1)
                    cv2.putText(imgBackground, str(studentInfo['Name']), (750,450 ),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo['Department']), (770, 480),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                    cv2.putText(imgBackground, str(id), (800, 425),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

                #imgBackground[700:700+150, 700:700+300] = imgStudent

                counter +=1

                if counter>=20:
                    counter = 0
                    modeType = 0
                    studentInfo = []
                    imgStudent = []
                    imgBackground[190:190 + 150, 700:700 + 300] = imgModeLists[modeType]


    else:
        modeType = 0
        counter = 0



    cv2.imshow("Face Attandance",imgBackground)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break