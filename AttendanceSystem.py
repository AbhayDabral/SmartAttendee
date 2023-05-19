#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime


# In[ ]:


#input the images

path='Images'
images=[]
classNames=[]
mylist=os.listdir(path)
for cls in mylist:
    currimg=cv2.imread(f'{path}/{cls}')
    images.append(currimg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)


# In[ ]:


#Encoding the images

def EncodeImg(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encodeimg=face_recognition.face_encodings(img)[0]
        encodelist.append(encodeimg)
    return encodelist

callencode=EncodeImg(images)
print("Encoding successful")

#Checking the attendance
 
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines()  # Avoids Duplicates
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()#current time
            dstring=now.strftime('%H:%M:%S')#time format
            f.writelines(f'\n{name},{dstring}')


# In[ ]:


#Taking input image from Webcam

cap=cv2.VideoCapture(0)
while True: 
    success, img=cap.read()
    camimg=cv2.resize(img,(0,0),None,0.25,0.25)
    camimg=cv2.cvtColor(camimg,cv2.COLOR_BGR2RGB)
    
    imgloc=face_recognition.face_locations(camimg)
    imgenc=face_recognition.face_encodings(camimg,imgloc)
    
    for encodeface,faceloc in zip(imgenc,imgloc):
        compare=face_recognition.compare_faces(callencode,encodeface)
        distance=face_recognition.face_distance(callencode,encodeface)
        #print(distance)
        matchIndex=np.argmin(distance)

        if compare[matchIndex]:
            name=classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
    
    
    
    cv2.imshow("WEBCAM",img)
    cv2.waitKey(0)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break

cap.release()
cv2.destroyAllWindows()


# 

# In[ ]:




