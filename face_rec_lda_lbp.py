
#face recognition using existing implementation of lbp, lda, pca

import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os
import numpy as np
from sklearn.metrics import accuracy_score,classification_report

def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #detect face using Haar cascade classifier
    face_cascade = cv2.CascadeClassifier('/home/vidhikatkoria/VR/GA2/haarcascade_frontalface_default.xml')

    #detect images
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3);
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    #extract the face area
    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

#this function will read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list 
# of faces and another list of labels for each face
def prepare_training_data(data_folder_path):
    
    dirs = os.listdir(data_folder_path)
    
    #list to hold all faces
    faces = []
    #list to hold labels for all faces
    labels = []
    
    for index, dir_name in enumerate(dirs):
    
        label = int(index)
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        #detect face and add face to list of faces
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue;
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            face, rect = detect_face(image)
            #ignore faces that are not detected
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
    return faces, labels

# def preprocessedd_data(data_folder_path):
#     X  = []#data
#     label_names = next(os.walk(data_folder_path), ([],[],[]))[1]
#     labels=[]
# #     height = 256
# #     widht = 256
#     for i in range(len(label_names)):
#         name=label_names[i]
#         face_img = os.listdir(data_folder_path + name + "/")
#         for j in face_img:
#             labels+=[i+1]
#             im = cv2.imread(data_folder_path+ name + "/" + j,0)
# #             im2 = cv2.resize(im,(height,widht))
#             p=(im/255).flatten()
#             X.append(p)
#     return X,labels

faces, labels = prepare_training_data("/media/vidhikatkoria/Research/VR/GP1/AVR_data-20200320T104848Z-001/ProcessedFaces")

X_train, X_test, y_train, y_test = train_test_split(faces, labels, random_state=42) #splitting test data
 
#resizing images for equal size image
height = 256
width = 256
train = []
for face in X_train:
    train.append(cv2.resize(face, (height, width)))
test = []
for face in X_test:
    test.append(cv2.resize(face, (height, width)))
    
#using LBPH method for face recognition
face_recognizer1 = cv2.face.LBPHFaceRecognizer_create()
face_recognizer1.train(train, np.array(y_train))
y_pred =[]
for face in test:
    label,confidence = face_recognizer1.predict(face)
    y_pred.append(label)
#     print(label)
print("LBPH accuracy : ",accuracy_score(y_test, y_pred))

#using LDA method for face recognition
face_recognizer2 = cv2.face.FisherFaceRecognizer_create()
face_recognizer2.train(train, np.array(y_train))
y_pred =[]
for face in test:
    label,confidence = face_recognizer2.predict(face)
    y_pred.append(label)
#     print(label)
print("LDA accuracy : ",accuracy_score(y_test, y_pred))

#using PCA method for face recognition
face_recognizer3 = cv2.face.EigenFaceRecognizer_create()
face_recognizer3.train(train, np.array(y_train))
y_pred =[]
for face in test:
    label,confidence = face_recognizer3.predict(face)
    y_pred.append(label)
#     print(label)
print("PCA accuracy : ",accuracy_score(y_test, y_pred))






