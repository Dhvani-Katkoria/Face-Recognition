import face_recognition
import cv2
import numpy as np
from skimage import io
import os

face_cascade = cv2.CascadeClassifier('/home/VR/GA2/haarcascade_frontalface_default.xml')

dirs = next(os.walk('/media/vidhikatkoria/Research/VR/GP1/AVR_data-20200320T104848Z-001/AVR_data/'), ([],[],[]))[1]
for name in dirs:
    name=str(name)
    os.mkdir("/media/vidhikatkoria/Research/VR/GP1/AVR_data-20200320T104848Z-001/ProcessedFaces1/" + name + "/")
    path = '/media/vidhikatkoria/Research/VR/GP1/AVR_data-20200320T104848Z-001/AVR_data/' + name + '/'
    face_img = os.listdir(path)
#     print(face_img)
    for i in range(len(face_img)):
        print(face_img[i])
        image = cv2.imread(path + face_img[i])
        # load image and find face locations.
        ###################
        # detect 68-landmarks from image. This includes left eye, right eye, lips, eye brows, nose and chins
        face_landmarks = face_recognition.face_landmarks(image)
        if(len(face_landmarks)!=0):
            # Let's find and angle of the face. First calculate the center of left and right eye by using eye landmarks.
            leftEyePts = face_landmarks[0]['left_eye']
            rightEyePts = face_landmarks[0]['right_eye']
            nosetip = face_landmarks[0]['nose_tip']
            nosebridge = face_landmarks[0]['nose_bridge']

            leftEyeCenter = np.array(leftEyePts).mean(axis=0).astype("int")
            rightEyeCenter = np.array(rightEyePts).mean(axis=0).astype("int")
            nosecenter = np.array(nosetip).mean(axis=0).astype("int")
            nosebridge = np.array(nosebridge).mean(axis=0).astype("int")

            leftEyeCenter = (leftEyeCenter[0], leftEyeCenter[1])
            rightEyeCenter = (rightEyeCenter[0], rightEyeCenter[1])
            nosecenter = (nosecenter[0], nosecenter[1])
            nosebridge = (nosebridge[0], nosebridge[1])

            """display facial marks"""
            # image1=image.copy()
            # # draw the circle at centers and line connecting to them
            # le = cv2.circle(image1, leftEyeCenter, 2, (0, 255, 0), 30)
            # re = cv2.circle(image1, rightEyeCenter, 2, (0, 255, 0), 30)
            # eyeline = cv2.line(image1, leftEyeCenter, rightEyeCenter, (255, 0, 0), 10)
            # nt = cv2.circle(image1, nosecenter, 2, (0, 0, 255), 30)
            # nt = cv2.circle(image1, nosebridge, 2, (0, 0, 255), 30)
            # cv2.imshow("1",cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
            # cv2.waitKey(0)

            # find and angle of line by using slop of the line.
            dY = rightEyeCenter[1] - leftEyeCenter[1]
            dX = rightEyeCenter[0] - leftEyeCenter[0]
            angle = np.degrees(np.arctan2(dY, dX))

            # to get the face at the center of the image, set desired left eye location(%). Right eye location will be found out by using left eye location.
            desiredLeftEye = (0.25, 0.25)
            desiredRightEyeX = 1.0 - desiredLeftEye[0]
            # Set the croped image(face) size after rotaion.
            desiredFaceWidth = 128
            desiredFaceHeight = 128

            # determine the scale of the new resulting image by taking the ratio of the distance between eyes in the *current* image
            # to the ratio of distance between eyes in the *desired* image
            dist = np.sqrt((dX ** 2) + (dY ** 2))
            desiredDist = (desiredRightEyeX - desiredLeftEye[0])
            desiredDist *= desiredFaceWidth  # c = c * a
            scale = desiredDist / dist
            # compute center (x, y)-coordinates (i.e., the median point)
            # between the two eyes in the input image
            eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                          (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

            # grab the rotation matrix for rotating and scaling the face
            M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
            # update the translation component of the matrix
            tX = desiredFaceWidth * 0.5
            tY = desiredFaceHeight * desiredLeftEye[1]
            M[0, 2] += (tX - eyesCenter[0])
            M[1, 2] += (tY - eyesCenter[1])
            # apply the affine transformation
            (w, h) = (desiredFaceWidth, desiredFaceHeight)
            output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

            ##########################
            face_landmarks = face_recognition.face_landmarks(output)
            if(len(face_landmarks)!=0):
                nosebridge = face_landmarks[0]['nose_bridge']
                nosebridge = np.array(nosebridge).mean(axis=0).astype("int")
                nosebridge = (nosebridge[0], nosebridge[1])
            else:
                print("can't calculate nose bridge, hence taking default values")
                nosebridge = (128,100)


            a = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            mask = np.zeros([128, 128], np.uint8)
            mask = cv2.ellipse(mask, (nosebridge[0], nosebridge[1]), (110, 150), 0, 0, 360, (127, 127, 127), -1)
            c = np.bitwise_and(a, mask)

            cv2.imwrite("/media/vidhikatkoria/Research/VR/GP1/AVR_data-20200320T104848Z-001/ProcessedFaces1/"+name+"/"+str(i+1)+".png",c)
            print(i+1)