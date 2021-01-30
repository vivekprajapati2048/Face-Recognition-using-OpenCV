import cv2
import os
import numpy as np
import faceRecognition as fr


# This module takes images  stored in disk and performs face recognition
test_img = cv2.imread('TestImages/sample.jpg') # test_img path
faces_detected,gray_img = fr.faceDetection(test_img)
print("faces_detected:", faces_detected)


faces, faceID = fr.labels_for_training_data('trainingImages')

# Comment belows lines when running this program second time. 
# Since it saves training.yml file in directory
face_recognizer = fr.train_classifier(faces, faceID)
face_recognizer.save('trainingData.yml')


# Uncomment below line for subsequent runs
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# Use this to load training data for subsequent runs
# face_recognizer.read('trainingData.yml')


# creating dictionary containing names for each label
name = {0: "Elon Musk", 1: "Bill Gates"}

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+h,x:x+h]
    
    # predicting the label of given image
    label, confidence = face_recognizer.predict(roi_gray)
    print("confidence:", confidence)
    print("label:", label)
    
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    
    # If confidence more than 37 then don't print predicted face text on screen
    # if(confidence>37):
    #     continue
    fr.put_text(test_img, predicted_name, x, y)

resized_img = cv2.resize(test_img, (500,500))
cv2.imshow("face dtecetion tutorial", resized_img)
# waits indefinitely until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows