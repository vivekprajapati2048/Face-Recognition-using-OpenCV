# Face-Recognition-using-OpenCV
This repository consists of Face Recognition using OpenCV in Python in which the model can recognize the face from image and realtime via webcam.

### Contents:  
* **trainingImages** folder contains the images for classifier.  
If you want to classifier to recognize the multiple people the add each persons image in a seperated folder markerd by seperate labels. Then add this labels in *tester.py* and *videoTester.py* script in ***'name'*** variable.  
* **TestImages** folder contains the images that you want to predict
* **tester.py** script for predicting label to the test images
* **videoimg.py** to generate test images for training classifier  
* **videoTester.py** for predicting face realtime via webcam.

### Running the test: 
Give the path of image in ***'test_img'*** variable in *tester.py*.  
Run **tester.py** script in command line to train the recognizer on training images and also to predict test images.  
Make sure that you run tester.py  first since it generates ***'training.yml'*** file that is being used in *videoTester.py* script.

#### Reference : [pythonprogramming.net](https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/)
