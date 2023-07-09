# Face-Detection-model
This repository contains a real-time face detection model implemented using TensorFlow and OpenCV. The model can accurately detect and track human faces in images and video streams, providing bounding box coordinates and confidence scores.

Description

The face detection model utilizes deep learning techniques to detect faces in real-time. It leverages the power of TensorFlow for efficient model training and inference, and OpenCV for capturing video streams and visualizing the results.

The model is based on a state-of-the-art convolutional neural network architecture, which has been trained on a large-scale face detection dataset. It is capable of detecting faces with high precision and can handle various real-world scenarios, such as different lighting conditions, angles, and facial expressions.

It consists of a custom FaceTracker model class that extends the tf.keras.Model class. The model is trained using a combination of classification and regression loss functions.

The repository also includes scripts for training the model, evaluating its performance, and applying real-time face detection on video streams.

This project has used matplotlib, keras, tensorflow, CNN and other concepts including Image augmentation. The deep learning model was built using different layers like input, Conv2D, Dense and GlobalMaxPooling2D. Furthermore, it made use of Vgg to get rid of the layers that we do not require in the model. 

The neural network was trained using a custom model class and then its performance was plotted to view the results. Lastly, predictions were made on the test set, I saved the model and then finally did a real-time face detection. 

This project helped me in expanding my knowledge in the deep learning domain as I learnt about how we can use pre-trained models, to create and train models of our own. Also, learnt a lot of interesting things which can be done using Python programming language. 
