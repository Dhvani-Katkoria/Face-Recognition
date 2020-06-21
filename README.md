# Face-Recognition
Implement a face recognition system by extracting relevant features from the images provided using PCA algorithm, LDA, and LBP algorithm.

The following steps are performed for data pre-processing:
1. Detect the faces in the image using Haar Cascades Classifier.
2. Detect the facial landmarks and extract the facial features.
3. Rotate and scale the image.
4. Using the extracted facial features create an oval region corresponding to the face and reset all pixels outside the oval region to a constant value.

Following steps are used to perform face recognition from the processed faces
stored as 256X256 images:
• PCA method implementation for face recognition
• Face recognition using existing LDA and LBP methods.
