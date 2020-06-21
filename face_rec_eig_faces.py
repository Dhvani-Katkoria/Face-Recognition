#Face recognition with pca implementation 

import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
import scipy.stats
import os
import numpy as np

X  = []#data
label_names = next(os.walk('/media/vidhikatkoria/Research/VR/GP1/AVR_data-20200320T104848Z-001/ProcessedFaces/'), ([],[],[]))[1]
labels=[]
height = 256
widht = 256
for i in range(len(label_names)):
    name=label_names[i]
    face_img = os.listdir("/media/vidhikatkoria/Research/VR/GP1/AVR_data-20200320T104848Z-001/ProcessedFaces/" + name + "/")
    for j in face_img:
        labels+=[i+1]
        im = cv2.imread("/media/vidhikatkoria/Research/VR/GP1/AVR_data-20200320T104848Z-001/ProcessedFaces/"+ name + "/" + j,0)
        im2 = cv2.resize(im,(height,widht))
        p=(im2/255).flatten()
        X.append(p)       #set consisting of flattened images

# original_face = X[20].reshape(256,256)
# cv2.imshow("original_face",original_face)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#PCA
#computing mean of data
X_mean = np.mean(X, axis=0)

# mean_face = X_mean.reshape(256,256)
# cv2.imshow("mean_face",mean_face)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

matrix = X - X_mean               # mean faces 
trans_matrix = matrix.T

# zm_images = trans_matrix[:,20].reshape(256,256)
# cv2.imshow("zero_mean_face",zm_images)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

mul_m = matrix.dot(trans_matrix)
cov_matrix = mul_m/(matrix.shape[0]) #finding the covaraince matrix

# Compute the eigen values and vectors using numpy
e_val, e_vec = np.linalg.eig(cov_matrix)
img_e_vec = trans_matrix.dot(e_vec)
reduced_dim =  25        # number of eigen faces retained
nor_vec = normalize(img_e_vec,norm = 'l1')
eigen_faces = nor_vec[:, :reduced_dim]
pca_vectors = matrix.dot(eigen_faces)      #reconstruct faces using eigen faces
pca_vectors = pca_vectors.astype('float64')
print(np.transpose(eigen_faces).shape)
reconstructed_faces = np.dot(pca_vectors, np.transpose(eigen_faces))
# r_face = reconstructed_faces[20]
# minf = np.min(r_face)
# maxf = np.max(r_face)
# r_face = r_face-float(minf)
# r_face = r_face/float((maxf-minf))
# r_face = r_face+np.transpose(X_mean)
# rf = r_face.reshape(256,256)
# cv2.imshow("r_face",rf)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#classification
X_train, X_test, y_train, y_test = train_test_split(pca_vectors, labels, random_state=42) #splitting test data
knn = KNeighborsClassifier(n_neighbors=10) #using a K means Classifier
knn.fit(X_train, y_train) #training data
print('Knn accuracy', knn.score(X_test, y_test)) #applying model on test data

# for calculation of top-n accuracy
def predict(X_train, y_train, X_test, y_test, n):
    # create list for distances and targets
    distances = []
    targets = []

    for i in range(len(X_train)):
        # compute and store L2 distance
        distances.append([np.sqrt(np.sum(np.square(X_test - X_train[i, :]))), i])

    # sort the list
    distances = sorted(distances)

    # make a list of the k neighbors' targets
    for i in range(n):
        index = distances[i][1]
        targets.append(y_train[index])

    if y_test in targets:
        return y_test
    else:
        return targets[0]

y_pred = []
for i in range(len(X_test)):
        y_pred.append(predict(X_train, y_train, X_test[i, :], y_test[i],1))
# for val in X_test:
#     k_index = knn.kneighbors(val,n_neighbors=1, return_distance=False)
#     for indx in k_index:
#         y_nTest.append(y_test[k_index])
    
print('Top accuracy ', accuracy_score(y_test,y_pred))     
