# import modules

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC


# read data
train_data = np.loadtxt('Processed_Data/NN_1/Train_data.txt')
train_labels = np.loadtxt('Processed_Data/NN_1/Train_labels.txt')
test_data = np.loadtxt('Processed_Data/NN_1/Test_data.txt')
test_labels = np.loadtxt('Processed_Data/NN_1/Test_labels.txt')


# fit k-NN model
model = KNeighborsClassifier(n_neighbors=1)
# model = SVC()
model.fit(train_data, train_labels)



# print accuracy
print("TRAINING DONE!")
print(model.score(test_data, test_labels))


#Predict New Data
#model.predict(array)
