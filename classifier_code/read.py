import numpy as np
import os
import pickle
import cv2
import random
import tensorflow as tf

### Directory with faces
directory = "./cacd_faces"
sz = 100

categories = os.listdir(directory)
print(categories)
training_data = []

for category in categories:
	path = os.path.join(directory,category)
	class_num = categories.index(category)
	for img in os.listdir(path):
		try:
			img_array = cv2.imread(os.path.join(path,img))
			new_array = cv2.resize(img_array,(sz,sz))
			training_data.append([new_array, class_num])
		except Exception as e:
			pass

random.shuffle(training_data)
X_train = []
y_train = []
X_validate = []
y_validate = []
X_test = []
y_test = []

### Split data in training, validate and test in ratio 75:10:15
train,validate,test = np.split(training_data,[int(.75 * len(training_data)), int(.85 * len(training_data))])


for features, label in train:
		X_train.append(features)
		y_train.append(label)

for features, label in validate:
		X_validate.append(features)
		y_validate.append(label)

for features, label in test:
		X_test.append(features)
		y_test.append(label)

X_validate = np.array(X_validate).reshape(-1,sz,sz,3)
X_train = np.array(X_train).reshape(-1,sz,sz,3)
X_test=np.array(X_test).reshape(-1,sz,sz,3)

print(X_train.shape)
pickle_out = open("X_train.pickle","wb")
pickle.dump(X_train,pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle","wb")
pickle.dump(y_train,pickle_out)
pickle_out.close()

print(X_validate.shape)
pickle_out = open("X_validate.pickle","wb")
pickle.dump(X_validate,pickle_out)
pickle_out.close()

pickle_out = open("y_validate.pickle","wb")
pickle.dump(y_validate,pickle_out)
pickle_out.close()


print(X_test.shape)
pickle_out = open("X_test.pickle","wb")
pickle.dump(X_test,pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle","wb")
pickle.dump(y_test,pickle_out)
pickle_out.close()