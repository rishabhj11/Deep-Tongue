from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import csv
import cv2
import random
import numpy as np

def partition_data(images, labels, data_name, ratio):
    training_set_size = int(len(images) * ratio)
    image_training_set, label_training_set = [], []
    images_copy = list(images)
    labels_copy = list(labels)
    data_name_copy = list(data_name)
    while len(image_training_set) < training_set_size:
        index = random.randrange(len(images_copy))
        image_training_set.append(images_copy.pop(index))
        label_training_set.append(labels_copy.pop(index))
        data_name_copy.pop(index)  # only names of test images left in the list
    return [image_training_set, images_copy, label_training_set, labels_copy, data_name_copy]

file_name = '1-20.csv'
partition_ratio = 0.75
data_set = csv.reader(open(file_name, "rb"))
image_set, label_set, image_set_name = [], [], []
for row in data_set:
    img = cv2.imread(row[0]+'.jpg')
    image_set.append(img)
    image_set_name.append(row[0]+'.jpg')
    label = int(row[1])
    label_set.append(label)
# print len(label_set)
X_train, X_test, y_train, y_test, test_set_name = partition_data(image_set, label_set, image_set_name, partition_ratio)
sample_list = y_test
# print ('y_train: ', y_train)
# print ('y_test: ', y_test)
X_train = np.array(X_train)
# print ('shape:', X_train.shape)
X_train = np.reshape(X_train, (len(y_train),3,256,256))
y_train = np.array(y_train)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (len(y_test),3,256,256))
y_test = np.array(y_test)
# print (X_test)
# print len(image_set)


batch_size = 32 # 128 images can be divided into batched of 32
nb_classes = 2 # two classes (0, 1)
nb_epoch = 5 # number of runs
data_augmentation = False

img_rows, img_cols = 256, 256
img_channels = 3

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
# print (Y_train)
model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols))) # first layer
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='Adam')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

if not data_augmentation:
   # print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test), 
              shuffle=True)

predicted = model.predict_classes(X_test, batch_size=32, verbose=1)
# print (predicted)
predicted_list = predicted.tolist()
print (predicted_list)
# print (type(predicted_list))
print ('Image name:Label:Predicted label')
count = 0 
for i in range(len(sample_list)):
    wrong_id = []
    if predicted_list[i] != sample_list[i]:
        count += 1
        wrong_id.append(test_set_name[i])
        wrong_id.append(sample_list[i])
        wrong_id.append(predicted_list[i])
        print (wrong_id)
print ('Total incorrect results:', count, '/', len(sample_list)) 

score = model.evaluate(X_test, Y_test, verbose=0, show_accuracy=True)
# print('Test score:', score[0])
print('Test accuracy:', score[1])
