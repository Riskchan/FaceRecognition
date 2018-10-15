import os
import cv2
import numpy as np
import keras
import matplotlib.pyplot as plt

names = ["takahashi", "tamaki"]

# Labelling training data
X_train = []
Y_train = []
for i in range(len(names)):
    img_file_name_list=os.listdir("./train/"+names[i])
    for j in range(0,len(img_file_name_list)-1):
        print("Loading" + "./train/"+names[i]+"/",img_file_name_list[j])
        n=os.path.join("./train/"+names[i]+"/",img_file_name_list[j])
        img = cv2.imread(n)
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        X_train.append(img)
        Y_train.append(i)

# Labelling test data
X_test = []
Y_test = []
for i in range(len(names)):
    img_file_name_list=os.listdir("./test/"+names[i])
    for j in range(0,len(img_file_name_list)-1):
        n=os.path.join("./test/"+names[i]+"/",img_file_name_list[j])
        img = cv2.imread(n)
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        X_test.append(img)
        Y_test.append(i)

X_train=np.array(X_train)
X_test=np.array(X_test)

from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)

# Model definition
model = Sequential()

model.add(Conv2D(input_shape=(64, 64, 3), filters=32, kernel_size=(3, 3),
                 strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("sigmoid"))
#model.add(Dense(128))
#model.add(Activation('sigmoid'))
model.add(Dense(len(names)))
model.add(Activation('softmax'))

# Compiling
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Learning
history = model.fit(X_train, y_train, batch_size=32,
                    epochs=50, verbose=1, validation_data=(X_test, y_test))

# Plot score
score = model.evaluate(X_test, y_test, batch_size=32, verbose=0)
print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))

# Plotting acc, val_acc
plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
plt.plot(history.history["val_acc"], label="val_acc", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()

# Save model
model.save("face_recognition_cnn.h5")
