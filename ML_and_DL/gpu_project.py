import tensorflow as tf
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()
classes=["airplane", "automobile", "bird","cat","deer","dog","frog","horse","ship","truck"]

x_train_scaled=x_train/255
x_test_scaled=x_test/255
print(x_train_scaled.shape)
y_train_catargorical=keras.utils.to_categorical(
    y_train,num_classes=10
)
print(y_train_catargorical[0:5])

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(32,32,3)),
    keras.layers.Dense(3000,activation="relu"),
    keras.layers.Dense(1000,activation="relu"),
    keras.layers.Dense(10,activation="sigmoid")
])
model.compile(optimizer="SGD",
              loss="categorical_crossentropy",
              metrics=['accuracy'])

model.fit(x_train_scaled,y_train_catargorical,epochs=50)

print(np.argmax(model.predict(x_test_scaled)[0]))
print(y_train[0][0])





















