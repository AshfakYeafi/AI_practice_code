import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
import pickle
path="demo_model_1"

(X_train, y_train) , (X_test, y_test)  =keras.datasets.mnist.load_data()
X_train_flat=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[1])
X_train_flat=X_train_flat/255
# print(y_train[0])
# print(X_train_flat[0])
# plt.matshow(X_train[0])
# plt.show()
# print(X_train_flat.shape)
model=keras.Sequential([
    keras.layers.Dense(400,input_shape=(784,),activation="relu"),
    keras.layers.Dense(200,activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(10,activation="sigmoid")
    ]
)
model.compile(optimizer="Adam",
              loss='sparse_categorical_crossentropy',
              metrics="accuracy")
model.fit(X_train_flat,y_train,epochs=50)

model.save(path)

# new_model=keras.models.load_model(path)
# y_predict=new_model.predict(X_train_flat)
# print(y_predict[100])
# plt.matshow(X_train[100])
# plt.show()
# print(np.argmax(y_predict[100]))