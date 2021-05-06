import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split

df=pd.read_csv('sonar.all-data',header=None)
x=df.drop(60, axis=1)
y=df[60]

y=pd.get_dummies(y,drop_first=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=1)

model=keras.Sequential([
    keras.layers.Dense(60,input_dim=60,activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100,batch_size=8)
print(model.evaluate(x_test, y_test))

y_pred = model.predict(x_test).reshape(-1)
print(y_pred[:10])

# round the values to nearest integer ie 0 or 1
y_pred = np.round(y_pred)
print(y_pred[:10])

print(y_test[:10])


from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test, y_pred))

modeld = keras.Sequential([
    keras.layers.Dense(60, input_dim=60, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

modeld.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

modeld.fit(x_train, y_train, epochs=100, batch_size=8)

y_pred = modeld.predict(x_test).reshape(-1)
print(y_pred[:10])

# round the values to nearest integer ie 0 or 1
y_pred = np.round(y_pred)
print(y_pred[:10])


from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test, y_pred))