import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
digits = load_digits()
print(dir(digits))
print(digits.target[0])
plt.gray()
plt.matshow(digits.images[10])

X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.2)
# model =make_pipeline(StandardScaler(),LogisticRegression())
model=LogisticRegression()
model.fit(X_train,y_train)
print(model.predict([digits.data[4]]))
print(model.score(X_test,y_test))