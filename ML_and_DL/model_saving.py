import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

df=pd.read_csv("model_saving.csv")
model=LinearRegression()
model.fit(df[['year']],df['per capita income (US$)'])
print(model.predict([[2000]]))
with open("model_saving",'wb') as f:
    pickle.dump(model,f)


with open("model_saving",'rb') as f:
    mp=pickle.load(f)
print(mp.predict([[2000]]))