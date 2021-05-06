from sklearn.linear_model import LinearRegression
import pandas as pd
import math
from word2number import w2n

df=pd.read_csv("lin_data_1.csv")
df.experience=df.experience.fillna("zero")
df.experience=df.experience.apply(w2n.word_to_num)
mean_test_srore=df['test_score(out of 10)'].median()
df['test_score(out of 10)']=df['test_score(out of 10)'].fillna(math.floor(mean_test_srore))
rec=LinearRegression()
fiture_df=df[["experience",'test_score(out of 10)','interview_score(out of 10)']]
rec.fit(fiture_df,df.salary)
print(rec.coef_)
print(rec.intercept_)
pre=[5,8,10]
ans=rec.predict([pre])
print(ans)