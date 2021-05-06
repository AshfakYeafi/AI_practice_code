import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
df=pd.read_csv("incur_data.csv")
x_train,x_test,y_train,y_test=train_test_split(df[["age", "affordibility"]],df.bought_insurance,test_size=0.2)
x_train_scale=x_train[["age","affordibility"]].copy()
x_train_scale.age=x_train_scale["age"]/100
x_test_scale=x_test[["age","affordibility"]].copy()
x_test_scale.age=x_test_scale["age"]/100
def sigmoid_numpy(X):
    return 1/1-(np.exp(-X))
def log_loss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_new = [max(i,epsilon) for i in y_predicted]
    y_predicted_new = [min(i,1-epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    # return -np.mean(np.dot(y_true,np.log(y_predicted_new))+np.dot((1-y_true),np.log(1-y_predicted_new)))
    return -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))

def gradent_decent(age,affordibility,y_true,ecops,lr):
    w1=w2=0
    bias=0
    n=len(age)
    loss_coll=[]
    ecop_col=[]
    for i in range(ecops):
        temp_ypredict=w1*age+w2*affordibility+bias
        y_predict=sigmoid_numpy(temp_ypredict)
        loss=log_loss(y_true,y_predict)

        w1d = (1 / n) * np.dot(np.transpose(age), (y_predict - y_true))
        w2d = (1 / n) * np.dot(np.transpose(affordibility), (y_predict - y_true))

        bias_d = np.mean(y_predict - y_true)

        w1=w1-w1d*lr
        w2=w2-w2d*lr
        bias=bias-bias_d*lr
        if i%20==0:
            loss_coll.append(loss)
            ecop_col.append(i)
        print(f"W1={w1} , W2={w2}, bias={bias} ,loss={loss}, ecops={i}")

    plt.xlabel('ecop')
    plt.ylabel('loss')
    plt.plot(ecop_col,loss_coll)
    plt.show()

    return w1,w2,bias

w1,w2,bias=gradent_decent(x_train_scale.age,x_train_scale.affordibility,y_train,50000,0.01)