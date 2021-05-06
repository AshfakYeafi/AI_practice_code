import numpy as np
y=np.array([1,0,1,1,1,0,0,1,0,1])
y_predict=np.array([1,1,1,0,1,0,0,1,0,0])
# −(ylog(p)+(1−y)log(1−p))
ellipsis = 1e-15
def log_loss(y,y_predict):
    ellipsis=1e-15
    y_predict = np.array([max(i, ellipsis) for i in y_predict])
    y_predict = np.array([min(i, 1 - ellipsis) for i in y_predict])
    return sum(-(y*np.log(y_predict)+(1-y)*np.log(1-y_predict)))/len(y)
print(log_loss(y,y_predict))


