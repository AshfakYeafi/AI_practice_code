import numpy as np
def gradent_dicent(x,y):
    m_init=0
    b_init=0
    n=len(x)
    itration=1000
    lr=0.01
    for i in range(itration):
        y_predict=m_init*x+b_init
        cost=(1/n)*sum([val**2 for val in (y-y_predict)])
        md=-(n/2)*sum(x*(y-y_predict))
        bd=-(n/2)*sum((y-y_predict))

        m_init=m_init-md*lr
        b_init=b_init-bd*lr

        print(f"m : {m_init} and b : {b_init} cost : {cost} for {i} ittration")


x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])
gradent_dicent(x,y)