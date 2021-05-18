import numpy as np

def gradientdecent(x,y):
    m_current = b_current = 0
    itretation  = 10000
    n = len(x)
    learning_rate = 0.08
    for i in range(itretation):
        y_prediction = m_current * x + b_current
        cost = (1/n) * sum([val ** 2 for val in  (y - y_prediction)])
        md = -(2/n) * sum(x * (y - y_prediction))
        bd = -(2/n) * sum(y - y_prediction)
        m_current = m_current - learning_rate * md
        b_current = b_current - learning_rate * md 
        print("m {}, b {}, iteration {}, cost {}".format(m_current,b_current,itretation,cost))
 
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradientdecent(x,y)