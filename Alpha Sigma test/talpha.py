import numpy as np
import time as time

def mfunc2(a=100000,b=1,c=0.5,d=0,x=None):
    """
    :param a: scales precision
    :param b: moves function up and down
    :param c: b*c = dist(min,max)
    :param d: x location of spike
    :param x: x
    :return: talpha function
    """
    start_time = time.time()
    input_x = a*x/2-d*a

    func = np.tanh(input_x/2)-1
    ret_func = c*(func)+b
    print(time.time() - start_time)
    return ret_func

print(mfunc2(x=1))