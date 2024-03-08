
def d():
    return b()
def a():
    import numpy as np
    import ffffffffff
    k = np.ones((2,2), dtype=float)
    b()
    print('a calls b')
    print('a: numpy %f' % k[1,0])
def b():
    c()
    print('b calls c')
def c():
    print('this is c_t1')
    
if __name__ == '__main__':
    a()
    