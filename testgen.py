import numpy as np
import random
import math
import sys
from scipy import stats
from skimage import data, io, color, transform, exposure
import matplotlib.pyplot as plt

plt.rc('figure', figsize=(6, 4), dpi=110)
plt.rc('font', size=10)


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour, marker='x')
    plt.show()

def gen_noise(n=20, mean_dist=0, var_dist=1):
    s=np.random.normal(loc=mean_dist, scale=math.sqrt(var_dist), size=n)
    return s

def gen_line_coef(x,y):
    coef=40*(np.random.rand(2)*2-1)
    coef[0]=y-coef[1]*x
    return coef

def gen_pol_coef(x,y,degree=3):
    coef=2*((np.random.rand(degree+1)*2-1)+1)
    return coef

def gen_sine_coef(x,y):
    coef=20*(np.random.rand(2)*2-1)
    coef[0]=y-coef[1]*math.sin(x)
    return coef

def applypolynomial(A, x):
    result=0
    for i in range(len(A)):
        result+=A[i]*x**i
    return result

def gen_x(min,max=10):
    max_x=random.randint(min+2,min+max+1)
    xs= sorted([random.uniform(min,max_x+1) for x in range(19)])
    xs.insert(0,min)
    return xs

def gen_values(types,n=20):
    X=[]
    Y=[]
    for i in range(len(types)):
        if i==0:
            tempX=random.randint(-100,101)
            tempY=random.randint(-100,101)
        else:
            tempX=math.ceil(X[i*20-1])
            tempY=math.ceil(Y[i*20-1])
        if types[i]=='p':
            X=np.concatenate((X,gen_x(tempX,4)))
            A=gen_pol_coef(tempX,tempY)
            r=random.randint(-5,10)
            print(tempY)
            [Y.append(applypolynomial(A,x-tempX)+tempY) for x in X[i*20:i*20+20]]
        if types[i]=='l':
            X=np.concatenate((X,gen_x(tempX,5)))
            A=gen_line_coef(tempX,tempY)
            [Y.append(A[0]+A[1]*x) for x in X[i*20:i*20+20]]
        if types[i]=='s':
            X=np.concatenate((X,gen_x(tempX)))
            A=gen_sine_coef(tempX,tempY)
            [Y.append(A[0]+A[1]*math.sin(x)) for x in X[i*20:i*20+20]]
    return X,Y

types=[random.choice(['l','l','p','p','s']) for i in range(int(sys.argv[1]))]
print(types)
xs,ys=gen_values(types)
ys+=gen_noise(20*len(types))
view_data_segments(xs,ys)
f= open("test.csv","w+")
for i in range(len(xs)):
    #print("%f, %f\n" % (xs[i], ys[i]))
    f.write("%f, %f\n" % (xs[i], ys[i]))
