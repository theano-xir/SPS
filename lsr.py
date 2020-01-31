import os
import sys
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


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

def calcerror(Y,ys):
    error=0
    for i in range(len(ys)):
        error+=(Y[i]-ys[i])**2
    return error

def getVandermondeMatrix(xs,p):
    X=np.zeros([len(xs),p])
    X[:,0]=1
    for i in range(p):
        X[:,i]=list(map(lambda x : x**i, xs))
    return X

def getSineMatrix(xs):
    X=np.zeros([len(xs),2])
    X[:,0]=1
    for i in range(len(xs)):
        X[i,1]=math.sin(xs[i])
    return X

def applypolynomial(A, x):
    result=0
    for i in range(len(A)):
        result+=A[i]*x**i
    return result

def findbestfit(xs,ys,startpoint, p=4):
    X=getVandermondeMatrix(xs,p)
    A=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(ys)
    #Tried linalg.inv but hit numerical instability so let's Gauss
    #A=np.linalg.solve(X.T.dot(X), X.T.dot(ys))
    Space=np.linspace(startpoint, xs[19], 100)
    Y=[applypolynomial(A,x) for x in Space]
    Xs=getSineMatrix(xs)
    As=np.linalg.solve(Xs.T.dot(Xs), Xs.T.dot(ys))
    Ys=As[0]+As[1]*np.sin(Space)
    sinerror=calcerror(As[0]+As[1]*np.sin(xs),ys)
    Xl=getVandermondeMatrix(xs,2)
    Al=np.linalg.inv(Xl.T.dot(Xl)).dot(Xl.T).dot(ys)
    Yl=Al[0]+Al[1]*Space
    linerror=calcerror(Al[0]+Al[1]*xs,ys)
    polerror=calcerror([applypolynomial(A,x) for x in xs],ys)
    if linerror<1.2*sinerror and (linerror<1.2*polerror or abs(A[3])<0.01):
        plt.plot(Space,Yl, c="r")
        #print('line')
        return linerror
    if sinerror<1.1*polerror:
        plt.plot(Space,Ys, c="r")
        #print('sine')
        return sinerror
    plt.plot(Space,Y, c="r")
    #print('pol')
    return polerror
        
xs, ys = load_points_from_file(sys.argv[1])
totalerror=0
for i in range(int(len(xs)/20)):
    temp=0
    if i!=0:
        temp=20*i-1
    totalerror+=findbestfit(xs[20*i:20*i+20],ys[20*i:20*i+20],xs[temp],4)
print(totalerror)
if len(sys.argv)==3 and sys.argv[2]=='--plot':
    view_data_segments(xs,ys)