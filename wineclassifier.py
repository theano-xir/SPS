#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton code for CW2 submission.
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""

from __future__ import print_function
import math
import argparse
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from sklearn import preprocessing
from utilities import load_data, print_features, print_predictions
import collections
import scipy
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.tree import export_graphviz
from sklearn import tree
import graphviz
# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'

MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']

plt.rc('figure', figsize=(50, 50), dpi=100)
plt.rc('font', size=12)

class Node:
    left = None
    right = None
    feature = None
    value = -1
    groups = None

    def __init__(self, feature, value, groups):
        self.feature = feature
        self.value = value
        self.groups = groups

    def predict(self, point):
        if point[self.feature] <= self.value:
            if isinstance(self.left, Node):
                return self.left.predict(point)
            else: #it's a Leaf
                return self.left.value
        else:
            if isinstance(self.right, Node):
                return self.right.predict(point)
            else:
                return self.right.value

class Leaf:
    value = -1
    def __init__(self, group):
        self.value = make_node_term(group)

def reduce_data(train_set, test_set, selected_features):
    train_set_red = train_set[:, selected_features]
    test_set_red = test_set[:, selected_features]
    return train_set_red, test_set_red

def calculate_centroids(train_set, train_labels):
    classes = np.unique(train_labels)
    centroids = np.array([np.mean(train_set[train_labels == c, :], axis=0) for c in classes])
    return centroids, classes

def nearest_centroid(centroids, test_set):
    dist = lambda x, y: np.sqrt(np.sum((x-y)**2))
    centroid_dist = lambda x : [dist(x, centroid) for centroid in centroids]
    predicted = np.argmin([centroid_dist(p) for p in test_set], axis=1).astype(np.int) + 1
    return predicted

def calculate_accuracy(gt_labels, pred_labels):
    correct=0
    for i in range(len(pred_labels)):
        if pred_labels[i]==gt_labels[i]:
            correct+=1
    return correct/len(gt_labels)

def calculate_confusion_matrix(gt_labels, pred_labels):
    nlabels=len(np.unique(train_labels))
    m = np.zeros([nlabels,nlabels])
    for i in range(len(gt_labels)):
        m[(gt_labels[i]-1),(pred_labels[i]-1).astype(np.int)]+=1
    confm = np.zeros([nlabels,nlabels])
    for i in range(nlabels):
        for j in range(nlabels):
            confm[i,j]=round(m[i,j]/(sum(m[i,:])),3)
    return confm

def plot_matrix(matrix, ax=None):
    """
    Displays a given matrix as an image.

    Args:
        - matrix: the matrix to be displayed
        - ax: the matplotlib axis where to overlay the plot.
          If you create the figure with `fig, fig_ax = plt.subplots()` simply pass `ax=fig_ax`.
          If you do not explicitily create a figure, then pass no extra argument.
          In this case the  current axis (i.e. `plt.gca())` will be used
    """
    if ax is None:
        ax = plt.gca()
    plt.imshow(matrix, cmap=plt.get_cmap('PuBu'))
    plt.clim(0,1)
    plt.yticks([])
    plt.xticks([])
    divider = make_axes_locatable(ax)
    plt.colorbar(orientation='horizontal')
    for i in range(1):
        for j in range(8):
            c = matrix[i,j]
            ax.text(j, i, '{:3.2f}'.format(c), va='center', ha='center')
    plt.show()

def plot_all_combos(train_set, train_labels, **kwargs):
    totalclasses=13
    m = np.zeros([totalclasses,totalclasses])
    for i in range(totalclasses):
        for j in range(totalclasses):
            train_set_red, test_set_red = reduce_data(train_set, test_set, [i,j])
            centroids, classes = calculate_centroids(train_set_red, train_labels)
            predicted = nearest_centroid(centroids, test_set_red)
            m[i,j]=round(calculate_accuracy(test_labels,predicted),2)
    plt.title("Accuracy with different features combinations")
    plt.xticks(np.arange(13))
    plt.yticks(np.arange(13))
    plot_matrix(m)

def plot_all_scatter(train_set, train_labels, **kwargs):
    n_features = train_set.shape[1]
    fig, ax = plt.subplots(n_features, n_features)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)

    class_1_colour = r'#3366ff'
    class_2_colour = r'#cc3300'
    class_3_colour = r'#ffc34d'

    colours = np.zeros_like(train_labels, dtype=np.object)
    colours[train_labels == 1] = class_1_colour
    colours[train_labels == 2] = class_2_colour
    colours[train_labels == 3] = class_3_colour

    for row in range(n_features):
        for col in range(n_features):
            ax[row][col].scatter(train_set[:, row], train_set[:, col], c=colours)
            ax[row][col].set_title('Features {} vs {}'.format(row+1, col+1))
    fig.savefig('plot.png')

def calc_distance(p1,p2):
    return math.sqrt(np.array([(p1[i]-p2[i])**2 for i in range(0,len(p1))]).sum())

def feature_selection(train_set, train_labels, **kwargs):
    return [0,6]

def knn(train_set, train_labels, test_set, k, **kwargs):
    features=feature_selection(train_set,train_labels)
    train_set_red, test_set_red=reduce_data(train_set, test_set, features)
    pred=np.zeros(len(test_set))
    dist=np.zeros((len(test_set), len(train_set)), dtype=object)
    topk=np.zeros(len(test_set),dtype=object)
    pred=np.zeros(len(test_set))
    for i in range(len(test_set)):
        tempk=k
        for j in range(len(train_set)):
            t=calc_distance([test_set_red[i,0],test_set_red[i,1]],[train_set_red[j,0],train_set_red[j,1]]),j
            dist[i,j]=t
        dist[i]=np.sort(dist[i])
        topk[i]=dist[i,:k]
        topk[i]=[train_labels[item[1]] for item in topk[i]]
        freq=scipy.stats.mode(topk[i])
        pred[i]=freq[0][0]
    return pred

# decision tree stuff based on https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
def get_gini(groups,classes):
    no_points=sum([len(g) for g in groups])
    gini=0
    for g in groups:
        size=len(g)
        if (size!=0):
            score=0
            for c in classes:
                p=[row[-1] for row in g].count(c)/size
                score+=(p**2)
            gini+=(1.0-score)*(size/no_points)
    return gini

def get_split(feature,value,data):
    small=[]
    big=[]
    for point in data:
        if point[feature] <= value:
            small.append(point.tolist())
        else:
            big.append(point.tolist())
    return (np.asarray(small),np.asarray(big))

def random_points(op, data):
    rp=np.zeros((op,len(data[0])))
    for i in range(op):
        rp[i]=data[np.random.randint(0,op)]
    return rp

def random_features(of):
    # n=round(math.sqrt(of))
    # return np.random.randint(0, of, n)
    return [0,1]

def get_best_split(data):
    classes=list(set(row[-1] for row in data))
    best_score=2/3
    best_feature=0
    best_value=0
    best_groups=None
    features=random_features(len(data[0])-1)
    features=np.append(features,len(data[0])-1)
    features=features.tolist()
    data_red=data[:,features]
    for feature in range(len(data_red[0])-1):
        for point in data_red:
            groups=get_split(feature,point[feature],data)
            gini = get_gini(groups, classes)
            if gini<best_score:
                best_feature=feature
                best_value=point[feature]
                best_score=gini
                best_groups=groups
    return Node(best_feature, best_value, best_groups)

def make_node_term(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split(node,max_depth,min_size,depth):
    left,right=node.groups
    del(node.groups)
    if len(left)==0:
        node.left=node.right = Leaf(right)
        return
    if len(right)==0:
        node.left=node.right = Leaf(left)
        return
    if depth >= max_depth:
        node.left=Leaf(left)
        node.right = Leaf(right)
        return
    if len(left)<=min_size:
        node.left = Leaf(left)
    else:
        node.left=get_best_split(left)
        split(node.left, max_depth, min_size, depth+1)
    if len(right)<=min_size:
        node.right=Leaf(right)
    else:
        node.right=get_best_split(right)
        split(node.right, max_depth, min_size, depth+1)

def build_tree(data,max_depth,min_size):
    root=get_best_split(random_points(len(data),data))
    split(root, max_depth, min_size, 1)
    return root

def print_tree(node, depth=0):
    if isinstance(node, Node):
        print('%s[X%d < %.3f]' % ((depth*' ', (node.feature+1), node.value)))
        print_tree(node.left, depth+1)
        print_tree(node.right, depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node.value)))

def alternative_classifier(train_set, train_labels, test_set, **kwargs):
    train_set,test_set=reduce_data(train_set, test_set, feature_selection(train_set,train_labels))
    train_labels=np.array(train_labels).reshape((-1,1))
    data=np.concatenate((train_set,train_labels),axis=1)
    predictions=np.zeros((100,len(test_set)))
    for i in range(100):
        root=build_tree(data,5,1)
        for j in range(len(test_set)):
            prediction=root.predict(test_set[j])
            predictions[i,j]=prediction
    f_pred=np.zeros(len(predictions[0]),dtype=object)
    for i in range(len(predictions[0])):
        f_pred[i]=scipy.stats.mode(predictions[:,i])[0][0]
    return f_pred

def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    features=feature_selection(train_set,train_labels)
    features.append(9)
    train_set_red, test_set_red=reduce_data(train_set, test_set, features)
    standard_scale=preprocessing.StandardScaler()
    standard_scale=standard_scale.fit(train_set)
    train_set=standard_scale.transform(train_set)
    test_set=standard_scale.transform(test_set)
    dist=np.zeros((len(test_set), len(train_set)), dtype=object)
    topk=np.zeros(len(test_set),dtype=object)
    pred=np.zeros(len(test_set))
    for i in range(len(test_set)):
        tempk=k
        for j in range(len(train_set)):
            t=calc_distance([test_set_red[i,0],test_set_red[i,1],[test_set_red[i,2] ]],[train_set_red[j,0],train_set_red[j,1],train_set_red[j,2]]),j
            dist[i,j]=t
        dist[i]=np.sort(dist[i])
        topk[i]=dist[i,:k]
        topk[i]=[train_labels[item[1]] for item in topk[i]]
        freq=scipy.stats.mode(topk[i])
        pred[i]=freq[0][0]
    return pred

def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    standard_scale=preprocessing.StandardScaler()
    standard_scale=standard_scale.fit(train_set)
    train_set=standard_scale.transform(train_set)
    test_set=standard_scale.transform(test_set)
    pca = PCA(n_components = n_components)
    pca = pca.fit(train_set)
    train_set_red = pca.transform(train_set)
    test_set_red = pca.transform(test_set)
    dist=np.zeros((len(test_set), len(train_set)), dtype=object)
    topk=np.zeros(len(test_set),dtype=object)
    pred=np.zeros(len(test_set))
    for i in range(len(test_set)):
        tempk=k
        for j in range(len(train_set)):
            t=calc_distance([test_set_red[i,0],test_set_red[i,1]],[train_set_red[j,0],train_set_red[j,1]]),j
            dist[i,j]=t
        dist[i]=np.sort(dist[i])
        topk[i]=dist[i,:k]
        topk[i]=[train_labels[item[1]] for item in topk[i]]
        freq=scipy.stats.mode(topk[i])
        pred[i]=freq[0][0]
    return pred

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
    parser.add_argument('--train_set_path', nargs='?', type=str, default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str, default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str, default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str, default='data/wine_test_labels.csv', help='Path to the test labels csv')

    args = parser.parse_args()
    mode = args.mode[0]

    return args, mode


if __name__ == '__main__':
    args, mode = parse_args() # get argument from the command line

    # load the data
    train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path,
                                                                       train_labels_path=args.train_labels_path,
                                                                       test_set_path=args.test_set_path,
                                                                       test_labels_path=args.test_labels_path)
    if mode == 'feature_sel':
        selected_features = feature_selection(train_set, train_labels)
        print_features(selected_features)
    elif mode == 'knn':
        predictions = knn(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set)
        print_predictions(predictions)
    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'knn_pca':
        predictions = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))