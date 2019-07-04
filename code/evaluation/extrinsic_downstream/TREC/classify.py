import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



import sys
import pickle

h = .02  # step size in the mesh
embedding_size = None
vectors = None

# python classify.py embedding_src num_classes x_train_pickle y_train_pickle x_val_pickle y_val_pickle x_test_pickle y_test_pickle

best_val = 0.0
best_test = 0.0

def loadVectors():
    global embedding_size
    global vectors
    data = open(sys.argv[1],"r").readlines()
    vectors = {}
    for row in data:
        vals = row.split()
        word = vals[0]
        vals = np.array( [float(val) for val in vals[1:]] )
        vectors[word] = vals
    embedding_size = len(vals)
    print("embedding_size = ",embedding_size)
    return vectors


def getFeats(sentence):
    global vectors
    global embedding_size
    ret = np.zeros(embedding_size)
    cnt = 0
    for word in sentence:
        if word.lower() in vectors:
            ret+=vectors[word.lower()]
            cnt+=1
    if cnt>0:
        ret/=cnt
    return ret

def getOneHot(vals, max_num):
    ret = np.zeros((vals.shape[0], max_num))
    for i,val in enumerate(vals):
        ret[i][val] = 1
    return ret

def trainAndTest(x_splits, y_splits, clf):
    global best_val, best_test
    clf.fit(x_splits[0], y_splits[0])
    flag = False
    if len(x_splits[1])>0:
        score = clf.score(x_splits[1], y_splits[1])
        if score > best_val:
          flag = True
          best_val = score
        print("Val Score = ", score)
    score = clf.score(x_splits[2], y_splits[2])
    if flag:
      best_test = score
    print("Test Score = ", score)

def main():
    loadVectors()
    num_classes = int(sys.argv[2])
    print("num_classes = ",num_classes)
    classifiers = None
    classifiers = [
    SVC(kernel="linear", C=0.025, class_weight='balanced'),
    SVC(kernel="linear", C=0.1, class_weight='balanced'),
    SVC(kernel="linear", C=5, class_weight='balanced'),
    SVC(kernel="linear", C=10, class_weight='balanced'),
    SVC(kernel="linear", C=50, class_weight='balanced'),
    SVC(kernel="linear", C=100, class_weight='balanced'),
    SVC(kernel="linear", C=500, class_weight='balanced'),
    SVC(kernel="linear", C=1000, class_weight='balanced'),
    SVC(kernel="linear", C=0.25, class_weight='balanced'),
    SVC(gamma=2, C=0.1, class_weight='balanced'),
    SVC(gamma=2, C=0.25, class_weight='balanced'),
    SVC(C=0.1, class_weight='balanced'),
    SVC(C=5, class_weight='balanced'),
    SVC(C=10, class_weight='balanced'),
    SVC(C=50, class_weight='balanced'),
    SVC(C=100, class_weight='balanced'),
    SVC(C=500, class_weight='balanced'),
    SVC(C=1000, class_weight='balanced'),
    SVC(class_weight='balanced'),
    MLPClassifier(alpha=1),
    GaussianNB(),
    RandomForestClassifier(),
    LogisticRegression(class_weight='balanced'),
    LogisticRegression(class_weight='balanced', C=.025),
    LogisticRegression(class_weight='balanced', C=0.1),
    LogisticRegression(class_weight='balanced', C=5),
    LogisticRegression(class_weight='balanced', C=10),
    LogisticRegression(class_weight='balanced', C=50),
    LogisticRegression(class_weight='balanced', C=100),
    LogisticRegression(class_weight='balanced', C=500),
    ]


    all_feats = []
    labels = []
    idx = 3
    while idx<8:
        texts = pickle.load( open(sys.argv[idx],"rb") )
        if len(texts)>0:
            feats = np.array( [getFeats(t) for t in texts] )
            print("feats : ",feats.shape)
            all_feats.append( feats )
            idx+=1
            cur_labels = np.array(pickle.load( open(sys.argv[idx],"rb") ) )
            #cur_labels = getOneHot(cur_labels, max(cur_labels)+1)
            labels.append( cur_labels )
            print("cur_labels : ",cur_labels.shape)
            idx+=1
        else:
            idx+=2
            labels.append([])
            all_feats.append([])

    for clf in classifiers:
        #print "="*33
        #print "clf = ",clf
        trainAndTest(all_feats, labels, clf)


main()
print('best val', best_val)
print('best test', best_test)
