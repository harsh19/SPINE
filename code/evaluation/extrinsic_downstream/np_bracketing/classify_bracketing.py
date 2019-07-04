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
import sys
import pickle

h = .02  # step size in the mesh
embedding_size = None
vectors = None

# python classify.py embedding_src num_classes x_train_pickle y_train_pickle x_val_pickle y_val_pickle x_test_pickle y_test_pickle

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
    #print "embedding_size = ",embedding_size
    return vectors


def getFeats(sentence):
    global vectors
    global embedding_size
    if False:
      ret = np.zeros(embedding_size)
      cnt = 0
      for word in sentence:
          if word in vectors:
              ret+=vectors[word]
              cnt+=1
      if cnt>0:
        ret/=cnt
      return ret
    else:
      ret = []
      cnt = 0
      for word in sentence:
          if word in vectors:
              ret.extend([ v for v in vectors[word]])
              cnt+=1
          else:
              ret.extend( [v for v in np.zeros(embedding_size)] )
      ret = np.array(ret)
      return ret

def getOneHot(vals, max_num):
    ret = np.zeros((vals.shape[0], max_num))
    for i,val in enumerate(vals):
        ret[i][val] = 1
    return ret

def trainAndTest(x_splits, y_splits, clf):
    clf.fit(x_splits[0], y_splits[0])
    train_score = clf.score(x_splits[0], y_splits[0])
    val_score = None
    if len(x_splits[1])>0:
        val_score = clf.score(x_splits[1], y_splits[1])
        #print "Val Score = ", val_score
    score = clf.score(x_splits[2], y_splits[2])
    #print "Test Score = ", score
    return train_score,val_score,score

def main():
    loadVectors()
    num_classes = int(sys.argv[2])
    #print "num_classes = ",num_classes
    classifiers = None
    if num_classes==2:
        classifiers = [
        SVC(kernel="linear", C=0.025),
        SVC(kernel="linear", C=0.1),
        SVC(kernel="linear", C=1.0),
        SVC(gamma=2, C=1),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        RandomForestClassifier(max_depth=5, n_estimators=50, max_features=10),
        MLPClassifier(alpha=1),
        RandomForestClassifier(n_estimators=20, max_features=10)]
    else:
        #TODO
        pass


    all_feats = []
    labels = []
    idx = 3
    while idx<8:
        texts = pickle.load( open(sys.argv[idx],"rb") )
        if len(texts)>0:
            feats = np.array( [getFeats(t) for t in texts] )
            #print "feats : ",feats.shape
            all_feats.append( feats )
            idx+=1
            cur_labels = np.array(pickle.load( open(sys.argv[idx],"rb") ) )
            #cur_labels = getOneHot(cur_labels, max(cur_labels)+1)
            labels.append( cur_labels )
            #print "cur_labels : ",cur_labels.shape
            idx+=1
        else:
            idx+=2
            labels.append([])
            all_feats.append([])
    print("Done loading data")

    best_test = 0.0
    best_clf = None
    best = 0.0
    for clf in classifiers:
        #print "="*33
        #print "clf = ",clf
        score, val_score, test_score = trainAndTest(all_feats, labels, clf)
        best_test = max(best_test, test_score)
        if score>best:
          best = score
          best_clf = clf
    print("best_test for this split= ", best_test)
    #print "best_test = ", best_test
    #print "best= ", best, best_clf

main()
