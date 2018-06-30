from nltk import word_tokenize
import math
import numpy as np
import cPickle as pickle
import random

np.random.seed(42)

train_file = "data/train_5500.label"
test_file = "data/TREC_10.label"


classes = {}
class_count = 0

def get_label(label):
  global classes, class_count
  if label not in classes:
    classes[label] = class_count
    class_count += 1
  return classes[label]

def get_Xy(lines):
  X, y = [], []
  for line in lines:
    label = line.split(":")[0]
    sentence = line.split(":")[1]
    words = word_tokenize(sentence)
    y.append(get_label(label))
    X.append(words)
  return X, y

def read_lines(filename):
  return open(filename).readlines()

task = "qa"


train_lines = read_lines(train_file)
test_lines = read_lines(test_file)

test_X, test_y = get_Xy(test_lines)

random.shuffle(train_lines)

X, y = get_Xy(train_lines)

val_number = int(math.floor(0.9 * len(X)))

print val_number

train_X, train_y = X[:val_number], y[:val_number]

val_X, val_y = X[val_number:], y[val_number:]

names = ['train_X.pickle', 'train_y.pickle', 'val_X.pickle', 'val_y.pickle', 'test_X.pickle', 'test_y.pickle']

data_files = [train_X, train_y, val_X, val_y, test_X, test_y]

for data, name in zip(data_files, names):
  pickle.dump(data, open("data/"+task+"_"+name, 'w'))
