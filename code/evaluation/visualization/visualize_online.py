from multiprocessing import Queue
from queue import PriorityQueue
from numpy import linalg as LA
from scipy import stats
import numpy as np
import argparse
import time
import sys
import os
from tqdm import tqdm

top_k_words = []
zeros = 0.0
threshold = 0.001
h_dim = None
total = None
vectors = {}
num = 5
width = 10


def load_vectors(filename):
    global vectors, dimensions, zeros, h_dim, total, top_k_words
    vectors = {}
    zeros = 0.0
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    dimension = len(lines[0].split()) - 1
    top_k_words = [ [] for i in range(dimension)]
    c = 0
    for line in tqdm(lines):
        start = time.time()
        words = line.strip().split()
        vectors[words[0]] = [abs(float(i)) for i in words[1:]]
        h_dim = len(words[1:])
        c += 1
        vector = vectors[words[0]]
        for i, val in enumerate(vector):
            temp = top_k_words[i]
            if len(temp) < width:
              temp.append((val,words[0]))
            else:
              check = temp[-1]
              if check[0] < val:
                temp[-1] = (val, words[0])
            top_k_words[i] = sorted(temp, reverse=True)
        zeros += sum([1 for i in vectors[words[0]] if i < threshold])
    print ("Sparsity =", 100. * zeros/(len(lines)*dimension))
    total = len(vectors)
    print ('done loading vectors')


def load_top_dimensions(k):
    global top_k_words
    return
    dimensions = len(vectors[vectors.keys()[0]])
    for i in range(dimensions):
        temp = []
        while top_k_words[i].qsize() > 0:
          temp.append(top_k_words.get_nowait()[1])
        top_k_words[i] = temp
    print ('loaded top dimensions')


def find_top_participating_dimensions(word, k):
        if word not in vectors:
            print ('word not found')
            return []
        temp = [(j, i) for i, j in enumerate(vectors[word])]
        answer = []
        print (" -----------------------------------------------------")
        print ("Word of interest = " , word)
        for i, j in sorted(temp, reverse=True)[:k]:
            print ("The contribution of the word '%s' in dimension %d = %f" %(word, j, i))
            print ('Following are the top words in dimension', j, 'along with their contributions')
            print (top_k_words[j])
            answer.append([k[1] for k in top_k_words[j]])
        return


