'''
Data set is from followinf paper:
Lev Finkelstein, Evgeniy Gabrilovich, Yossi Matias, Ehud Rivlin, Zach Solan, Gadi Wolfman, and Eytan Ruppin. 2001. Placing search in context: the concept revisited. In Proc. of WWW.
'''

import numpy as np
import scipy
from scipy.stats import *
import sys

embeddingSource = sys.argv[1]
embeddings = ""


def loadData():
	ret = {}
	data = open(embeddingSource,"r").readlines()
	for row in data:
		word = row.strip().split(' ')[0]
		vals = row.strip().split(' ')[1:]
		vals = np.array( [float(val) for val in vals] )
		ret[word] = vals
	return ret


def loadTestData():
	tmp = open("word_sim.tab","r").readlines()
	data = {}
	data['words'] = [ row.strip().split('\t')[0:2] for i,row in enumerate(tmp) if i!=0 ]
	data['sim_scores'] = [ float(row.strip().split('\t')[2]) for i,row in enumerate(tmp) if i!=0  ]
	return data

def getSimilarity(e1, e2):
	# cosine similarity
	return np.sum(e1 * e2)/( np.sqrt(np.sum(e1*e1)) * np.sqrt(np.sum(e2*e2)))

def getSimilarityScoreForWords(w1,w2):
	global embeddings
	#print w1
	#print embeddings
	if (w2 not in embeddings) or (w1 not in embeddings) :
		return -1
	finalVector_w1 = embeddings[w1]
	finalVector_w2 = embeddings[w2]
	return getSimilarity(finalVector_w1, finalVector_w2)

def evaluate():
	global embeddings
	print("--- loading data...")
	embeddings = loadData()
	data = loadTestData()
	print("#words = ", len(data['words']))
	print("#scores = ", len(data['sim_scores']))
	print("--- checking...")
	pred_scores = []
	invalid = 0
	pred_scores = [ [getSimilarityScoreForWords(w1w2[0],w1w2[1]),human_score] for w1w2,human_score in zip(data['words'], data['sim_scores']) ]			
	pred_scores = np.array( [ val for val in pred_scores if val[0]!=-1 ] )
	#print pred_scores
	spearman_rank_coeff,sp_rho = spearmanr( pred_scores[:,0], pred_scores[:,1] )
	print("total, valid,spearman_rank_coeff,sp_rho ", len(data['words']),len(pred_scores), spearman_rank_coeff,sp_rho)
		
evaluate()
