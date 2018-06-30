import numpy as np
import logging
from sklearn.datasets import make_blobs
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)

############################################

class DataHandler:

	def __init__(self):
		pass

	def loadData(self, filename):
		logging.info("Loading data from " + filename )
		#limit = 1000 ## for debnugging. TODO: remove this
		lines = open(filename).readlines()#[:limit] 
		self.data = []
		self.words = []
		for line in lines:
			tokens = line.strip().split()
			self.words.append(tokens[0])
			self.data.append([float(i) for i in tokens[1:]])
		self.data = np.array(self.data)
		logging.info("Loaded data. #shape = " + str(self.data.shape) )
		logging.info(" #words = %d " %(len(self.words)) )
		self.data_size = self.data.shape[0]
		self.inp_dim = self.data.shape[1]
		self.original_data = self.data[:]
		logging.debug("original_data[0][0:5] = " + str(self.original_data[0][0:5]))

	def getWordsList(self):
		return self.words

	def getDataShape(self):
		return self.data.shape

	def resetDataOrder(self):
		self.data = self.original_data[:]
		logging.debug("original_data[0][0:5] = " + str(self.original_data[0][0:5]))

	def getNumberOfBatches(self, batch_size):
		return int(( self.data_size + batch_size - 1 ) / batch_size)

	def getBatch(self, i, batch_size, noise_level, denoising):
		batch_y = self.data[i*batch_size:min((i+1)*batch_size, self.data_size)]
		batch_x = batch_y
		if denoising:
			batch_x = batch_y + get_noise_features(batch_y.shape[0], self.inp_dim, noise_level)
		return batch_x, batch_y

	def shuffleTrain(self):
		indices = np.arange(self.data_size)
		np.random.shuffle(indices)
		self.data = self.data[indices]

############################################

def compute_sparsity(X):
	non_zeros = 1. * np.count_nonzero(X)
	total = X.size
	sparsity = 100. * (1 - (non_zeros)/total)
	return sparsity

def dump_vectors(X, outfile, words):
	print ("shape", X.shape)
	assert len(X) == len(words) #TODO print error statement
	fw = open(outfile, 'w')
	for i in range(len(words)):
		fw.write(words[i] + " ")
		for j in X[i]:
			fw.write(str(j) + " ")
		fw.write("\n")
	fw.close()

def get_noise_features(n_samples, n_features, noise_amount):
	noise_x,  _ =  make_blobs(n_samples=n_samples, n_features=n_features, 
			cluster_std=noise_amount,
			centers=np.array([np.zeros(n_features)]))
	return noise_x
