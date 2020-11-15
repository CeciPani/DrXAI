import numpy as np
import pickle
from collections import OrderedDict
import os
os.environ["THEANO_FLAGS"] = "optimizer_excluding=scanOp_pushout_output"
import theano
from theano import config

def xrange(x):
    return range(x)

def unzip(zipped):
	new_params = OrderedDict()
	for key, value in zipped.items():
		new_params[key] = value.get_value()
	return new_params

def numpy_floatX(data):
	return np.asarray(data, dtype=config.floatX)

def load_embedding(infile):
	Wemb = np.array(pickle.load(open(infile, 'rb'))).astype(config.floatX)
	return Wemb

def init_params(options):
	params = OrderedDict()
	timeFile = options['timeFile']
	embFile = options['embFile']
	embSize = options['embSize']
	inputDimSize = options['inputDimSize']
	numClass = options['numClass']

	if len(embFile) > 0:
		print('using external code embedding')
		params['W_emb'] = load_embedding(embFile)
		embSize = params['W_emb'].shape[1]
	else:
		print('using randomly initialized code embedding')
		params['W_emb'] = np.random.uniform(-0.01, 0.01, (inputDimSize, embSize)).astype(config.floatX)
	params['b_emb'] = np.zeros(embSize).astype(config.floatX)

	prevDimSize = embSize
	if len(timeFile) > 0: prevDimSize += 1 #We need to consider an extra dimension for the duration information
	for count, hiddenDimSize in enumerate(options['hiddenDimSize']):
		params['W_'+str(count)] = np.random.uniform(-0.01, 0.01, (prevDimSize, hiddenDimSize)).astype(config.floatX)
		params['W_r_'+str(count)] = np.random.uniform(-0.01, 0.01, (prevDimSize, hiddenDimSize)).astype(config.floatX)
		params['W_z_'+str(count)] = np.random.uniform(-0.01, 0.01, (prevDimSize, hiddenDimSize)).astype(config.floatX)
		params['U_'+str(count)] = np.random.uniform(-0.01, 0.01, (hiddenDimSize, hiddenDimSize)).astype(config.floatX)
		params['U_r_'+str(count)] = np.random.uniform(-0.01, 0.01, (hiddenDimSize, hiddenDimSize)).astype(config.floatX)
		params['U_z_'+str(count)] = np.random.uniform(-0.01, 0.01, (hiddenDimSize, hiddenDimSize)).astype(config.floatX)
		params['b_'+str(count)] = np.zeros(hiddenDimSize).astype(config.floatX)
		params['b_r_'+str(count)] = np.zeros(hiddenDimSize).astype(config.floatX)
		params['b_z_'+str(count)] = np.zeros(hiddenDimSize).astype(config.floatX)
		prevDimSize = hiddenDimSize

	params['W_output'] = np.random.uniform(-0.01, 0.01, (prevDimSize, numClass)).astype(config.floatX)
	params['b_output'] = np.zeros(numClass).astype(config.floatX)

	if options['predictTime']:
		params['W_time'] = np.random.uniform(-0.01, 0.01, (prevDimSize, 1)).astype(config.floatX)
		params['b_time'] = np.zeros(1).astype(config.floatX)

	return params


def init_tparams(params, options):
	tparams = OrderedDict()
	for key, value in params.items():
		if not options['embFineTune'] and key == 'W_emb': continue
		tparams[key] = theano.shared(value, name=key)
	return tparams


def padMatrixWithTime(seqs, labels, times, options):
	lengths = np.array([len(seq) for seq in seqs]) - 1
	n_samples = len(seqs)
	maxlen = np.max(lengths)
	inputDimSize = options['inputDimSize']
	numClass = options['numClass']

	x = np.zeros((maxlen, n_samples, inputDimSize)).astype(config.floatX)
	y = np.zeros((maxlen, n_samples, numClass)).astype(config.floatX)
	t = np.zeros((maxlen, n_samples)).astype(config.floatX)
	mask = np.zeros((maxlen, n_samples)).astype(config.floatX)
	for idx, (seq,time,label) in enumerate(zip(seqs,times,labels)):
		for xvec, subseq in zip(x[:,idx,:], seq[:-1]):
			xvec[subseq] = 1.
		for yvec, subseq in zip(y[:,idx,:], label[1:]):
			yvec[subseq] = 1.
		mask[:lengths[idx], idx] = 1.
		t[:lengths[idx], idx] = time[:-1]

	lengths = np.array(lengths, dtype=config.floatX)
	if options['useLogTime']:
		t = np.log(t + options['logEps'])

	return x, y, t, mask, lengths


def padMatrixWithoutTime(seqs, labels, options):
	lengths = np.array([len(seq) for seq in seqs]) - 1
	n_samples = len(seqs)
	maxlen = np.max(lengths)
	inputDimSize = options['inputDimSize']
	numClass = options['numClass']

	x = np.zeros((maxlen, n_samples, inputDimSize)).astype(config.floatX)
	y = np.zeros((maxlen, n_samples, numClass)).astype(config.floatX)
	mask = np.zeros((maxlen, n_samples)).astype(config.floatX)
	for idx, (seq,label) in enumerate(zip(seqs,labels)):
		for xvec, subseq in zip(x[:,idx,:], seq[:-1]):
			xvec[subseq] = 1.
		for yvec, subseq in zip(y[:,idx,:], label[1:]):
			#print subseq
			yvec[subseq] = 1.
		mask[:lengths[idx], idx] = 1.

	lengths = np.array(lengths, dtype=config.floatX)

	return x, y, mask, lengths


def padMatrixWithTimePrediction(seqs, labels, times, options):
	lengths = np.array([len(seq) for seq in seqs]) - 1
	n_samples = len(seqs)
	maxlen = np.max(lengths)
	inputDimSize = options['inputDimSize']
	numClass = options['numClass']

	x = np.zeros((maxlen, n_samples, inputDimSize)).astype(config.floatX)
	y = np.zeros((maxlen, n_samples, numClass)).astype(config.floatX)
	t = np.zeros((maxlen, n_samples)).astype(config.floatX)
	t_label = np.zeros((maxlen, n_samples)).astype(config.floatX)
	mask = np.zeros((maxlen, n_samples)).astype(config.floatX)
	for idx, (seq,time,label) in enumerate(zip(seqs,times,labels)):
		for xvec, subseq in zip(x[:,idx,:], seq[:-1]):
			xvec[subseq] = 1.
		for yvec, subseq in zip(y[:,idx,:], label[1:]):
			yvec[subseq] = 1.
		mask[:lengths[idx], idx] = 1.
		t[:lengths[idx], idx] = time[:-1]
		t_label[:lengths[idx], idx] = time[1:]

	lengths = np.array(lengths, dtype=config.floatX)
	if options['useLogTime']:
		t = np.log(t + options['logEps'])
		t_label = np.log(t_label + options['logEps'])

	return x, y, t, t_label, mask, lengths

def load_data(seqFile, labelFile, timeFile):
	train_set_x = pickle.load(open(seqFile+'.train', 'rb'))
	valid_set_x = pickle.load(open(seqFile+'.valid', 'rb'))
	test_set_x = pickle.load(open(seqFile+'.test', 'rb'))
	train_set_y = pickle.load(open(labelFile+'.train', 'rb'))
	valid_set_y = pickle.load(open(labelFile+'.valid', 'rb'))
	test_set_y = pickle.load(open(labelFile+'.test', 'rb'))
	train_set_t = None
	valid_set_t = None
	test_set_t = None

	if len(timeFile) > 0:
		train_set_t = pickle.load(open(timeFile+'.train', 'rb'))
		valid_set_t = pickle.load(open(timeFile+'.valid', 'rb'))
		test_set_t = pickle.load(open(timeFile+'.test', 'rb'))

	'''For debugging purposes
	sequences = np.array(pickle.load(open(seqFile, 'rb')))
	labels = np.array(pickle.load(open(labelFile, 'rb')))
	if len(timeFile) > 0:
		times = np.array(pickle.load(open(timeFile, 'rb')))

	dataSize = len(labels)
	np.random.seed(0)
	ind = np.random.permutation(dataSize)
	nTest = int(0.15 * dataSize)
	nValid = int(0.10 * dataSize)

	test_indices = ind[:nTest]
	valid_indices = ind[nTest:nTest+nValid]
	train_indices = ind[nTest+nValid:]

	train_set_x = sequences[train_indices]
	train_set_y = labels[train_indices]
	test_set_x = sequences[test_indices]
	test_set_y = labels[test_indices]
	valid_set_x = sequences[valid_indices]
	valid_set_y = labels[valid_indices]
	train_set_t = None
	test_set_t = None
	valid_set_t = None

	if len(timeFile) > 0:
		train_set_t = times[train_indices]
		test_set_t = times[test_indices]
		valid_set_t = times[valid_indices]
	'''

	def len_argsort(seq):
		return sorted(range(len(seq)), key=lambda x: len(seq[x]))

	train_sorted_index = len_argsort(train_set_x)
	train_set_x = [train_set_x[i] for i in train_sorted_index]
	train_set_y = [train_set_y[i] for i in train_sorted_index]

	valid_sorted_index = len_argsort(valid_set_x)
	valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
	valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

	test_sorted_index = len_argsort(test_set_x)
	test_set_x = [test_set_x[i] for i in test_sorted_index]
	test_set_y = [test_set_y[i] for i in test_sorted_index]

	if len(timeFile) > 0:
		train_set_t = [train_set_t[i] for i in train_sorted_index]
		valid_set_t = [valid_set_t[i] for i in valid_sorted_index]
		test_set_t = [test_set_t[i] for i in test_sorted_index]

	train_set = (train_set_x, train_set_y, train_set_t)
	valid_set = (valid_set_x, valid_set_y, valid_set_t)
	test_set = (test_set_x, test_set_y, test_set_t)

	return train_set, valid_set, test_set


def test_load_data(dataFile, labelFile, timeFile):
	test_set_x = np.array(pickle.load(open(dataFile, 'rb')))
	test_set_y = np.array(pickle.load(open(labelFile, 'rb')))
	test_set_t = None
	if len(timeFile) > 0:
		test_set_t = np.array(pickle.load(open(timeFile, 'rb')))

	def len_argsort(seq):
		return sorted(range(len(seq)), key=lambda x: len(seq[x]))

	sorted_index = len_argsort(test_set_x)
	test_set_x = [test_set_x[i] for i in sorted_index]
	test_set_y = [test_set_y[i] for i in sorted_index]
	if len(timeFile) > 0:
		test_set_t = [test_set_t[i] for i in sorted_index]

	test_set = (test_set_x, test_set_y, test_set_t)

	return test_set


def new_test_load_data(dataFile, labelFile, pidsFile, files=False):
	"""This function sorts sequences in the test set according to their lenghts"""

	if files:
		test_set_x = np.array(pickle.load(open(dataFile, 'rb')))
		test_set_y = np.array(pickle.load(open(labelFile, 'rb')))
		pids = np.array(pickle.load(open(pidsFile, 'rb')))
	else:
		test_set_x = dataFile
		test_set_y = labelFile
		pids = pidsFile

	def len_argsort(seq):
		return sorted(range(len(seq)), key=lambda x: len(seq[x]))

	sorted_index = len_argsort(test_set_x)
	test_set_x = [test_set_x[i] for i in sorted_index]
	test_set_y = [test_set_y[i] for i in sorted_index]
	pids = [pids[i] for i in sorted_index]

	test_set = (test_set_x, test_set_y, pids)

	return test_set