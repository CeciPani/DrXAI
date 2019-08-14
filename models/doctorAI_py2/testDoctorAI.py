#################################################################
# Code written by Edward Choi (mp2893@gatech.edu)
# For bug report, please contact author using the email address
#################################################################

import sys
import numpy as np
import cPickle as pickle
from collections import OrderedDict
import argparse

import theano
import theano.tensor as T
from theano import config

from Queue import heapq
import operator
import time
import warnings

def recallTop(y_true, y_pred, rank=[10, 20, 30]):
	recall = list()
	for i in range(len(y_pred)):
		thisOne = list()
		codes = y_true[i]
		tops = y_pred[i]
		for rk in rank:
			thisOne.append(len(set(codes).intersection(set(tops[:rk])))*1.0/len(set(codes)))
		recall.append( thisOne )
	return (np.array(recall)).mean(axis=0).tolist()

def calculate_r_squared(trueVec, predVec, options):
	mean_duration = options['mean_duration']
	if options['useLogTime']:
		trueVec = np.log(np.array(trueVec) + options['logEps'])
	else:
		trueVec = np.array(trueVec)
	predVec = np.array(predVec)

	numerator = ((trueVec - predVec) ** 2).sum()
	denominator = ((trueVec - mean_duration) ** 2).sum()

	return 1.0 - (numerator / denominator)

def numpy_floatX(data):
	return np.asarray(data, dtype=config.floatX)

def init_tparams(params):
	tparams = OrderedDict()
	for key, value in params.iteritems():
		tparams[key] = theano.shared(value, name=key)
	return tparams

def gru_layer(tparams, emb, layerIndex, hiddenDimSize, mask=None):
	timesteps = emb.shape[0]
	if emb.ndim == 3: n_samples = emb.shape[1]
	else: n_samples = 1

	W_rx = T.dot(emb, tparams['W_r_'+layerIndex])
	W_zx = T.dot(emb, tparams['W_z_'+layerIndex])
	Wx = T.dot(emb, tparams['W_'+layerIndex])

	def stepFn(stepMask, wrx, wzx, wx, h):
		r = T.nnet.sigmoid(wrx + T.dot(h, tparams['U_r_'+layerIndex]) + tparams['b_r_'+layerIndex])
		z = T.nnet.sigmoid(wzx + T.dot(h, tparams['U_z_'+layerIndex]) + tparams['b_z_'+layerIndex])
		h_tilde = T.tanh(wx + T.dot(r*h, tparams['U_'+layerIndex]) + tparams['b_'+layerIndex])
		h_new = z * h + ((1. - z) * h_tilde)
		h_new = stepMask[:, None] * h_new + (1. - stepMask)[:, None] * h
		return h_new#, output, time

	results, updates = theano.scan(fn=stepFn, sequences=[mask,W_rx,W_zx,Wx], outputs_info=T.alloc(numpy_floatX(0.0), n_samples, hiddenDimSize), name='gru_layer'+layerIndex, n_steps=timesteps)

	return results

def build_model(tparams, options):
	x = T.tensor3('x', dtype=config.floatX)
	t = T.matrix('t', dtype=config.floatX)
	mask = T.matrix('mask', dtype=config.floatX)

	n_timesteps = x.shape[0]
	n_samples = x.shape[1]

	emb = T.dot(x, tparams['W_emb'])
	if options['useTime']:
		emb = T.concatenate([t.reshape([n_timesteps,n_samples,1]), emb], axis=2) #Adding the time element to the embedding

	inputVector = emb
	for i, hiddenDimSize in enumerate(options['hiddenDimSize']):
		memories = gru_layer(tparams, inputVector, str(i), hiddenDimSize, mask=mask)
		inputVector = memories * 0.5

	def softmaxStep(memory2d):
		return T.nnet.softmax(T.dot(memory2d, tparams['W_output']) + tparams['b_output'])

	results, updates = theano.scan(fn=softmaxStep, sequences=[inputVector], outputs_info=None, name='softmax_layer', n_steps=n_timesteps)
	results = results * mask[:,:,None]

	duration = 0.0
	if options['predictTime']:
		duration = T.maximum(T.dot(inputVector, tparams['W_time']) + tparams['b_time'], 0)
		duration = duration.reshape([n_timesteps,n_samples]) * mask
		return x, t, mask, results, duration
	elif options['useTime']:
		return x, t, mask, results
	else:
		return x, mask, results

def load_data(dataFile, labelFile, timeFile):
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

def padMatrixWithTime(seqs, times, options):
	lengths = np.array([len(seq) for seq in seqs]) - 1
	n_samples = len(seqs)
	maxlen = np.max(lengths)
	inputDimSize = options['inputDimSize']
	numClass = options['numClass']

	x = np.zeros((maxlen, n_samples, inputDimSize)).astype(config.floatX)
	t = np.zeros((maxlen, n_samples)).astype(config.floatX)
	mask = np.zeros((maxlen, n_samples)).astype(config.floatX)
	for idx, (seq,time) in enumerate(zip(seqs,times)):
		for xvec, subseq in zip(x[:,idx,:], seq[:-1]):
			xvec[subseq] = 1.
		mask[:lengths[idx], idx] = 1.
		t[:lengths[idx], idx] = time[:-1]
	
	if options['useLogTime']:
		t = np.log(t + options['logEps'])

	return x, t, mask, lengths

def padMatrixWithoutTime(seqs, options):
	lengths = np.array([len(seq) for seq in seqs]) - 1
	n_samples = len(seqs)
	maxlen = np.max(lengths)
	inputDimSize = options['inputDimSize']
	numClass = options['numClass']

	x = np.zeros((maxlen, n_samples, inputDimSize)).astype(config.floatX)
	mask = np.zeros((maxlen, n_samples)).astype(config.floatX)
	for idx, seq in enumerate(seqs):
		for xvec, subseq in zip(x[:,idx,:], seq[:-1]):
			xvec[subseq] = 1.
		mask[:lengths[idx], idx] = 1.
	
	return x, mask, lengths

def test_doctorAI(
	modelFile='model.txt',
	seqFile='seq.txt',
	inputDimSize=20000,
	labelFile='label.txt',
	numClass=500,
	timeFile='',
	predictTime=False,
	useLogTime=True,
	hiddenDimSize=[200,200],
	batchSize=100,
	logEps=1e-8,
	mean_duration=20.0,
	verbose=False
):
	options = locals().copy()

	if len(timeFile) > 0: useTime = True
	else: useTime = False
	options['useTime'] = useTime

	models = np.load(modelFile)
	tparams = init_tparams(models)

	print 'build model ... ',
	if predictTime:
		x, t, mask, codePred, timePred = build_model(tparams, options)
		predict_code = theano.function(inputs=[x,t,mask], outputs=codePred, name='predict_code')
		predict_time = theano.function(inputs=[x,t,mask], outputs=timePred, name='predict_time')
	elif useTime:
		x, t, mask, codePred = build_model(tparams, options)
		predict_code = theano.function(inputs=[x,t,mask], outputs=codePred, name='predict_code')
	else:
		x, mask, codePred = build_model(tparams, options)
		predict_code = theano.function(inputs=[x,mask], outputs=codePred, name='predict_code')

	options['inputDimSize']=models['W_emb'].shape[0]
	options['numClass']=models['b_output'].shape[0]
	print 'load data ... ', 
	testSet = load_data(seqFile, labelFile, timeFile)
	n_batches = int(np.ceil(float(len(testSet[0])) / float(batchSize)))
	print 'done'

	predVec = []
	trueVec = []
	predTimeVec = []
	trueTimeVec = []
	iteration = 0
	for batchIndex in range(n_batches):
		tempX = testSet[0][batchIndex*batchSize: (batchIndex+1)*batchSize]
		tempY = testSet[1][batchIndex*batchSize: (batchIndex+1)*batchSize]
		if predictTime:
			tempT = testSet[2][batchIndex*batchSize: (batchIndex+1)*batchSize]
			x, t, mask, lengths = padMatrixWithTime(tempX, tempT, options)
			codeResults = predict_code(x, t, mask)
			timeResults = predict_time(x, t, mask)
		elif useTime:
			tempT = testSet[2][batchIndex*batchSize: (batchIndex+1)*batchSize]
			x, t, mask, lengths = padMatrixWithTime(tempX, tempT, options)
			codeResults = predict_code(x, t, mask)
		else:
			x, mask, lengths = padMatrixWithoutTime(tempX, options)
			codeResults = predict_code(x, mask)

		for i in range(codeResults.shape[1]):
			tensorMatrix = codeResults[:,i,:]
			thisY = tempY[i][1:]
			for timeIndex in range(lengths[i]):
				if len(thisY[timeIndex]) == 0: continue
				trueVec.append(thisY[timeIndex])
				output = tensorMatrix[timeIndex]
				predVec.append(zip(*heapq.nlargest(30, enumerate(output), key=operator.itemgetter(1)))[0])

		if predictTime:
			for i in range(timeResults.shape[1]):
				timeVec = timeResults[:,i]
				trueTimeVec.extend(tempT[i][1:])
				for timeIndex in range(lengths[i]):
					predTimeVec.append(timeVec[timeIndex])

		if (iteration % 10 == 0) and verbose: print 'iteration:%d/%d' % (iteration, n_batches)
		iteration += 1
		if iteration == 10: break
			
	recall = recallTop(trueVec, predVec)
	print 'recall@10:%f, recall@20:%f, recall@30:%f' % (recall[0], recall[1], recall[2])

	if predictTime: 
		r_squared = calculate_r_squared(trueTimeVec, predTimeVec, options)
		print 'R2:%f' % r_squared

def parse_arguments(parser):
	parser.add_argument('model_file', type=str, metavar='<model_file>', help='The path to the model file saved by Doctor AI')
	parser.add_argument('seq_file', type=str, metavar='<visit_file>', help='The path to the Pickled file containing visit information of patients')
	parser.add_argument('label_file', type=str, metavar='<label_file>', help='The path to the Pickled file containing label information of patients')
	parser.add_argument('hidden_dim_size', type=str, metavar='<hidden_dim_size>', help='The size of the hidden layers of the Doctor AI. This is a string argument. For example, [500,400] means you are using a two-layer GRU where the lower layer uses a 500-dimensional hidden layer, and the upper layer uses a 400-dimensional hidden layer. (default value: [200,200])')
	parser.add_argument('--time_file', type=str, default='', help='The path to the Pickled file containing durations between visits of patients. If you are not using duration information, do not use this option')
	parser.add_argument('--predict_time', type=int, default=0, choices=[0,1], help='Use this option if you want Doctor AI to also predict the time duration until the next visit (0 for false, 1 for true) (default value: 0)')
	parser.add_argument('--use_log_time', type=int, default=1, choices=[0,1], help='Use logarithm of time duration to dampen the impact of the outliers (0 for false, 1 for true) (default value: 1)')
	parser.add_argument('--batch_size', type=int, default=100, help='The size of a single mini-batch (default value: 100)')
	parser.add_argument('--mean_duration', type=float, default=20.0, help='The mean value of the durations between visits of the training data. This will be used to calculate the R^2 error (default value: 20.0)')
	parser.add_argument('--verbose', action='store_true', help='Print output after every 10 mini-batches (default false)')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	args = parse_arguments(parser)
	hiddenDimSize = [int(strDim) for strDim in args.hidden_dim_size[1:-1].split(',')]

	if args.predict_time and args.time_file == '':
		print 'Cannot predict time duration without time file'
		sys.exit()

	test_doctorAI(
		modelFile=args.model_file,
		seqFile=args.seq_file, 
		labelFile=args.label_file, 
		timeFile=args.time_file, 
		predictTime=args.predict_time,
		useLogTime=args.use_log_time,
		hiddenDimSize=hiddenDimSize,
		batchSize=args.batch_size,
		mean_duration=args.mean_duration,
		verbose=args.verbose
	)
