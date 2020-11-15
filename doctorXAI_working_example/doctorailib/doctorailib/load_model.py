from collections import OrderedDict
import theano
import theano.tensor as T
from .utils_train import numpy_floatX
from theano import config
import pickle
import numpy as np



def init_tparams(params):
    tparams = OrderedDict()
    for key, value in params.items():
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
    #if options['useTime']:
    #    emb = T.concatenate([t.reshape([n_timesteps,n_samples,1]), emb], axis=2) #Adding the time element to the embedding

    inputVector = emb
    for i, hiddenDimSize in enumerate(options['hiddenDimSize']):
        memories = gru_layer(tparams, inputVector, str(i), hiddenDimSize, mask=mask)
        inputVector = memories * 0.5

    def softmaxStep(memory2d):
        return T.nnet.softmax(T.dot(memory2d, tparams['W_output']) + tparams['b_output'])

    results, updates = theano.scan(fn=softmaxStep, sequences=[inputVector], outputs_info=None, name='softmax_layer', n_steps=n_timesteps)
    results = results * mask[:,:,None]

    #duration = 0.0
    #if options['predictTime']:
    #    duration = T.maximum(T.dot(inputVector, tparams['W_time']) + tparams['b_time'], 0)
    #    duration = duration.reshape([n_timesteps,n_samples]) * mask
    #    return x, t, mask, results, duration
    #elif options['useTime']:
    #    return x, t, mask, results
    #else:
        #return x, mask, results
    return x, mask, results


def build_trained_model(tparams, options):
    x = T.tensor3('x', dtype=config.floatX)
    t = T.matrix('t', dtype=config.floatX)
    mask = T.matrix('mask', dtype=config.floatX)

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = T.dot(x, tparams['W_emb'])

    inputVector = emb
    for i, hiddenDimSize in enumerate(options['hiddenDimSize']):
        memories = gru_layer(tparams, inputVector, str(i), hiddenDimSize, mask=mask)
        inputVector = memories * 0.5

    def fwdStep(memory2d):
        return T.dot(memory2d, tparams['W_output']) + tparams['b_output']

    raw_results, _ = theano.scan(fn=fwdStep, sequences=[inputVector], outputs_info=None, name='fwd_layer', n_steps=n_timesteps)

    def softmaxStep(memory2d):
        return T.nnet.softmax(T.dot(memory2d, tparams['W_output']) + tparams['b_output'])

    results, _ = theano.scan(fn=softmaxStep, sequences=[inputVector], outputs_info=None, name='softmax_layer', n_steps=n_timesteps)

    results = results * mask[:, :, None]

    return x, mask, raw_results, results


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

    x = np.zeros((maxlen, n_samples, inputDimSize)).astype(config.floatX)
    t = np.zeros((maxlen, n_samples)).astype(config.floatX)
    mask = np.zeros((maxlen, n_samples)).astype(config.floatX)
    for idx, (seq, time) in enumerate(zip(seqs, times)):
        for xvec, subseq in zip(x[:, idx, :], seq[:-1]):
            xvec[subseq] = 1.
        mask[:lengths[idx], idx] = 1.
        t[:lengths[idx], idx] = time[:-1]

    if options['useLogTime']:
        t = np.log(t + options['logEps'])

    return x, t, mask, lengths

def padtrainedMatrixWithoutTime(seqs, options):
    #this one predicts the last code (code number len(seqs))
    #for test purposes
    #print('predicting the last visit of the patient (not the future one) - for test purposes')
    lengths = np.array([len(seq) for seq in seqs]) - 1
    n_samples = len(seqs)
    maxlen = np.max(lengths)
    inputDimSize = options['inputDimSize']

    x = np.zeros((maxlen, n_samples, inputDimSize)).astype(config.floatX)
    mask = np.zeros((maxlen, n_samples)).astype(config.floatX)
    for idx, seq in enumerate(seqs):
        for xvec, subseq in zip(x[:, idx, :], seq[:-1]):
            xvec[subseq] = 1.
        mask[:lengths[idx], idx] = 1.

    return x, mask, lengths

def padMatrixWithoutTime(seqs, options):
    #this one predicts the last code (code number len(seqs))
    #for test purposes
    #print('predicting the last visit of the patient (not the future one) - for test purposes')
    lengths = np.array([len(seq) for seq in seqs]) - 1
    n_samples = len(seqs)
    maxlen = np.max(lengths)
    inputDimSize = options['inputDimSize']

    x = np.zeros((maxlen, n_samples, inputDimSize)).astype(config.floatX)
    mask = np.zeros((maxlen, n_samples)).astype(config.floatX)
    for idx, seq in enumerate(seqs):
        for xvec, subseq in zip(x[:, idx, :], seq[:-1]):
            xvec[subseq] = 1.
        mask[:lengths[idx], idx] = 1.

    return x, mask, lengths


def newpadMatrixWithoutTime(seqs, options):
    #this one predicts the future code (code number len(seqs)+1)
    #for "real" predicition purposes
    print('predicting the FUTURE visit of the patient, there is no ground truth for this')
    lengths = np.array([len(seq) for seq in seqs])
    n_samples = len(seqs)
    maxlen = np.max(lengths)
    inputDimSize = options['inputDimSize']

    x = np.zeros((maxlen, n_samples, inputDimSize)).astype(config.floatX)
    mask = np.zeros((maxlen, n_samples)).astype(config.floatX)
    for idx, seq in enumerate(seqs):
        for xvec, subseq in zip(x[:, idx, :], seq):
            xvec[subseq] = 1.
        mask[:lengths[idx], idx] = 1.

    return x, mask, lengths