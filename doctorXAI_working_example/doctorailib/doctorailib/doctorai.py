from .utils_train import *
from .metrics import calculate_auc, recallTop
from .load_model import init_tparams as init_trained_theano_params
from .load_model import build_model as old_build_trained_model
from .load_model import build_trained_model
#from .load_model import padMatrixWithoutTime as padtrainedMatrixWithoutTime
from .load_model import padtrainedMatrixWithoutTime
from .load_model import newpadMatrixWithoutTime
import random
import theano
import theano.tensor as T
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import heapq
import operator


class DoctorAI(object):

    def __init__(self,
                 ICD9_to_int_dict,
                 modelFile=None,
                 CCS_to_int_dict=None,
                 embSize=200,
                 hiddenDimSize=[200, 200],
                 verbose=False
                 ):

        """
        :param ICD9_to_int_dict: str, path to the pickle file containing the python dictionary that maps ICD9 code into the internal representation of doctorAI
        :param modelFile: str, path to the .npz file with the pre-trained doctorAI model (if already trained)
        :param CCS_to_int_dict: str, path to the pickle file containing the python dictionary that maps CCS code into the internal representation of doctorAI
        :param embSize: int, The size of the visit embedding before passing it to the GRU layers. If you are not providing your own medical code vectors, you must specify this value (default value: 200)
        :param hiddenDimSize: list of int, The size of the hidden layers of the GRU. This is a string argument. For example, [500,400] means you are using a two-layer GRU where the lower layer uses a 500-dimensional hidden layer, and the upper layer uses a 400-dimensional hidden layer. (default value: [200,200])
        :param verbose: bool, print output after every 10 mini-batches (default False)
        """
        self.ICD9_to_int_dict = ICD9_to_int_dict
        self.CCS_to_int_dict = CCS_to_int_dict
        self.hiddenDimSize = hiddenDimSize
        self.embSize = embSize
        self.verbose = verbose
        self.modelFile = modelFile

    def _dropout_layer(self, state_before, use_noise, trng, dropout_rate):
        proj = T.switch(use_noise, (state_before * trng.binomial(state_before.shape, p=dropout_rate, n=1, dtype=state_before.dtype)),state_before * 0.5)
        return proj

    def _gru_layer(self, tparams, emb, layerIndex, hiddenDimSize, mask=None):
        timesteps = emb.shape[0]
        if emb.ndim == 3:
            n_samples = emb.shape[1]
        else:
            n_samples = 1

        W_rx = T.dot(emb, tparams['W_r_' + layerIndex])
        W_zx = T.dot(emb, tparams['W_z_' + layerIndex])
        Wx = T.dot(emb, tparams['W_' + layerIndex])

        def stepFn(stepMask, wrx, wzx, wx, h):
            r = T.nnet.sigmoid(wrx + T.dot(h, tparams['U_r_' + layerIndex]) + tparams['b_r_' + layerIndex])
            z = T.nnet.sigmoid(wzx + T.dot(h, tparams['U_z_' + layerIndex]) + tparams['b_z_' + layerIndex])
            h_tilde = T.tanh(wx + T.dot(r * h, tparams['U_' + layerIndex]) + tparams['b_' + layerIndex])
            h_new = z * h + ((1. - z) * h_tilde)
            h_new = stepMask[:, None] * h_new + (1. - stepMask)[:, None] * h
            return h_new

        results, updates = theano.scan(fn=stepFn, sequences=[mask, W_rx, W_zx, Wx],
                                       outputs_info=T.alloc(numpy_floatX(0.0), n_samples, hiddenDimSize),
                                       name='gru_layer' + layerIndex, n_steps=timesteps)

        return results

    def _build_model(self, tparams, options, W_emb=None):
        trng = RandomStreams(123)
        use_noise = theano.shared(numpy_floatX(0.))

        if len(options['timeFile']) > 0:
            useTime = True
        else:
            useTime = False

        x = T.tensor3('x', dtype=config.floatX)
        t = T.matrix('t', dtype=config.floatX)
        y = T.tensor3('y', dtype=config.floatX)
        t_label = T.matrix('t_label', dtype=config.floatX)
        mask = T.matrix('mask', dtype=config.floatX)
        lengths = T.vector('lengths', dtype=config.floatX)

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        if options['embFineTune']:
            emb = T.tanh(T.dot(x, tparams['W_emb']) + tparams['b_emb'])
        else:
            emb = T.tanh(T.dot(x, W_emb) + tparams['b_emb'])
        if useTime:
            emb = T.concatenate([t.reshape([n_timesteps, n_samples, 1]), emb],axis=2)  # Adding the time element to the embedding

        inputVector = emb
        for i, hiddenDimSize in enumerate(options['hiddenDimSize']):
            memories = self._gru_layer(tparams, inputVector, str(i), hiddenDimSize, mask=mask)
            memories = self._dropout_layer(memories, use_noise, trng, options['dropout_rate'])
            inputVector = memories

        def softmaxStep(memory2d):
            return T.nnet.softmax(T.dot(memory2d, tparams['W_output']) + tparams['b_output'])

        logEps = options['logEps']
        results, updates = theano.scan(fn=softmaxStep, sequences=[inputVector], outputs_info=None, name='softmax_layer',
                                       n_steps=n_timesteps)
        results = results * mask[:, :, None]
        cross_entropy = -(y * T.log(results + logEps) + (1. - y) * T.log(1. - results + logEps))
        prediction_loss = cross_entropy.sum(axis=2).sum(axis=0) / lengths

        if options['predictTime']:
            duration = T.maximum(T.dot(inputVector, tparams['W_time']) + tparams['b_time'], 0)  # ReLU
            duration = duration.reshape([n_timesteps, n_samples]) * mask
            duration_loss = 0.5 * ((duration - t_label) ** 2).sum(axis=0) / lengths
            cost = T.mean(prediction_loss) + options['tradeoff'] * T.mean(duration_loss) + options['L2_output'] * (
                        tparams['W_output'] ** 2).sum() + options['L2_time'] * (tparams['W_time'] ** 2).sum()
        else:
            cost = T.mean(prediction_loss) + options['L2_output'] * (tparams['W_output'] ** 2).sum()

        if options['predictTime']:
            return use_noise, x, y, t, t_label, mask, lengths, cost
        elif useTime:
            return use_noise, x, y, t, mask, lengths, cost
        else:
            return use_noise, x, y, mask, lengths, cost

    def _adadelta(self, tparams, grads, x, y, mask, lengths, cost, options, t=None, t_label=None):

        zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k) for k, p in tparams.items()]
        running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rup2' % k) for k, p in tparams.items()]
        running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad2' % k) for k, p in tparams.items()]
        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

        if options['predictTime']:
            f_grad_shared = theano.function([x, y, t, t_label, mask, lengths], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')
        elif len(options['timeFile']) > 0:
            f_grad_shared = theano.function([x, y, t, mask, lengths], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')
        else:
            f_grad_shared = theano.function([x, y, mask, lengths], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')

        updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in
                 zip(zipped_grads, running_up2, running_grads2)]
        ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

        f_update = theano.function([], [], updates=ru2up + param_up, on_unused_input='ignore', name='adadelta_f_update')
        return f_grad_shared, f_update


    def train(self,
              seqFile,
              inputDimSize,
              labelFile,
              numClass,
              outFile,
              timeFile='',
              predictTime=False,
              tradeoff=1.0,
              useLogTime=True,
              embFile='',
              embFineTune=True,
              batchSize=100,
              max_epochs=10,
              L2_output=0.001,
              L2_time=0.001,
              dropout_rate=0.5,
              logEps=1e-8
              ):

        options = locals().copy()
        options['embSize'] = self.embSize
        options['hiddenDimSize'] = self.hiddenDimSize
        options['verbose'] = self.verbose

        self.numClass_ = numClass
        self.inputDimSize_ = inputDimSize

        if len(timeFile) > 0:
            useTime = True
        else:
            useTime = False
        options['useTime'] = useTime

        print('Initializing the parameters ... ', )
        params = init_params(options)
        tparams = init_tparams(params, options)

        print('Building the model ... ', )
        f_grad_shared = None
        f_update = None
        if predictTime and embFineTune:
            print('predicting duration, fine-tuning code representations')
            use_noise, x, y, t, t_label, mask, lengths, cost = self._build_model(tparams, options)
            grads = T.grad(cost, wrt=list(tparams.values()))
            f_grad_shared, f_update = self._adadelta(tparams, grads, x, y, mask, lengths, cost, options, t, t_label)
        elif predictTime and not embFineTune:
            print('predicting duration, not fine-tuning code representations')
            W_emb = theano.shared(params['W_emb'], name='W_emb')
            use_noise, x, y, t, t_label, mask, lengths, cost = self._build_model(tparams, options, W_emb)
            grads = T.grad(cost, wrt=list(tparams.values()))
            f_grad_shared, f_update = self._adadelta(tparams, grads, x, y, mask, lengths, cost, options, t, t_label)
        elif useTime and embFineTune:
            print('using duration information, fine-tuning code representations')
            use_noise, x, y, t, mask, lengths, cost = self._build_model(tparams, options)
            grads = T.grad(cost, wrt=list(tparams.values()))
            f_grad_shared, f_update = self._adadelta(tparams, grads, x, y, mask, lengths, cost, options, t)
        elif useTime and not embFineTune:
            print('using duration information, not fine-tuning code representations')
            W_emb = theano.shared(params['W_emb'], name='W_emb')
            use_noise, x, y, t, mask, lengths, cost = self._build_model(tparams, options, W_emb)
            grads = T.grad(cost, wrt=list(tparams.values()))
            f_grad_shared, f_update = self._adadelta(tparams, grads, x, y, mask, lengths, cost, options, t)
        elif not useTime and embFineTune:
            print('not using duration information, fine-tuning code representations')
            use_noise, x, y, mask, lengths, cost = self._build_model(tparams, options)
            grads = T.grad(cost, wrt=list(tparams.values()))
            f_grad_shared, f_update = self._adadelta(tparams, grads, x, y, mask, lengths, cost, options)
        elif useTime and not embFineTune:
            print('not using duration information, not fine-tuning code representations')
            W_emb = theano.shared(params['W_emb'], name='W_emb')
            use_noise, x, y, mask, lengths, cost = self._build_model(tparams, options, W_emb)
            grads = T.grad(cost, wrt=list(tparams.values()))
            f_grad_shared, f_update = self._adadelta(tparams, grads, x, y, mask, lengths, cost, options)

        print('Loading data ... ', )
        trainSet, validSet, testSet = load_data(seqFile, labelFile, timeFile)
        n_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))
        print('done')

        if predictTime:
            test_model = theano.function(inputs=[x, y, t, t_label, mask, lengths], outputs=cost, name='test_model')
        elif useTime:
            test_model = theano.function(inputs=[x, y, t, mask, lengths], outputs=cost, name='test_model')
        else:
            test_model = theano.function(inputs=[x, y, mask, lengths], outputs=cost, name='test_model')

        bestValidCrossEntropy = 1e20
        bestValidEpoch = 0
        testCrossEntropy = 0.0
        print('Optimization start !!')
        for epoch in xrange(max_epochs):
            iteration = 0
            costVector = []
            for index in random.sample(range(n_batches), n_batches):
                use_noise.set_value(1.)
                batchX = trainSet[0][index * batchSize:(index + 1) * batchSize]
                batchY = trainSet[1][index * batchSize:(index + 1) * batchSize]
                if predictTime:
                    batchT = trainSet[2][index * batchSize:(index + 1) * batchSize]
                    x, y, t, t_label, mask, lengths = padMatrixWithTimePrediction(batchX, batchY, batchT, options)
                    cost = f_grad_shared(x, y, t, t_label, mask, lengths)
                elif useTime:
                    batchT = trainSet[2][index * batchSize:(index + 1) * batchSize]
                    x, y, t, mask, lengths = padMatrixWithTime(batchX, batchY, batchT, options)
                    cost = f_grad_shared(x, y, t, mask, lengths)
                else:
                    x, y, mask, lengths = padMatrixWithoutTime(batchX, batchY, options)
                    cost = f_grad_shared(x, y, mask, lengths)
                costVector.append(cost)
                f_update()
                if (iteration % 10 == 0) and self.verbose: print(
                    'epoch:%d, iteration:%d/%d, cost:%f' % (epoch, iteration, n_batches, cost))
                iteration += 1

            print('epoch:%d, mean_cost:%f' % (epoch, np.mean(costVector)))
            use_noise.set_value(0.)
            validAuc = calculate_auc(test_model, validSet, options)
            print('Validation cross entropy:%f at epoch:%d' % (validAuc, epoch))
            if validAuc < bestValidCrossEntropy:
                bestValidCrossEntropy = validAuc
                bestValidEpoch = epoch
                bestParams = unzip(tparams)
                testCrossEntropy = calculate_auc(test_model, testSet, options)
                print('Test cross entropy:%f at epoch:%d' % (testCrossEntropy, epoch))
                tempParams = unzip(tparams)
                np.savez_compressed(outFile + '.' + str(epoch), **tempParams)
        print('The best valid cross entropy:%f at epoch:%d' % (bestValidCrossEntropy, bestValidEpoch))
        print('The test cross entropy: %f' % testCrossEntropy)
        self.modelFile = f"{outFile}.{bestValidEpoch}.npz"


    def predict_future(self,
                patient_seq,
                future=True,
                batchSize=100
                ):

        """
        This function returns the prediction of doctorAI for the future unobserved visit.
        If a sequence's (patient) length is N (N visits), it predicts element at time N+1.
        :param patient_seq: list of lists or list of lists of lists, patient sequence o sequences
        :param future: bool, if you want to predict next visit or last visit (for performance calculation)
        :param batchSize: int, default 100
        :return: list (one single patients) or list of lists (multiple patients) containing one prediction for each patient.
        """

        # import dict to convert patient's ICD9 to the internal integer representation
        dict_icd2int = pickle.load(open(self.ICD9_to_int_dict, 'rb'))
        dict_int2icd = {v: k for k, v in dict_icd2int.items()}
        if self.CCS_to_int_dict is not None:
            dict_ccs2int = pickle.load(open(self.CCS_to_int_dict, 'rb'))
            dict_int2ccs = {v: k for k, v in dict_ccs2int.items()}

        options = locals().copy()
        options['hiddenDimSize'] = self.hiddenDimSize

        # load model
        try:
            models = np.load(self.modelFile)
        except:
            print(f'There is no modelFile, current modelFile path is {self.modelFile}')

        tparams = init_trained_theano_params(models)

        # build model
        x, mask, codePred = old_build_trained_model(tparams, options)
        predict_code = theano.function(inputs=[x, mask], outputs=codePred, name='predict_code')

        options['inputDimSize'] = models['W_emb'].shape[0]
        options['numClass'] = models['b_output'].shape[0]

        def pred_vec_single_patient(new_patient, codeResults, lengths):
            tempY = [new_patient]
            predVec = []

            for i in range(codeResults.shape[1]):
                tensorMatrix = codeResults[:, i, :]
                thisY = tempY[i][1:]
                for timeIndex in range(lengths[i]):
                    output = tensorMatrix[timeIndex]
                    predVec.append(list(zip(*heapq.nlargest(30, enumerate(output), key=operator.itemgetter(1))))[0])
            return predVec[-1]

        def pred_vec_multiple_patients(new_patient_seqs, codeResults, lengths):
            tempY = new_patient_seqs
            predVec = []
            predVecs = []
            for i in range(codeResults.shape[1]):
                # print(f'codeResults.shape[1] = {i}')
                tensorMatrix = codeResults[:, i, :]
                thisY = tempY[i][1:]
                for timeIndex in range(lengths[i]):
                    output = tensorMatrix[timeIndex]
                    predVec.append(list(zip(*heapq.nlargest(30, enumerate(output), key=operator.itemgetter(1))))[0])
                predVecs.append(predVec[-1])
                # print(predVec[-1])
                # print()
            # print(predVec)
            return predVecs

        # check if we have one patient or multiple patients in patient_seq
        multiple_patient = isinstance(patient_seq[0][0], list)

        if not multiple_patient:
            # transform the patient in the internal integer-representation of ICD9 codes useful for doctorAI
            new_patient = [[dict_icd2int[x] for x in visit] for visit in patient_seq]

            if future:
                ### uncomment the following line to predict the code at t+1 where t is the last visit in the dataset:
                xf, maskf, lengthsf = newpadMatrixWithoutTime([new_patient], options)
            else:
                ### uncomment the following line to predict the code at t where t is the last visit in the dataset (for test purposes):
                xf, maskf, lengthsf = padtrainedMatrixWithoutTime([new_patient], options)

            future_codeResults = predict_code(xf, maskf)
            future_predVec = pred_vec_single_patient(new_patient, future_codeResults, lengthsf)

            if self.CCS_to_int_dict is not None:
                return [dict_int2ccs[x] for x in future_predVec]
            else:
                return [dict_int2icd[x] for x in future_predVec]

        else:
            # transform each patient in the internal integer-representation of ICD9 codes useful for doctorAI
            new_patients = [[[dict_icd2int[x] for x in visit] for visit in patient] for patient in patient_seq]

            if future:
                ### uncomment the following line to predict the code at t+1 where t is the last visit in the dataset:
                xf, maskf, lengthsf = newpadMatrixWithoutTime(new_patients, options)
            else:
                ### uncomment the following line to predict the code at t where t is the last visit in the dataset (for test purposes):
                xf, maskf, lengthsf = padtrainedMatrixWithoutTime(new_patients, options)

            future_codeResults = predict_code(xf, maskf)
            future_predVec = pred_vec_multiple_patients(new_patients, future_codeResults, lengthsf)

            if self.CCS_to_int_dict is not None:
                return [[dict_int2ccs[x] for x in prediction] for prediction in future_predVec]
            else:
                return [[dict_int2icd[x] for x in prediction] for prediction in future_predVec]


    def predict_all_visits(self, patient_seq, batchSize=100):
        """
        This function returns the predictions of doctorAI on each element (visit) of the sequence (patient),
        so if a patient has N visits {0,1,2,...,N-1} this functions return N-1 predictions of the visits {1,2,...,N-1}
        :param patient_seq: list of lists or list of lists of lists, patient sequence o sequences
        :param batchSize: int, default 100
        :return:
        """

        # import dict to convert patient's ICD9 to the internal integer representation
        dict_icd2int = pickle.load(open(self.ICD9_to_int_dict, 'rb'))
        dict_int2icd = {v: k for k, v in dict_icd2int.items()}
        if self.CCS_to_int_dict is not None:
            dict_ccs2int = pickle.load(open(self.CCS_to_int_dict, 'rb'))
            dict_int2ccs = {v: k for k, v in dict_ccs2int.items()}

        options = locals().copy()
        options['hiddenDimSize'] = self.hiddenDimSize

        # load model
        try:
            models = np.load(self.modelFile)
        except:
            print(f'There is no modelFile, current modelFile path is {self.modelFile}')

        options['inputDimSize'] = models['W_emb'].shape[0]
        options['numClass'] = models['b_output'].shape[0]

        # build model
        tparams = init_trained_theano_params(models)
        x, mask, rawPred, codePred = build_trained_model(tparams, options)

        # check if we have one patient or multiple patients in patient_seq
        multiple_patient = isinstance(patient_seq[0][0], list)

        ################# FINO QUI OK

        #### DEFINE FUNCTION PREDICT CODE
        #predict_raw = theano.function(inputs=[x, mask], outputs=rawPred, name='predict_raw')
        predict_code = theano.function(inputs=[x, mask], outputs=codePred, name='predict_code')

        #### LOAD SORTED DATA
        #testSet = new_test_load_data(visit_test, label_test, pid_test, files=False)

        #### SET EMPTY predVec, trueVec

        topkpred = []

        ### DECIDE BATCHES
        n_batches = int(np.ceil(float(len(patient_seq)) / float(batchSize)))

        for batchIndex in range(n_batches):
            # tempX = testSet[0][batchIndex * batchSize: (batchIndex + 1) * batchSize]
            # tempY = testSet[1][batchIndex * batchSize: (batchIndex + 1) * batchSize]
            # pids_ = testSet[2][batchIndex * batchSize: (batchIndex + 1) * batchSize]
            if multiple_patient:
                new_patients = [[[dict_icd2int[x] for x in visit] for visit in patient] for patient in patient_seq]
                tempX = new_patients[batchIndex * batchSize: (batchIndex + 1) * batchSize]
                x, mask, lengths = padtrainedMatrixWithoutTime(tempX, options)
            else:
                new_patient = [[dict_icd2int[x] for x in visit] for visit in patient_seq]
                tempX=new_patient
                x, mask, lengths = padtrainedMatrixWithoutTime([tempX], options)

            #x, mask, lengths = padtrainedMatrixWithoutTime(tempX, options)
            #rawResults = predict_raw(x, mask)
            codeResults = predict_code(x, mask)

            for i in range(codeResults.shape[1]):
                #pid = pids_[i]
                #rawMatrix = rawResults[:, i, :]
                codeMatrix = codeResults[:, i, :]

                # prendi la GT (thisY)
                #thisY = tempY[i][1:]
                # prendi le hadm_id di questo paziente
                #hadm_id_list = hadm_ids[pid]

                for timeIndex in range(lengths[i]):

                    #hadm_id = hadm_id_list[timeIndex]

                    #if len(thisY[timeIndex]) == 0: continue
                    # trueDict[pid,hadm_id].append(thisY[timeIndex])
                    #trueDict[pid, hadm_id] = thisY[timeIndex]
                    #
                    #raw = rawMatrix[timeIndex]
                    # rawDict[pid,hadm_id].append(list(raw))
                    #rawDict[pid, hadm_id] = np.array(raw)
                    #
                    output = codeMatrix[timeIndex]
                    # predDict[pid,hadm_id].append(list(output))
                    #predDict[pid, hadm_id] = np.array(output)
                    topk_output = list(zip(*heapq.nlargest(30, enumerate(output), key=operator.itemgetter(1))))[0]
                    # topkDict[pid,hadm_id].append(list(topk_output))
                    #topkDict[pid, hadm_id] = list(topk_output)
                    topkpred.append(topk_output)

        if not multiple_patient:
            if self.CCS_to_int_dict is not None:
                return [dict_int2ccs[x] for x in topkpred[0]]
            else:
                return [dict_int2icd[x] for x in topkpred[0]]
        else:
            if self.CCS_to_int_dict is not None:
                return [[dict_int2ccs[x] for x in visit] for visit in topkpred]
            else:
                return [[dict_int2icd[x] for x in visit] for visit in topkpred]


    def predict(self, patient_seq, batchSize=100):

        """
        This function returns the prediction of doctorAI for the last visit of each patient.
        If a sequence's (patient) length is N (N visits), it predicts element at time N.
        :param patient_seq: list of lists or list of lists of lists, patient sequence o sequences
        :param future: bool, if you want to predict next visit or last visit (for performance calculation)
        :param batchSize: int, default 100
        :return: list (one single patients) or list of lists (multiple patients) containing one prediction for each patient.
        """

        # import dict to convert patient's ICD9 to the internal integer representation
        dict_icd2int = pickle.load(open(self.ICD9_to_int_dict, 'rb'))
        dict_int2icd = {v: k for k, v in dict_icd2int.items()}
        if self.CCS_to_int_dict is not None:
            dict_ccs2int = pickle.load(open(self.CCS_to_int_dict, 'rb'))
            dict_int2ccs = {v: k for k, v in dict_ccs2int.items()}

        options = locals().copy()
        options['hiddenDimSize'] = self.hiddenDimSize

        # load model
        try:
            models = np.load(self.modelFile)
        except:
            print(f'There is no modelFile, current modelFile path is {self.modelFile}')

        options['inputDimSize'] = models['W_emb'].shape[0]
        options['numClass'] = models['b_output'].shape[0]

        # build model
        tparams = init_trained_theano_params(models)
        x, mask, rawPred, codePred = build_trained_model(tparams, options)

        # check if we have one patient or multiple patients in patient_seq
        multiple_patient = isinstance(patient_seq[0][0], list)

        ################# FINO QUI OK

        #### DEFINE FUNCTION PREDICT CODE
        #predict_raw = theano.function(inputs=[x, mask], outputs=rawPred, name='predict_raw')
        predict_code = theano.function(inputs=[x, mask], outputs=codePred, name='predict_code')

        #### LOAD SORTED DATA
        #testSet = new_test_load_data(visit_test, label_test, pid_test, files=False)

        #### SET EMPTY predVec, trueVec

        topkpred = []

        ### DECIDE BATCHES
        n_batches = int(np.ceil(float(len(patient_seq)) / float(batchSize)))

        for batchIndex in range(n_batches):
            # tempX = testSet[0][batchIndex * batchSize: (batchIndex + 1) * batchSize]
            # tempY = testSet[1][batchIndex * batchSize: (batchIndex + 1) * batchSize]
            # pids_ = testSet[2][batchIndex * batchSize: (batchIndex + 1) * batchSize]
            if multiple_patient:
                new_patients = [[[dict_icd2int[x] for x in visit] for visit in patient] for patient in patient_seq]
                tempX = new_patients[batchIndex * batchSize: (batchIndex + 1) * batchSize]
                x, mask, lengths = padtrainedMatrixWithoutTime(tempX, options)
            else:
                new_patient = [[dict_icd2int[x] for x in visit] for visit in patient_seq]
                tempX=new_patient
                x, mask, lengths = padtrainedMatrixWithoutTime([tempX], options)

            #x, mask, lengths = padtrainedMatrixWithoutTime(tempX, options)
            #rawResults = predict_raw(x, mask)
            codeResults = predict_code(x, mask)

            for i in range(codeResults.shape[1]):
                #pid = pids_[i]
                #rawMatrix = rawResults[:, i, :]
                codeMatrix = codeResults[:, i, :]
                # prendi la GT (thisY)
                #thisY = tempY[i][1:]
                # prendi le hadm_id di questo paziente
                #hadm_id_list = hadm_ids[pid]

                for timeIndex in range(lengths[i]):
                    #loop over the visits
                    #hadm_id = hadm_id_list[timeIndex]

                    #if len(thisY[timeIndex]) == 0: continue
                    # trueDict[pid,hadm_id].append(thisY[timeIndex])
                    #trueDict[pid, hadm_id] = thisY[timeIndex]
                    #
                    #raw = rawMatrix[timeIndex]
                    # rawDict[pid,hadm_id].append(list(raw))
                    #rawDict[pid, hadm_id] = np.array(raw)
                    #
                    output = codeMatrix[timeIndex]
                    # predDict[pid,hadm_id].append(list(output))
                    #predDict[pid, hadm_id] = np.array(output)
                    topk_output = list(zip(*heapq.nlargest(30, enumerate(output), key=operator.itemgetter(1))))[0]
                    # topkDict[pid,hadm_id].append(list(topk_output))
                    #topkDict[pid, hadm_id] = list(topk_output)

                #append only the last prediction of each patients
                topkpred.append(topk_output)


        if not multiple_patient:
            if self.CCS_to_int_dict is not None:
                return [dict_int2ccs[x] for x in topkpred[0]]
            else:
                return [dict_int2icd[x] for x in topkpred[0]]
        else:
            if self.CCS_to_int_dict is not None:
                return [[dict_int2ccs[x] for x in visit] for visit in topkpred]
            else:
                return [[dict_int2icd[x] for x in visit] for visit in topkpred]


    def test_doctorAI(self, modelFile, visit_test, label_test, hiddenDimSize, predictTime=False, timeFile='', batchSize=100, verbose=False):

        options = locals().copy()
        options['hiddenDimSize'] = self.hiddenDimSize
        options['embSize'] = self.embSize

        if len(timeFile) > 0:
            useTime = True
        else:
            useTime = False
        options['useTime'] = useTime

        # load model
        try:
            models = np.load(self.modelFile)
        except:
            print(f'There is no modelFile, current modelFile path is {self.modelFile}')

        tparams = init_trained_theano_params(models)

        print( 'build model ... ')

        if predictTime:
            #x, t, mask, codePred, timePred = build_trained_model(tparams, options)
            x, mask, codePred = old_build_trained_model(tparams, options)
            predict_code = theano.function(inputs=[x, t, mask], outputs=codePred, name='predict_code')
            predict_time = theano.function(inputs=[x, t, mask], outputs=timePred, name='predict_time')
        elif useTime:
            #x, t, mask, codePred = build_trained_model(tparams, options)
            x, mask, codePred = old_build_trained_model(tparams, options)
            predict_code = theano.function(inputs=[x, t, mask], outputs=codePred, name='predict_code')
        else:
            #x, mask, codePred = build_trained_model(tparams, options)
            x, mask, codePred = old_build_trained_model(tparams, options)
            predict_code = theano.function(inputs=[x, mask], outputs=codePred, name='predict_code')

        options['inputDimSize'] = models['W_emb'].shape[0]
        options['numClass'] = models['b_output'].shape[0]
        print('load data ... ')
        testSet = test_load_data(visit_test, label_test, timeFile)
        n_batches = int(np.ceil(float(len(testSet[0])) / float(batchSize)))
        print('done')

        predVec = []
        trueVec = []
        predTimeVec = []
        trueTimeVec = []
        iteration = 0

        for batchIndex in range(n_batches):
            tempX = testSet[0][batchIndex * batchSize: (batchIndex + 1) * batchSize]
            tempY = testSet[1][batchIndex * batchSize: (batchIndex + 1) * batchSize]
            if predictTime:
                tempT = testSet[2][batchIndex * batchSize: (batchIndex + 1) * batchSize]
                x, t, mask, lengths = padMatrixWithTimePrediction(tempX, tempT, options)
                codeResults = predict_code(x, t, mask)
                timeResults = predict_time(x, t, mask)
            elif useTime:
                tempT = testSet[2][batchIndex * batchSize: (batchIndex + 1) * batchSize]
                x, t, mask, lengths = padMatrixWithTime(tempX, tempT, options)
                codeResults = predict_code(x, t, mask)
            else:
                x, mask, lengths = padtrainedMatrixWithoutTime(tempX, options)
                codeResults = predict_code(x, mask)

            for i in range(codeResults.shape[1]):
                tensorMatrix = codeResults[:, i, :]
                thisY = tempY[i][1:]
                for timeIndex in range(lengths[i]):
                    if len(thisY[timeIndex]) == 0: continue
                    trueVec.append(thisY[timeIndex])
                    output = tensorMatrix[timeIndex]
                    predVec.append(list(zip(*heapq.nlargest(30, enumerate(output), key=operator.itemgetter(1))))[0])

            if predictTime:
                for i in range(timeResults.shape[1]):
                    timeVec = timeResults[:, i]
                    trueTimeVec.extend(tempT[i][1:])
                    for timeIndex in range(lengths[i]):
                        predTimeVec.append(timeVec[timeIndex])

            if (iteration % 10 == 0) and verbose: print(f'iteration:{iteration}/{n_batches}')
            iteration += 1
            if iteration == 10: break

        recall = recallTop(trueVec, predVec)
        print(f'recall@10:{recall[0]}, recall@20:{recall[1]}, recall@30:{recall[2]}')

        return