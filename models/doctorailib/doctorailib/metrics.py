import numpy as np
from .utils_train import xrange, padMatrixWithTimePrediction, padMatrixWithoutTime, padMatrixWithTime

def calculate_auc(test_model, dataset, options):
    inputDimSize = options['inputDimSize']
    numClass = options['numClass']
    batchSize = options['batchSize']
    useTime = options['useTime']
    predictTime = options['predictTime']
    n_batches = int(np.ceil(float(len(dataset[0])) / float(batchSize)))
    aucSum = 0.0
    dataCount = 0.0
    for index in xrange(n_batches):
        batchX = dataset[0][index * batchSize:(index + 1) * batchSize]
        batchY = dataset[1][index * batchSize:(index + 1) * batchSize]
        if predictTime:
            batchT = dataset[2][index * batchSize:(index + 1) * batchSize]
            x, y, t, t_label, mask, lengths = padMatrixWithTimePrediction(batchX, batchY, batchT, options)
            auc = test_model(x, y, t, t_label, mask, lengths)
        elif useTime:
            batchT = dataset[2][index * batchSize:(index + 1) * batchSize]
            x, y, t, mask, lengths = padMatrixWithTime(batchX, batchY, batchT, options)
            auc = test_model(x, y, t, mask, lengths)
        else:
            x, y, mask, lengths = padMatrixWithoutTime(batchX, batchY, options)
            auc = test_model(x, y, mask, lengths)
        aucSum += auc * len(batchX)
        dataCount += float(len(batchX))
    return aucSum / dataCount


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