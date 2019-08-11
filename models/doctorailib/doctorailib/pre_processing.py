# This script processes MIMIC-III dataset and builds longitudinal diagnosis records for patients with at least two visits.
# The output data are cPickled, and suitable for training Doctor AI or RETAIN
# Written by Edward Choi (mp2893@gatech.edu) and modified by Cecilia Panigutti (cecilia.panigutti@sns.it)
# Usage: Put this script to the foler where MIMIC-III CSV files are located. Then execute the below command.
# python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv <output file>

# Output files
# <output file>.pids: List of unique Patient IDs. Used for intermediate processing
# <output file>.dates: List of List of Python datetime objects. The outer List is for each patient. The inner List is for each visit made by each patient
# <output file>.seqs: List of List of List of integer diagnosis codes. The outer List is for each patient. The middle List contains visits made by each patient. The inner List contains the integer diagnosis codes that occurred in each visit
# <output file>.types: Python dictionary that maps string diagnosis codes to integer diagnosis codes.

import os
import pickle
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split


def _convert_to_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return dxStr[:4] + '.' + dxStr[4:]
        else: return dxStr
    else:
        if len(dxStr) > 3: return dxStr[:3] + '.' + dxStr[3:]
        else: return dxStr

def _convert_to_3digit_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return dxStr[:4]
        else: return dxStr
    else:
        if len(dxStr) > 3: return dxStr[:3]
        else: return dxStr

def _prepare_mimic(path):
    """
    This function removes the admissions without diagnosis code and save the "cleaned tables in the same folder as the old tables"
    :param path: str, input path where the MIMIC tabels are stored (ADMISSIONS.csv and DIAGNOSES_ICD.csv)
     e.g. /home/user/venvs/drAI+/drAIplus/data/MIMIC_data/
    """
    admissionFile = path + 'ADMISSIONS.csv'
    diagnosisFile = path + 'DIAGNOSES_ICD.csv'

    diag_csv = pd.read_csv(diagnosisFile)
    admission_csv = pd.read_csv(admissionFile)

    ratio_of_null = len(diag_csv[diag_csv.ICD9_CODE.isna()]) / float(len(diag_csv))
    print(f'{ratio_of_null}% of diagnoses do not have the related ICD9 code associated (NaN values)')
    #drop the cases where there is no diagnosis code
    #remove the admissions where there is no ICD9 code
    print("dropping the cases where there is no ICD9 code in the DIAGNOSES_ICD table")
    clean_diag_df = diag_csv.dropna(subset=['ICD9_CODE'])

    clean_diag_df.to_csv(path + 'clean_DIAGNOSES_ICD.csv', index=False)
    clean_admission_df = admission_csv[admission_csv.HADM_ID.isin(clean_diag_df.HADM_ID)]
    clean_admission_df.to_csv(path + 'clean_ADMISSIONS.csv', index=False)

    return


def _css_preprocessing():

    # dictionary with key the ICD9 code and value the CSS grouper
    css_grouper = pickle.load(open("./conv_dict", "rb"))
    css_grouper

    return

def _pids_dates_seqs_types(admissionFile,diagnosisFile,output_path,ccs):

    print('Building pid-admission mapping, admission-date mapping')
    pidAdmMap = {}
    admDateMap = {}
    infd = open(admissionFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[1])
        admId = int(tokens[2])
        admTime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
        admDateMap[admId] = admTime
        if pid in pidAdmMap:
            pidAdmMap[pid].append(admId)
        else:
            pidAdmMap[pid] = [admId]
    infd.close()

    print('Building admission-dxList mapping')
    admDxMap = {}
    infd = open(diagnosisFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        ##########debugging##########################
        # print(tokens)
        # print(tokens[4])
        #############################################
        admId = int(tokens[2])
        # dxStr = 'D_' + _convert_to_icd9(tokens[4][1:-1]) ############## Uncomment this line and comment the line below, if you want to use the entire ICD9 digits.
        # dxStr = 'D_' + _convert_to_icd9(tokens[4])
        dxStr = _convert_to_icd9(tokens[4])
        # dxStr = 'D_' + _convert_to_3digit_icd9(tokens[4][1:-1])
        if admId in admDxMap:
            admDxMap[admId].append(dxStr)
        else:
            admDxMap[admId] = [dxStr]
    infd.close()

    print('Building pid-sortedVisits mapping')
    pidSeqMap = {}
    for pid, admIdList in pidAdmMap.items():
        if len(admIdList) < 2: continue
        sortedList = sorted([(admDateMap[admId], admDxMap[admId]) for admId in admIdList])
        pidSeqMap[pid] = sortedList

    print('Building pids, dates, strSeqs')
    pids = []
    dates = []
    seqs = []
    for pid, visits in pidSeqMap.items():
        pids.append(pid)
        seq = []
        date = []
        for visit in visits:
            date.append(visit[0])
            seq.append(visit[1])
        dates.append(date)
        seqs.append(seq)

    print('Converting strSeqs to intSeqs, and making types')
    types = {}
    newSeqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if code in types:
                    newVisit.append(types[code])
                else:
                    types[code] = len(types)
                    newVisit.append(types[code])
            newPatient.append(newVisit)
        newSeqs.append(newPatient)


    flat_list = []
    for patient in newSeqs:
        for visit in patient:
            for code in visit:
                flat_list.append(code)

    print(f'number of uniques codes: {len(set(flat_list))}')

    pickle.dump(pids, open(output_path + '.pids', 'wb'), -1)
    pickle.dump(dates, open(output_path + '.dates', 'wb'), -1)
    pickle.dump(newSeqs, open(output_path + '.seqs', 'wb'), -1)
    pickle.dump(types, open(output_path + '.types', 'wb'), -1)

    print('creating visit and label ".train", ".valid", and ".test')
    #"visit file" is clean_mimic.seqs
    visit_file = seqs
    #"label file" could be the same as the visit file, or a grouped version using CCS groupers
    if ccs: #if we want to group the labels
        newCCSseq = _css_preprocessing()
        label_file = newCCSseq
    else:
        label_file = seqs

    visit_train, visit_Test, label_train, label_Test = train_test_split(visit_file, label_file, test_size=0.33,
                                                                        random_state=42)
    visit_test, visit_valid, label_test, label_valid = train_test_split(visit_Test, label_Test, test_size=0.33,
                                                                        random_state=42)

    pickle.dump(visit_train, open(output_path + 'visit.train', 'wb'), -1)
    pickle.dump(visit_valid, open(output_path + 'visit.valid', 'wb'), -1)
    pickle.dump(visit_test, open(output_path + 'visit.test', 'wb'), -1)

    pickle.dump(label_train, open(output_path + 'label.train', 'wb'), -1)
    pickle.dump(label_valid, open(output_path + 'label.valid', 'wb'), -1)
    pickle.dump(label_test, open(output_path + 'label.test', 'wb'), -1)

    len(visit_train), len(visit_test), len(visit_valid)

    return


def mimic_preprocessing(path, output_path, ccs=False):
    """
    :param path: str, input path where the MIMIC tabels are stored (ADMISSIONS.csv and DIAGNOSES_ICD.csv)
     e.g. /home/user/venvs/drAI+/drAIplus/data/MIMIC_data/
    :param output_path: str, output path where you want to store the files for doctorAI training
     e.g. /home/user/venvs/drAI+/drAIplus/data/doctorAI_preprocessing_output/
    :param ccs: default False, switch to True if you want to perfrom a CCS grouper on the sequences
    :return:
    """

    #checking if the ADMISSIONS.csv and DIAGNOSES_ICD.csv have already been cleaned
    admission_exists = os.path.isfile(path+'clean_ADMISSIONS.csv')
    diagnosis_exists = os.path.isfile(path+'clean_DIAGNOSES_ICD.csv')

    admissionFile = path + 'clean_ADMISSIONS.csv'
    diagnosisFile = path + 'clean_DIAGNOSES_ICD.csv'

    if admission_exists & diagnosis_exists: #if the clean tables exist
        # create the train and test files (sequences)
        _pids_dates_seqs_types(admissionFile,diagnosisFile,output_path, ccs)
        return
    else:  #if the clean tables do not exist
        #create the clean tables
        _prepare_mimic(path)
        #create the train and test files (sequences)
        _pids_dates_seqs_types(admissionFile,diagnosisFile,output_path,ccs)
        return

