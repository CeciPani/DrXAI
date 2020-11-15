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
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from collections import defaultdict
from pathlib import Path


def _convert_to_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return dxStr[:4] + '.' + dxStr[4:]
        else: return dxStr
    else:
        if len(dxStr) > 3: return dxStr[:3] + '.' + dxStr[3:]
        else: return dxStr

def prepare_mimic(mimic_path, CCS_grouper_csv_file_path, output_path='./doctorAI_preprocessing/', CCS=False):
    """
    This function removes the admissions without diagnosis code and save the "cleaned tables in the same folder as the old tables"
    :param path: str, input path where the MIMIC tabels are stored (ADMISSIONS.csv and DIAGNOSES_ICD.csv)
     e.g. /home/user/venvs/drAI+/drAIplus/data/MIMIC_data/
    """

    ################################### REMOVING PROBLMES WITH MIMIC DATA ##############################################

    admissionFile = mimic_path + 'ADMISSIONS.csv'
    diagnosisFile = mimic_path + 'DIAGNOSES_ICD.csv'

    try:
        diag_csv = pd.read_csv(diagnosisFile)
    except:
        print(f'There is no "DIAGNOSES_ICD.csv" file in {mimic_path}')

    try:
        admission_csv = pd.read_csv(admissionFile)
    except:
        print(f'There is no "ADMISSIONS.csv" file in {mimic_path}')


    ratio_of_null = len(diag_csv[diag_csv.ICD9_CODE.isna()]) / float(len(diag_csv))
    print(f'{round(ratio_of_null,5)} % of diagnoses do not have the related ICD9 code associated (NaN values)')
    #drop the cases where there is no diagnosis code
    #remove the admissions where there is no ICD9 code
    print("dropping the cases where there is no ICD9 code in the DIAGNOSES_ICD table")
    clean_diag_df = diag_csv.dropna(subset=['ICD9_CODE'])

    clean_diag_df.to_csv(mimic_path + 'clean_DIAGNOSES_ICD.csv', index=False)
    clean_admission_df = admission_csv[admission_csv.HADM_ID.isin(clean_diag_df.HADM_ID)]
    clean_admission_df.to_csv(mimic_path + 'clean_ADMISSIONS.csv', index=False)
    print()
    print(f'Created "clean_DIAGNOSES_ICD.csv" and "clean_ADMISSIONS.csv" in {mimic_path}')

    admissionFile = mimic_path + 'clean_ADMISSIONS.csv'
    diagnosisFile = mimic_path + 'clean_DIAGNOSES_ICD.csv'

    ###################################### CRETING THE FILES FOR DOCTORAI TRAINING #####################################

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
        # dxStr = 'D_' + convert_to_icd9(tokens[4][1:-1]) ############## Uncomment this line and comment the line below, if you want to use the entire ICD9 digits.
        # dxStr = 'D_' + convert_to_icd9(tokens[4])
        dxStr = _convert_to_icd9(tokens[4])
        # dxStr = 'D_' + convert_to_3digit_icd9(tokens[4][1:-1])
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

    Path(output_path).mkdir(parents=True, exist_ok=True)
    np.save(output_path+'mimic_sequences', np.array(seqs))
    path_mapping = output_path+'ICD9_to_int_dict'
    pickle.dump(types, open(path_mapping, 'wb'), -1)
    n_unique_ICD9 = len(set([a for b in [a for b in newSeqs for a in b] for a in b]))
    print(f'number of unique ICD9 codes in mimic: {n_unique_ICD9}')
    visit_file = newSeqs

    if CCS:
        # the CCS grouper can be downloaded here: https://www.nber.org/data/icd-ccs-crosswalk.html
        #CCS_grouper_csv_file_path = os.path.join(os.path.dirname(__file__), 'dxicd2ccsxw.csv')
        #CCS_grouper_csv_file_path = input_path+'dxicd2ccsxw.csv'
        CCS_grouper_csv = pd.read_csv(CCS_grouper_csv_file_path+'dxicd2ccsxw.csv')
        new_dict_types = defaultdict(int)

        for item in types.items():
            clean_ICD9 = item[0].replace('.', '')
            if len(clean_ICD9) == 0:
                clean_ICD9 = 'unknown'
            new_dict_types[clean_ICD9] = item[1]

        types_df = pd.Series(new_dict_types).reset_index().rename(columns={'index': 'icd', 0: 'internal_code'})
        mapping = pd.merge(types_df, CCS_grouper_csv)[['ccs', 'internal_code']]
        mapping_dict = dict(zip(mapping.internal_code, mapping.ccs))
        newSeqs_ccs = [[list(np.vectorize(mapping_dict.get)(np.array(visit))) for visit in patient] for patient in newSeqs]

        n_unique_CCS = len(set([a for b in [a for b in newSeqs_ccs for a in b] for a in b]))
        print(f'number of unique CCS-grouper codes in mimic: {n_unique_CCS}')

        print('Converting strSeqs to intSeqs, and making CCStypes')
        CCStypes = {}
        newCCSSeqs = []
        for patient in newSeqs_ccs:
            newPatient = []
            for visit in patient:
                newVisit = []
                for code in visit:
                    if code in CCStypes:
                        newVisit.append(CCStypes[code])
                    else:
                        CCStypes[code] = len(CCStypes)
                        newVisit.append(CCStypes[code])
                newPatient.append(newVisit)
            newCCSSeqs.append(newPatient)

        path_CCS = output_path+'CCS_to_int_dict'
        pickle.dump(CCStypes, open(path_CCS, 'wb'), -1)
        label_file = newCCSSeqs
    else:
        label_file = newSeqs

    pickle.dump(visit_file, open(output_path + 'visit_complete', 'wb'), protocol=-1)
    pickle.dump(label_file, open(output_path + 'label_complete', 'wb'), protocol=-1)

    visit_train, visit_Test, label_train, label_Test = train_test_split(visit_file, label_file, test_size=0.33, random_state=42)
    visit_test, visit_valid, label_test, label_valid = train_test_split(visit_Test, label_Test, test_size=0.33, random_state=42)

    pickle.dump(visit_train, open(output_path + 'visit.train', 'wb'), protocol=-1)
    pickle.dump(visit_valid, open(output_path + 'visit.valid', 'wb'), protocol=-1)
    pickle.dump(visit_test, open(output_path + 'visit.test', 'wb'), protocol=-1)
    pickle.dump(label_train, open(output_path + 'label.train', 'wb'), protocol=-1)
    pickle.dump(label_valid, open(output_path + 'label.valid', 'wb'), protocol=-1)
    pickle.dump(label_test, open(output_path + 'label.test', 'wb'), protocol=-1)
    print()
    print(f'You can use the files visit and label (.train/.valid/.test) in {output_path} to train doctorAI:')
    print(f'seqFile="{output_path}visit"')
    print(f'labelFile="{output_path}label"')
    print(f'outFile="<output_path>/trained_drAI_model"')
    if CCS:
        print(f'dr = doctorai.DoctorAI(ICD9_to_int_dict="{output_path}/ICD9_to_int_dict",CCS_to_int_dict="{output_path}/CCS_to_int_dict",verbose=True)')
        print(f"dr.train_doctorAI(seqFile=seqFile, inputDimSize={n_unique_ICD9}, labelFile=labelFile, numClass={n_unique_CCS}, outFile=outFile, max_epochs=50)")
    else:
        print(f'dr = doctorai.DoctorAI(ICD9_to_int_dict="{output_path}/ICD9_to_int_dict",verbose=True)')
        print(f"dr.train_doctorAI(seqFile=seqFile, inputDimSize={n_unique_ICD9}, labelFile=labelFile, numClass={n_unique_ICD9}, outFile=outFile, max_epochs=50)")

    return


def prepare_mimic4(admissionFile_path, diagnosisFile_path, dict_path, output_path, CCS=False):

    admissionFile = admissionFile_path
    diagnosisFile = diagnosisFile_path

    try:
        diag_csv = pd.read_csv(diagnosisFile_path)
    except:
        print(f'There is no diagnoses file in {diagnosisFile_path}')

    try:
        admission_csv = pd.read_csv(admissionFile_path)
    except:
        print(f'There is no admission file in {admissionFile_path}')

    ratio_of_null = len(diag_csv[diag_csv.icd_code.isna()]) / float(len(diag_csv))

    if ratio_of_null > 0:
        print(f'{round(ratio_of_null, 5)} % of diagnoses do not have the related ICD9 code associated (NaN values)')
        # drop the cases where there is no diagnosis code
        # remove the admissions where there is no ICD9 code

        print("dropping the cases where there is no ICD9 code in the DIAGNOSES_ICD table")
        clean_diag_df = diag_csv.dropna(subset=['icd_code'])

        clean_diag_df.to_csv(output_path + 'clean_DIAGNOSES_ICD9.csv', index=False)
        clean_admission_df = admission_csv[admission_csv.hadm_id.isin(clean_diag_df.hadm_id)]
        clean_admission_df.to_csv(output_path + 'clean_ADMISSIONS_ICD9.csv', index=False)
        print()
        print(f'Created "clean_DIAGNOSES_ICD.csv" and "clean_ADMISSIONS.csv"')

        admissionFile = output_path + 'clean_ADMISSIONS_ICD9.csv'
        diagnosisFile = output_path + 'clean_DIAGNOSES_ICD9.csv'
    else:
        print('No diagnoses without related ICD9 (no NaN values)')
        admissionFile = admissionFile_path
        diagnosisFile = diagnosisFile_path

    print('Building pid-admission mapping, admission-date mapping')
    pidAdmMap = {}
    admDateMap = {}
    infd = open(admissionFile, 'r')
    infd.readline()

    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[0])
        admId = int(tokens[1])
        admTime = datetime.strptime(tokens[2], '%Y-%m-%d %H:%M:%S')
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
        admId = int(tokens[1])
        # dxStr = 'D_' + convert_to_icd9(tokens[4][1:-1]) ############## Uncomment this line and comment the line below, if you want to use the entire ICD9 digits.
        # dxStr = 'D_' + convert_to_icd9(tokens[4])
        dxStr = _convert_to_icd9(tokens[3].replace(' ', ''))
        # dxStr = 'D_' + convert_to_3digit_icd9(tokens[4][1:-1])
        if admId in admDxMap:
            admDxMap[admId].append(dxStr.replace(' ', ''))
        else:
            admDxMap[admId] = [dxStr.replace(' ', '')]
    infd.close()

    print('Building pid-sortedVisits mapping')
    pidSeqMap = {}
    for pid, admIdList in pidAdmMap.items():
        if len(admIdList) < 2:
            print(f'pid escluso: {pid}')
            continue
        sortedList = sorted([(admDateMap[admId], admDxMap[admId], admId) for admId in admIdList])
        pidSeqMap[pid] = sortedList

    print('Building pids, dates, strSeqs')
    pids = []
    dates = []
    # seqs = []
    seqs = {}
    had_ids = {}

    for pid, visits in pidSeqMap.items():
        pids.append(pid)
        seq = []
        date = []
        hadm_id = []
        for visit in visits:
            date.append(visit[0])
            seq.append(visit[1])
            hadm_id.append(visit[2])
        dates.append(date)
        # seqs.append(seq)
        seqs[pid] = [seq, hadm_id]

    print('Converting strSeqs to intSeqs, and making types')
    types = {}
    # newSeqs = []
    newSeqs = {}
    for pid, patient in seqs.items():
        # newPatient = [pid]
        newPatient = []
        for visit in patient[0]:
            # print(visit)
            newVisit = []
            for code in visit:
                if code in types:
                    newVisit.append(types[code])
                else:
                    types[code] = len(types)
                    newVisit.append(types[code])
            newPatient.append(newVisit)
        # newSeqs.append(newPatient)
        newSeqs[pid] = [newPatient, patient[1]]

    # np.save(output_path+'mimic_sequences', np.array(seqs))
    pickle.dump(seqs, open(output_path + 'mimic_sequences', 'wb'), -1)
    path_mapping = dict_path + 'ICD9_to_int_dict'
    pickle.dump(types, open(path_mapping, 'wb'), -1)

    only_visits_newSeqs = np.array(list(newSeqs.values()))[:, 0]
    n_uniques_ICD9 = len(set([a for b in [a for b in only_visits_newSeqs for a in b] for a in b]))
    print(f'number of unique ICD9 codes in mimic: {n_uniques_ICD9}')
    visit_file = newSeqs

    if CCS:
        # the CCS grouper can be downloaded here: https://www.nber.org/data/icd-ccs-crosswalk.html
        # CCS_grouper_csv_file_path = os.path.join(os.path.dirname(__file__), 'dxicd2ccsxw.csv')
        CCS_grouper_csv_file_path = dict_path + 'dxicd2ccsxw.csv'
        CCS_grouper_csv = pd.read_csv(CCS_grouper_csv_file_path)
        new_dict_types = defaultdict(int)

        for item in types.items():
            clean_ICD9 = item[0].replace('.', '')
            if len(clean_ICD9) == 0:
                clean_ICD9 = 'unknown'
            new_dict_types[clean_ICD9] = item[1]

        types_df = pd.Series(new_dict_types).reset_index().rename(columns={'index': 'icd', 0: 'internal_code'})
        mapping = pd.merge(types_df, CCS_grouper_csv)[['ccs', 'internal_code']]
        mapping_dict = dict(zip(mapping.internal_code, mapping.ccs))
        newSeqs_ccs = {pid: [[[mapping_dict[code] for code in visit] for visit in patient[0]], patient[1]] for
                       pid, patient in newSeqs.items()}

        only_visits_newSeqs_ccs = np.array(list(newSeqs_ccs.values()))[:, 0]
        n_unique_CCS = len(set([a for b in [a for b in only_visits_newSeqs_ccs for a in b] for a in b]))
        print(f'number of unique CCS-grouper codes in mimic: {n_unique_CCS}')

        print('Converting strSeqs to intSeqs, and making CCStypes')
        CCStypes = {}
        # newCCSSeqs = []
        newCCSSeqs = {}
        for pid, patient in newSeqs_ccs.items():
            newPatient = []
            for visit in patient[0]:
                newVisit = []
                for code in visit:
                    if code in CCStypes:
                        newVisit.append(CCStypes[code])
                    else:
                        CCStypes[code] = len(CCStypes)
                        newVisit.append(CCStypes[code])
                newPatient.append(newVisit)
            # newCCSSeqs.append(newPatient)
            newCCSSeqs[pid] = [newPatient, patient[1]]

        path_CCS = dict_path + 'CCS_to_int_dict'
        pickle.dump(CCStypes, open(path_CCS, 'wb'), -1)
        label_file = newCCSSeqs

    else:
        # label_file = [patient[1:] for patient in newSeqs]
        label_file = newSeqs

    ### FINO QUI TUTTO OK

    pickle.dump(visit_file, open(output_path + 'visit_complete', 'wb'), protocol=-1)
    pickle.dump(label_file, open(output_path + 'label_complete', 'wb'), protocol=-1)

    # creating TRANÌNING VALIDATION and TEST sets
    visit_train, visit_Test, label_train, label_Test = [split.to_dict() for split in
                                                        train_test_split(pd.Series(visit_file), pd.Series(label_file),
                                                                         test_size=0.33, random_state=42)]
    visit_test, visit_valid, label_test, label_valid = [split.to_dict() for split in
                                                        train_test_split(pd.Series(visit_Test), pd.Series(label_Test),
                                                                         test_size=0.33, random_state=42)]

    # saving TRAINING SET
    pickle.dump(visit_train, open(output_path + 'complete_visit.train', 'wb'), protocol=-1)
    pickle.dump(list(np.array(list(visit_train.values()))[:, 0]), open(output_path + 'visit.train', 'wb'), protocol=-1)
    pickle.dump(list(np.array(list(visit_train.values()))[:, 1]), open(output_path + 'prova_hadmid.train', 'wb'),
                protocol=-1)
    pickle.dump(list(np.array(list(label_train.values()))[:, 0]), open(output_path + 'label.train', 'wb'), protocol=-1)

    # saving VALIDATION SET
    pickle.dump(visit_valid, open(output_path + 'complete_visit.valid', 'wb'), protocol=-1)
    pickle.dump(list(np.array(list(visit_valid.values()))[:, 0]), open(output_path + 'visit.valid', 'wb'), protocol=-1)
    pickle.dump(list(np.array(list(visit_valid.values()))[:, 1]), open(output_path + 'hadmid.valid', 'wb'), protocol=-1)
    pickle.dump(list(np.array(list(label_valid.values()))[:, 0]), open(output_path + 'label.valid', 'wb'), protocol=-1)
    # saving TEST SET
    pickle.dump(visit_test, open(output_path + 'complete_visit.test', 'wb'), protocol=-1)
    pickle.dump(list(np.array(list(visit_test.values()))[:, 0]), open(output_path + 'visit.test', 'wb'), protocol=-1)
    pickle.dump(list(np.array(list(visit_test.values()))[:, 1]), open(output_path + 'hadmid.test', 'wb'), protocol=-1)
    pickle.dump(list(np.array(list(label_test.values()))[:, 0]), open(output_path + 'label.test', 'wb'), protocol=-1)

    print()
    print(f'You can use the files visit and label (.train/.valid/.test) in {output_path} to train doctorAI:')
    print(f'seqFile="{output_path}visit"')
    print(f'labelFile="{output_path}label"')
    print(f'outFile="<output_path>/trained_drAI_model"')
    if CCS:
        print()
        print(
            f'dr = doctorai.DoctorAI(ICD9_to_int_dict="{dict_path}ICD9_to_int_dict",\nCCS_to_int_dict={dict_path}CCS_to_int_dict\nverbose=True)\n')
        print(
            f"dr.train_doctorAI(seqFile=seqFile,\ninputDimSize={n_uniques_ICD9},\nlabelFile=labelFile,\nnumClass={n_unique_CCS},\noutFile=outFile,\nmax_epochs=50)")
    else:
        print()
        print(f'dr = doctorai.DoctorAI(ICD9_to_int_dict="{dict_path}ICD9_to_int_dict",\nverbose=True)\n')
        print(
            f"dr.train_doctorAI(seqFile=seqFile,\ninputDimSize={n_uniques_ICD9},\nlabelFile=labelFile,\nnumClass={n_uniques_ICD9},\noutFile=outFile,\nmax_epochs=50)")

    return
