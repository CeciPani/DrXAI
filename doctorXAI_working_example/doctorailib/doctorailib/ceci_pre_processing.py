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

import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from collections import defaultdict

def _convert_to_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return dxStr[:4] + '.' + dxStr[4:]
        else: return dxStr
    else:
        if len(dxStr) > 3: return dxStr[:3] + '.' + dxStr[3:]
        else: return dxStr


def prepare_mimic(mimic_path, input_path, output_path, CSS=False, test_set_size = 0.33):
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
    print(f'{round(ratio_of_null, 5)} % of diagnoses do not have the related ICD9 code associated (NaN values)')
    # drop the cases where there is no diagnosis code
    # remove the admissions where there is no ICD9 code
    print("dropping the cases where there is no ICD9 code in the DIAGNOSES_ICD table")
    clean_diag_df = diag_csv.dropna(subset=['ICD9_CODE'])

    clean_diag_df.to_csv(mimic_path + 'clean_DIAGNOSES_ICD.csv', index=False)
    clean_admission_df = admission_csv[admission_csv.HADM_ID.isin(clean_diag_df.HADM_ID)]
    clean_admission_df.to_csv(mimic_path + 'clean_ADMISSIONS.csv', index=False)
    print()
    print(f'Created "clean_DIAGNOSES_ICD.csv" and "clean_ADMISSIONS.csv" in {mimic_path}')

    admissionFile = mimic_path + 'clean_ADMISSIONS.csv'
    diagnosisFile = mimic_path + 'clean_DIAGNOSES_ICD.csv'

    ###################################### CREATING THE FILES FOR DOCTORAI TRAINING #####################################

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
    for pid, patient in zip(pids, seqs):
        newPatient = [pid]
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

    np.save(mimic_path + 'mimic_sequences', np.array(seqs))
    path_mapping = input_path + 'ICD9_to_int_dict'
    pickle.dump(types, open(path_mapping, 'wb'), -1)

    only_visits_newSeqs = [patient[1:] for patient in newSeqs]
    n_uniques_ICD9 = len(set([a for b in [a for b in only_visits_newSeqs for a in b] for a in b]))
    print(f'number of unique ICD9 codes in mimic: {n_uniques_ICD9}')
    visit_file = newSeqs

    if CSS:
        # the CCS grouper can be downloaded here: https://www.nber.org/data/icd-ccs-crosswalk.html
        # css_grouper_csv_file_path = os.path.join(os.path.dirname(__file__), 'dxicd2ccsxw.csv')
        css_grouper_csv_file_path = input_path + 'dxicd2ccsxw.csv'
        CSS_grouper_csv = pd.read_csv(css_grouper_csv_file_path)
        new_dict_types = defaultdict(int)

        for item in types.items():
            clean_ICD9 = item[0].replace('.', '')
            if len(clean_ICD9) == 0:
                clean_ICD9 = 'unknown'
            new_dict_types[clean_ICD9] = item[1]

        types_df = pd.Series(new_dict_types).reset_index().rename(columns={'index': 'icd', 0: 'internal_code'})
        mapping = pd.merge(types_df, CSS_grouper_csv)[['ccs', 'internal_code']]
        mapping_dict = dict(zip(mapping.internal_code, mapping.ccs))
        newSeqs_ccs = [[list(np.vectorize(mapping_dict.get)(np.array(visit))) for visit in patient] for patient in
                       only_visits_newSeqs]
        #print(f'newSeqs_ccs = {newSeqs_ccs}')

        print(f'number of unique CSS-grouper codes in mimic: {len(set([a for b in [a for b in newSeqs_ccs for a in b] for a in b]))}')

        print('Converting strSeqs to intSeqs, and making CCStypes')
        CCStypes = {}
        newCSSSeqs = []
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
            newCSSSeqs.append(newPatient)

        path_CCS = input_path + 'CCS_to_int_dict'
        pickle.dump(CCStypes, open(path_CCS, 'wb'), -1)

        label_file = newCSSSeqs
    else:
        label_file = [patient[1:] for patient in newSeqs]

    n_uniques_output_codes = len(set([a for b in [a for b in label_file for a in b] for a in b]))
    print(f'number of unique output codes in mimic: {n_uniques_output_codes}')

    pickle.dump(visit_file, open(output_path + 'visit_complete', 'wb'), protocol=-1)
    pickle.dump(label_file, open(output_path + 'label_complete', 'wb'), protocol=-1)

    visit_train, visit_Test, label_train, label_Test = train_test_split(visit_file, label_file, test_size=test_set_size,
                                                                        random_state=42)
    #visit_test, visit_valid, label_test, label_valid = train_test_split(visit_Test, label_Test, test_size=0.33,
                                                                        #random_state=42)
    visit_test = visit_Test
    visit_valid = visit_Test
    label_test = label_Test
    label_valid = label_Test

    only_visits_train = [patient[1:] for patient in visit_train]
    patients_ids_train = [patient[0] for patient in visit_train]
    only_visits_validation = [patient[1:] for patient in visit_valid]
    patients_ids_validation = [patient[0] for patient in visit_valid]
    only_visits_test = [patient[1:] for patient in visit_test]
    patients_ids_test = [patient[0] for patient in visit_test]

    pickle.dump(only_visits_train, open(output_path + 'visit.train', 'wb'), protocol=-1)
    pickle.dump(patients_ids_train, open(output_path + 'pids.train', 'wb'), protocol=-1)
    pickle.dump(only_visits_validation, open(output_path + 'visit.valid', 'wb'), protocol=-1)
    pickle.dump(patients_ids_validation, open(output_path + 'pids.valid', 'wb'), protocol=-1)
    pickle.dump(only_visits_test, open(output_path + 'visit.test', 'wb'), protocol=-1)
    pickle.dump(patients_ids_test, open(output_path + 'pids.test', 'wb'), protocol=-1)

    pickle.dump(label_train, open(output_path + 'label.train', 'wb'), protocol=-1)
    pickle.dump(label_valid, open(output_path + 'label.valid', 'wb'), protocol=-1)
    pickle.dump(label_test, open(output_path + 'label.test', 'wb'), protocol=-1)

    print()
    print(f'You can use the files visit and label (.train/.valid/.test) in {output_path} to train doctorAI:')
    print(f'seqFile="{output_path}visit"')
    print(f'labelFile="{output_path}label"')
    print(f'outFile="<output_path>/trained_drAI_model"')

    if CSS:
        print(f'dr = doctorai.DoctorAI(ICD9_to_int_dict="{input_path}/CCS_to_int_dict",verbose=True)')
        print(
            f"dr.train(seqFile=seqFile, inputDimSize={n_uniques_ICD9}, labelFile=labelFile, numClass={n_uniques_output_codes}, outFile=outFile, max_epochs=50)")
    else:
        print(f'dr = doctorai.DoctorAI(ICD9_to_int_dict="{input_path}/ICD9_to_int_dict",verbose=True)')
        print(
            f"dr.train(seqFile=seqFile, inputDimSize={n_uniques_ICD9}, labelFile=labelFile, numClass={n_uniques_output_codes}, outFile=outFile, max_epochs=50)")

    return