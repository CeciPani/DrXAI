DoctorXAI
==============================

DoctorXAI is an ontology-based approach to black box sequential data classification explanations. Its main characteristics are:

* **It is an agnostic explainer** It does not use any internal parameter of the model to generate the explanations. 
* **It deals with sequential data inputs** It uses a temporal encoder/decoder scheme that preserves feature interpretability and its sequential information.
* **It exploits medical ontologies** It uses the medical knowledge encoded in the ontology during the explanation process.
* **It deals with multilabel outputs** It allows to explain the outcome of multilabel classifiers.

Having such characteristics makes DoctorXAI well suited to explain black boxes that deal with healthcare data. Indeed healthcare data are often sequential (e.g. patient's clinical history), multi labeled (in case of comorbidities) and ontology-linked (the medical knowledge is encoded in ontologies).

This project is based on our paper *Doctor XAI: an ontology-based approach to black-box sequential data classification explanations* published on the *FAT* '20: Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency*:

```
@inproceedings{panigutti2020doctor,
  title={Doctor XAI: an ontology-based approach to black-box sequential data classification explanations},
  author={Panigutti, Cecilia and Perotti, Alan and Pedreschi, Dino},
  booktitle={Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency},
  pages={629--639},
  year={2020}
}
```

* The experiments of the paper are carried out in a jupyter notebook in the notebook folder, to run our experiments from start to end you need [MIMIC-III dataset](https://mimic.physionet.org/). However, our results are saved as pickled python dictionary in the output folder. 
* You can find a notebook with a working example of doctorXAI in the **doctorXAI_working_example** folder, the example is illustrated below

# Doctor XAI usage example on MIMIC-III dataset

To run this example you need to have access to the [MIMIC-III dataset](https://mimic.physionet.org/). MIMIC-III is a large, freely-available database comprising deidentified health-related data associated with over 40,000 patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012.

You can find this example in the jupyter notebook contained in the **doctorXAI_working_example** folder.

## Installing DoctorXAI and DoctorAI

0. **Install the requirements.txt listed packages**

1. **Install *DoctorXAI***: In order to run DoctorXAI you first have to locally install the pyhton module *doctorXAIlib*. You can do this by running the following command into the module directory:
 ~~~~
 $ cd doctorXAIlib
 $ pip install .
 ~~~~

To run the working example you also need to install doctorAI, the python3 implementation of the 2016 paper of Choi et. al *Doctor ai: Predicting clinical events via recurrent neural networks*

 ~~~~
 $ cd doctorailib
 $ pip install .
 ~~~~

### Importing the modules and other useful file 

```python
from doctorailib import doctorai
from doctorXAIlib import doctorXAI
import pickle
import numpy as np
import pandas as pd
```

If you want the explanation of doctorXAI to be more clear you need to import the pickle dictionaries containing the descriptions of the codes (ICD-9 and CCS codes)

```python
ICD9_description_dict = pickle.load(open('./ICD9_description_dict.pkl', 'rb'))
CCS_description_dict = pickle.load(open('./CCS_description_dict.pkl', 'rb'))
```

### MIMIC-III pre-processing

```python
# path containing the MIMIC-III tables ADMISSIONS.csv and DIAGNOSES_ICD.csv
mimic_path = '/home/user/datasets/mimic-iii/'
# the output path of doctorAI preprocessing
output_path = './preprocessing_doctorai/'
# the path of CCS grouper csv
CCS_grouper_csv_file_path = '../models/doctorailib/'
```

You need to set CCS = True in the prepare_mimic method to make doctorAI predict CCS codes instead of ICD9 codes (higher performance), the CCS codes are codes that group the ICD9 codes into a manageable number of clinically meaningful categories.

```python
from doctorailib import pre_processing
pre_processing.prepare_mimic(mimic_path=mimic_path, 
                             CCS_grouper_csv_file_path=CCS_grouper_csv_file_path, 
                             output_path=output_path, CCS=True)
```

### Training or loading a pre-trained doctor AI model

If you need to train doctorAI on MIMIC you can run the following code 

```python
import datetime
today = datetime.datetime.today()

seqFile="./preprocessing_doctorai/visit"
labelFile="./preprocessing_doctorai/label"
outFile=f"../models/trained_doctorAI_output/{today.year}_{today.month}_{today.day}_MIMIC_III_"

dr = doctorai.DoctorAI(ICD9_to_int_dict="./preprocessing_doctorai/ICD9_to_int_dict",
                       CCS_to_int_dict="./preprocessing_doctorai/CCS_to_int_dict",
                       verbose=True)

dr.train(seqFile=seqFile, 
         inputDimSize=4880, 
         labelFile=labelFile, 
         numClass=272, 
         outFile=outFile, 
         max_epochs=50)
```

Otherwise you can load your model and test its recall@k performance running the following code:

```python
model_file = '../models/trained_doctorAI_output/2020_9_23_MIMIC_III_.44.npz'
dr = doctorai.DoctorAI(modelFile=model_file,
                       ICD9_to_int_dict="./preprocessing_doctorai/ICD9_to_int_dict",
                       CCS_to_int_dict="./preprocessing_doctorai/CCS_to_int_dict")

#test set performance
visit_test = "./preprocessing_doctorai/visit.test"
label_test = "./preprocessing_doctorai/label.test"
dr.test_doctorAI(modelFile=model_file, 
                 hiddenDimSize=[200,200], 
                 visit_test=visit_test, 
                 label_test=label_test)

```
        build model ... 
        load data ... 
        done
        recall@10:0.3458427244425019, recall@20:0.5125027800359246, recall@30:0.6237559040635579


## Run DoctorXAI

DoctorXAI takes as input the following parameters:

* **patient_sequence**: `list of lists of ICD9`, the clinical history of the patient whose black-box decision you want to explain

* **dataset_sequences**: `lists of lists of ICD9`, the training set of the black-box or the set of patients where you want doctorXAI to look for neighbors of the patient you want to explain

* **black_box_oracle**: trained predictive model, it needs to have a "predict" method and it needs to take as input lists of lists of ICD9

* **ontology_path_file**: `str`, the path pf the csv file containing the onthology (ICD9) `default: '../ICD9_ontology.csv'`

* **n_first_neighbors**: `int`, the number of first neighbors to be used in the synthetic neighborhood creation, `default: 10`

* **syn_neigh_size**: `int`, the number of synthetic neighbors to create, `default: 1000`

* **random_seed**: `int`, the random seed used in the synthetic neighborhood creation, `default: 42`

* **c2c_distance_matrix**: the pre-computed code-to-code ontological distance matrix (wup distance), `default: None` (doctorXAI evaluates the matrix)

* **p2p_distance_matrix**: the pre-computed patient-to-patient ontological distance matrix, `default: None` (doctorXAI evaluates the matrix)

* **pre_compute_matrices**: `bool`, if the matrixes are already evaluated `default: False`

* **output_path**: `str`, output path for doctorXAI, `default: '../'`

```python
# taking all the sequences generated during MIMIC-III preprocessing
dataset_sequences = np.load('./preprocessing_doctorai/mimic_sequences.npy',allow_pickle=True)
# selecting one patient whose doctorAI decision we want to explain
patient_sequence = dataset_sequences[2]
# setting doctorAI as the black-box oracle
black_box_oracle = dr
# setting the ontology path 
ontology_path_file = '../models/doctorXAIlib/ICD9_ontology.csv'
```

This is the clinical history of a patient whose decision we want to explain:

        [['414.01', '411.1', '496', '401.9', '305.1', '530.81', '600.00', 'V10.51', '596.8'],
         ['998.31', '998.11', '415.11', '453.8', '996.72', '496', '414.01', 'V45.81', '401.9', '600.00', '530.81', 'V10.51'],   
         ['553.21', '415.11', '518.5', '486', '997.39', '518.0', '414.00', '496', '401.9', '600.00', '300.00', 'V10.51']]

As you can see it is a list of lists. Each list represents a visit of the patient to the ICU, i.e. a list of ICD-9 codes. This patient visited the ICU 3 times and, consulting the meaning of these codes [here](http://www.icd9data.com/2015/Volume1/default.htm) we can see that this patient has experiences several *Diseases Of The Respiratory System* and *Diseases Of The Circulatory System*.

If we run our black box model, doctorAI, on this patient, it predicts the following CCS codes for next visit:

        [101, 108, 127, 98, 106, 96, 53, 131, 105, 55, 238, 259, 157, 19, 122, 138, 257, 59, 663, 100, 49, 97, 117, 2616, 159, 130, 115, 103, 48, 2]

DoctorAI always precits 30 codes, in this case we can see that the first three ones are:

* 101: Coronary atherosclerosis and other heart disease
* 108: Congestive heart failure; nonhypertensive
* 127: Chronic obstructive pulmonary disease and bronchiectasis

You can see the meaning of other CCS codes [here](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/CCSUsersGuide.pdf)

To check the reasoning behind this classification we can use doctorXAI: 

```python
drXAI = doctorXAI.DoctorXAI(patient_sequence=patient_sequence,
                            dataset_sequences=dataset_sequences,
                            black_box_oracle=black_box_oracle,
                            ontology_path_file=ontology_path_file,
                            syn_neigh_size=500)
```

We can extract the explanation in the form of a decision rule and measure its **fidelity** and **hit** in the following way: 

```python
decision_rule,\
istance_string,\
list_split_conditions,\
code_names,\
fidelity_synth,\
hit_synth = drXAI.extract_rule(ICD9_description_dict=ICD9_description_dict, 
                               CCS_description_dict=CCS_description_dict)
```

        272.4 <= 0.25, V45.81 <= 0.375, 197.7 <= 0.125, 599.0 <= 0.125, V10.11 <= 0.125, 496 > 0.625, 414.00 > 0.125, 585.9 <= 0.375, 600.00 > 0.6875, V10.51 > 0.4375, 453.8 > 0.125, V45.82 > -2.0 -> 257, 2, 131, 259, 130, 138, 19, 663, 157, 159, 48, 49, 53, 55, 2616, 59, 96, 97, 98, 100, 101, 103, 105, 106, 108, 238, 115, 117, 122, 127

        hit on patient: 1.0
        fidelity to black-box: 0.9453074433656958

However, to make the decision rule more interpretable we can visualize the meaning of the codes and the meaning of the split condition by simply running:

```python
drXAI.visualize_explanation()
```


![png](https://github.com/CeciPani/DrXAI/blob/master/doctorXAI_explanation_visualization.png)



        DECISION RULE PREMISE:
        272.4 = "Other and unspecified hyperlipidemia", was not observed
        V45.81 = "Aortocoronary bypass status",  was observed in visit 2
        197.7 = "Malignant neoplasm of liver, secondary", was not observed
        599.0 = "Urinary tract infection, site not specified", was not observed
        V10.11 = "Personal history of malignant neoplasm of bronchus and lung", was not observed
        496 = "Chronic airway obstruction, not elsewhere classified",  was observed in visits 1, 2 and 3
        414.00 = "Coronary atherosclerosis of unspecified type of vessel, native or graft",  was observed in visit 3
        585.9 = "Chronic kidney disease, unspecified", was not observed
        600.00 = "Hypertrophy (benign) of prostate without urinary obstruction and other lower urinary tract symptom (LUTS)",  was observed in visits 1, 2 and 3
        V10.51 = "Personal history of malignant neoplasm of bladder",  was observed in visits 1, 2 and 3
        453.8 = "Acute venous embolism and thrombosis of other specified veins",  was observed in visit 2
        V45.82 = "Percutaneous transluminal coronary angioplasty status", was not observed

        PREDICTION:
        101 = "Coron athero"
        108 = "chf;nonhp"
        127 = "COPD"
        98 = "HTN"
        106 = "Dysrhythmia"
        96 = "Hrt valve dx"
        53 = "Hyperlipidem"
        131 = "Adlt resp fl"
        105 = "Conduction"
        55 = "Fluid/elc dx"
        238 = "Complic proc"
        259 = "Unclassified"
        157 = "Ac renl fail"
        19 = "Brnch/lng ca"
        122 = "Pneumonia"
        138 = "Esophgeal dx"
        257 = "Ot aftercare"
        59 = "Anemia"
        663 = "Screening and history of mental health an"
        100 = "Acute MI"
        49 = "DiabMel no c"
        97 = "Carditis"
        117 = "Ot circul dx"
        2616 = "E Codes: Adverse effects of medical care"
        159 = "UTI"
        130 = "Pleurisy"
        115 = "Aneurysm"
        103 = "Pulm hart dx"
        48 = "Thyroid dsor"
        2 = "Septicemia (except in labor)"

        METRICS:
        hit: 1.0
        fidelity: 0.9453074433656958













