from sklearn.metrics import f1_score
from skmultilearn.model_selection import iterative_train_test_split
from scipy.spatial import distance
from .utils import *
from .pre_processing import *
import itertools
from sklearn.tree import DecisionTreeClassifier


class DoctorXAI():

    def __init__(self,
                 patient_sequence,
                 dataset_sequences,
                 black_box_oracle,
                 ontology_path_file='../ICD9_ontology.csv',
                 n_first_neighbors=10,
                 syn_neigh_size=1000,
                 random_seed=42,
                 c2c_distance_matrix=None,
                 p2p_distance_matrix=None,
                 pre_compute_matrices=False,
                 output_path='../'
                 ):
        """
        :param patient_sequence: list of lists, it is a sequence of visits of ICD9 representing the patient clinical history
        :param dataset_sequences: list of lists of lists, it is a list containing all patients sequences (list of visits of ICD9)
        :param black_box_oracle: trained ML model, must have a .predict() methods that take as input one or more patient sequence/s
        :param ontology_path_file: str, path to the ontology file. It must be a CSV containing an edge list
        :param n_first_neighbors: int, default 10, the number of first neighbors to be used to perturb the instance
        :param random_seed: int, seed for random perturbation (for reproducibility concerns)
        :param c2c_distance_matrix: precomputed distance matrix between all ICD9 codes
        :param p2p_distance_matrix: precomputed distance matrix betwwen all patients in the dataset
        :param pre_compute_matrices: if you want to pre-compute all the distances among codes and patients in advance
        """

        self.patient_sequence = patient_sequence
        self.dataset_sequences = dataset_sequences
        self.black_box_oracle = black_box_oracle
        self.ontology_path_file = ontology_path_file
        self.n_first_neighbors = n_first_neighbors
        self.syn_neigh_size = syn_neigh_size
        self.icd9_tree = read_CSV_ontology(self.ontology_path_file)
        self.random_seed = random_seed
        self.output_path = output_path
        self.decision_rule = ''
        self.code_names = []
        self.hit = -1
        self.fidelity = -1
        self.istance_string = ''
        self.list_split_conditions = ''
        self.ICD9_description_dict = None


        if pre_compute_matrices:
            print('WARNING The precomputation of distance matrices might take very long')
            self.c2c_distance_matrix = compute_c2c()
            self.p2p_distance_matrix = compute_p2p()
        else:
            self.c2c_distance_matrix = c2c_distance_matrix
            self.p2p_distance_matrix = p2p_distance_matrix


    def find_closest_neighbors(self):
        """
        This function finds the first k (n_first_neighbors) closest neighbors of the analyzed patient (patient_sequence)
        in the dataset (dataset_sequences). It does so by using the ontology.
        :return: list of lists of lists, closest neighbours patients' sequences including the patient under analysis
        """

        if self.n_first_neighbors<0:
            print('n_first_neighbors should be a positive integer or 0')
        if self.n_first_neighbors==0:
            print('Perturbing only the patients to be explained')
            real_neighs = [self.patient_sequence]
        else:
            if self.p2p_distance_matrix:
                #take the firs k element of the p2p_distance_matrix
                print('BE CAREFUL: using the precomputed patient2patient distance matrix')
                real_neighs = []#self.p2p_distance_matrix[self.n_first_neighbors]

            else:
                #calculate the k closest neighbors in the dataset
                if self.c2c_distance_matrix:
                    c2c_dict = self.c2c_distance_matrix
                else:
                    c2c_dict = compute_c2c(self.icd9_tree,self.dataset_sequences,self.output_path,verbose=False)
                patients_dist = {}

                for i, patient in enumerate(self.dataset_sequences):
                    wup_patient_dist = wup_patient(c2c_dict, self.patient_sequence, patient)
                    patients_dist[i] = wup_patient_dist

                closest_neigh_indexes = list(
                    {k: v for k, v in sorted(patients_dist.items(), key=lambda item: item[1])}.keys())[
                                        :self.n_first_neighbors+1]
                real_neighs = [self.dataset_sequences[idx] for idx in closest_neigh_indexes]

        return real_neighs


    def create_synthetic_neighborhood(self):
        """
        This function creates the synthetic neighborhood that will become the training set for the interpretable classifier
        It does so by masking similar conditions in the patient's real neighborhood using the ontology.
        :return: syn_neighs, syn_neigh_labels
        """

        real_neighs = self.find_closest_neighbors()

        if len(real_neighs)==1:
            syn_neighs = ontological_perturbation_of_one_patient(real_neighs[0],seed=self.random_seed)
        else:
            synth_neigh_onto_seq = ontological_perturbation(real_neighbors=real_neighs,
                                                            size=self.syn_neigh_size,
                                                            mask_prob=0.3,
                                                            G=self.icd9_tree,
                                                            seed=self.random_seed)
            syn_neighs = [syn_patient for syn_patient in synth_neigh_onto_seq if len(syn_patient) >= 2]

        # I also add to the syn_neighs the real_neighs because these patients will become the training set
        # for the interpretable classifier
        for patient in real_neighs:
            syn_neighs.append(patient)
        #I remove patients duplicates if there is any
        syn_neighs.sort()
        syn_neighs = list(syn_neigh for syn_neigh, _ in itertools.groupby(syn_neighs))

        # assining a blackbox label to the syn_neighs
        syn_neigh_labels = self.black_box_oracle.predict(syn_neighs)

        return syn_neighs, syn_neigh_labels


    def extract_rule(self, ICD9_description_dict=None, CCS_description_dict=None):

        self.ICD9_description_dict = ICD9_description_dict
        self.CCS_description_dict = CCS_description_dict

        print(f'Creating the synthetic neighborhood...')
        #create synthetic neighbors
        syn_neighs, syn_neigh_labels = self.create_synthetic_neighborhood()

        print('Encoding sequences...')
        #encode the sequence into a flat representation
        flat_syn_neighs, features_names = flat_cloud(syn_neighs, last_visit=True)
        flat_patient, features_names = flat_cloud([self.patient_sequence], all_codes=features_names, last_visit=True)

        #one-hot-encoding of the labels
        one_hot_syn_neigh_labels, labels_names = dummify_labels(syn_neigh_labels)
        dummy_patient_label = dummify_label_given_columns(label=self.black_box_oracle.predict(self.patient_sequence),
                                                          feat_names=labels_names)

        #print(f'len(flat_syn_neighs) = {len(flat_syn_neighs)}')
        #print(f'len(one_hot_syn_neigh_labels) = {len(one_hot_syn_neigh_labels)}')

        print('Training the interpretable classifier...')
        #split in training and test for the DT ITERATIVELY using the skmultilearn library
        x_train, y_train, x_test, y_test = iterative_train_test_split(flat_syn_neighs, one_hot_syn_neigh_labels,
                                                                      test_size=0.2)
        #make sure that the training set of the DT contains the instance we want to explain:
        x_train = np.vstack([x_train,flat_patient.reshape(1,-1)])
        y_train = np.vstack([y_train,dummy_patient_label.reshape(1,-1)])

        #train the DT
        DT_synth = DecisionTreeClassifier()
        DT_synth.fit(x_train, y_train)

        print('Evaluating explainability metrics...')
        # fidelity of the DT to the black box on the synthetic neighborhood
        fidelity_synth = f1_score(y_true=y_test, y_pred=DT_synth.predict(x_test), average='micro')
        #hit of the DT
        hit_synth = 1 - distance.hamming(dummy_patient_label, DT_synth.predict(flat_patient.reshape(1, -1)))


        print('Extracting the rule...')
        #rule_extraction
        istance_string, list_split_conditions, code_names, decision_rule = rule_extractor(DT_synth,
                                                                                          self.patient_sequence,
                                                                                          flat_patient,
                                                                                          features_names,
                                                                                          labels_names,
                                                                                          ICD9_description_dict=self.ICD9_description_dict)
        print(decision_rule)
        print(f'DT hit on patient: {hit_synth}')
        print(f'DT fidelity to black-box: {fidelity_synth}')

        self.decision_rule = decision_rule
        self.code_names = code_names
        self.hit = hit_synth
        self.fidelity = fidelity_synth
        self.istance_string = istance_string
        self.list_split_conditions = list_split_conditions

        return decision_rule, istance_string, list_split_conditions, code_names, fidelity_synth, hit_synth


    def visualize_explanation(self ,save=False, figure_path = './explanation.png'):

        if len(self.decision_rule)==0:
            self.decision_rule, \
            self.istance_string, \
            self.list_split_conditions, \
            self.code_names, \
            self.fidelity,\
            self.hit = self.extract_rule()

        bb_prediction = self.black_box_oracle.predict(self.patient_sequence)
        bb_prediction = [str(x) for x in bb_prediction]
        # tree_prediction = self.decision_rule.split('->')[1].replace(' ', '').split(',')

        extract_explanation_from_rule(patient2bexplained=self.patient_sequence,
                                      code_names=self.code_names,
                                      prediction = bb_prediction,
                                      save=save,
                                      path=figure_path)

        print('DECISION RULE PREMISE:')
        for elem in self.istance_string:
            print(elem)
        print()
        print('PREDICTION:')
        if self.CCS_description_dict:
            for code in bb_prediction:
                print(f'{code} = "{self.CCS_description_dict[int(code)]}"')
        else:
            for code in bb_prediction:
                print(f'{code} = "{self.ICD9_description_dict[str(code)]}"')
        print()
        print('METRICS:')
        print(f'hit: {self.hit}')
        print(f'fidelity: {self.fidelity}')

        return


