import networkx as nx
import pickle
from .utils import subset_ontology_from_data, wup

def pickle_save(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def pickle_load(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def compute_c2c(icd9_tree, dataset_sequences, output_path, verbose=True):
    """
    This function computes and save a python dictionary containing all pairwise distances between the concepts of the
    ontology present in the dataset used to train and test the blackbox

    :param icd9_tree: nx.DiGraph, is the ontology
    :param dataset_sequences: list of lists of lists, is the dataset used to train and test the blackbox
    :param output_path: str, is the path where you want to save the output
    :return: python dictionary, with key the tuple of concepts and value the pairwise distances between the concepts
    """

    data_concepts = sorted(list(set([a for b in [c for d in dataset_sequences for c in d] for a in b])))
    subset_ontology = subset_ontology_from_data(icd9_tree, dataset_sequences)
    paths_from_ROOT = [nx.shortest_path(subset_ontology, 'ROOT', c) for c in data_concepts]
    n_paths = len(paths_from_ROOT)

    c2c_dict = {}

    for i, v1 in enumerate(paths_from_ROOT):
        if verbose:
            if i % 100 == 0:
                print(f'{i} concepts elaborated out of {n_paths}')
        for v2 in paths_from_ROOT:
            c2c_dict[(v1[-1], v2[-1])] = wup(v1, v2)
    if verbose:
        print(f'{i} concepts elaborated')
    pickle_save(c2c_dict, f'{output_path}/c2c_dist')

    return c2c_dict


def compute_p2p(icd9_tree, dataset_sequences, output_path):

    return



