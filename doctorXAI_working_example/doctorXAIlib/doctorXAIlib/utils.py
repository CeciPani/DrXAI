import numpy as np
import pandas as pd
import csv
from similarity.weighted_levenshtein import WeightedLevenshtein
from similarity.weighted_levenshtein import CharacterSubstitutionInterface
import networkx as nx
from sklearn.metrics import hamming_loss, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from collections import defaultdict
import plotly.graph_objects as go
import matplotlib.cm as cm
from matplotlib.colors import to_hex

def read_CSV_ontology(ontology_path_file):
    """
    :param ontology_path_file: the path to the ontology file with the name of the CSV file
    e.g. "/path/to/file/ICD9_ontology.csv"
    :return: a networkx Directed Graph
    """
    icd9_tree = nx.DiGraph()
    nodes = set()
    line_count = 0
    with open(ontology_path_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            line_count += 1
            icd9_tree.add_edge(row[1], row[0])
            nodes.add(row[1])
            nodes.add(row[0])
        print(f'The ontology has {line_count} "is-a" links.')
        print(f'The ontology has {len(nodes)} nodes')

    return icd9_tree


def subset_ontology_from_data(icd9_tree, dataset_sequences):
    """
    This function selects the subset of concepts from the original ontology icd9_tree that also appears in the data
    :param icd9_tree: nx.DiGraph, original ontology
    :param dataset_sequences: list of lists of lists, the sequences of the dataset (list of patients's visits)
    :return: nx.DiGraph
    """

    data_concepts = sorted(list(set([a for b in [c for d in dataset_sequences for c in d] for a in b])))
    data_nodes = []

    for l in data_concepts:
        data_nodes += nx.shortest_path(icd9_tree, 'ROOT', l)
    data_nodes = sorted(list(set(data_nodes)))

    # PARENT->CHILD
    subset_ontology = nx.DiGraph()
    for child in data_nodes:
        if child == 'ROOT':
            continue
        parent = list(icd9_tree.in_edges(child))[0][0]
        subset_ontology.add_edge(parent, child)

    return subset_ontology


def wup(a1, a2):
    """
    This function calculates the WUP distance between concept C1 and concept C2 in the ontology:

    WUP(C1,C2) = (2*depth_LCA)/(N_1+N_2+depth_LCA)

    where depth_LCA is the depth of the last common ancestor (LCA) of concept1 and concept2
    N_1 is the number of nodes from LCA to concept C1
    N_2 is the number of nodes from LCA to concept C2

    :param a1: path from ROOT to the concept C1
    :param a2: path from ROOT to the concept C2
    :return:
    """
    # common substring length -> depth of last common ancestor (LCA)
    # a1, a2 are PATHS FROM ROOT

    depth_LCA = 0
    while True:
        if depth_LCA >= len(a1) or depth_LCA >= len(a2) or a1[depth_LCA] != a2[depth_LCA]:
            break
        depth_LCA += 1

    dr = depth_LCA - 1
    da = len(a1) - depth_LCA
    db = len(a2) - depth_LCA

    return ((2 * dr) / (da + db + 2 * dr))


def get_closest_datapoints(start_index, dist_matrix, k):
    candidates = sorted(enumerate(dist_matrix[start_index]), key=lambda x: x[1])[:k + 1]
    return list(candidates)

def wup_visit(c2c, v1, v2):
    all_icd9 = sorted(list(set(v1 + v2)))
    symbols = [chr(i) for i in range(len(all_icd9))]  # fix this
    coded_all = [symbols[i] for i in range(len(all_icd9))]
    coded_v1 = ''.join([coded_all[all_icd9.index(c)] for c in sorted(v1)])
    coded_v2 = ''.join([coded_all[all_icd9.index(c)] for c in sorted(v2)])
    encoder = {k: v for (k, v) in zip(all_icd9, coded_all)}
    decoder = {v: k for (k, v) in zip(all_icd9, coded_all)}

    encoded_c2c = {}
    for icd9_1 in v1:
        for icd9_2 in v2:
            encoded_c2c[(encoder[icd9_1], encoder[icd9_2])] = 1 - c2c[(icd9_1, icd9_2)]

    class CharacterSubstitution(CharacterSubstitutionInterface):
        def cost(self, c0, c1):
            return encoded_c2c[(c0, c1)]

    weighted_levenshtein = WeightedLevenshtein(CharacterSubstitution())
    return weighted_levenshtein.distance(coded_v1, coded_v2)


def wup_patient(c2c, s, t):
    n = len(s)
    m = len(t)
    dtw = np.full((n, m), 10000, dtype=np.float64)
    dtw[0, 0] = 0
    for i in range(n):
        for j in range(m):
            cost = np.round(wup_visit(c2c, s[i], t[j]), 3)
            in_cost = dtw[i - 1, j] if i > 0 else 0
            del_cost = dtw[i, j - 1] if j > 0 else 0
            edit_cost = dtw[i - 1, j - 1] if i > 0 and j > 0 else 0
            dtw[i, j] = cost + min(in_cost, del_cost, edit_cost)
    return dtw[-1][-1]


def distance_matrix(syn_seqs, c2c):
    dist_syn_matrix = np.full((len(syn_seqs), len(syn_seqs)), 0.)
    for i in range(len(syn_seqs)):
        for j in range(len(syn_seqs)):
            dist = wup_patient(c2c, syn_seqs[i], syn_seqs[j])
            dist_syn_matrix[i][j] = dist
            dist_syn_matrix[j][i] = dist
    return dist_syn_matrix


def flat_dp(patient, last_visit=True):
    fdp = {}
    if last_visit:
        codes = list(set([a for b in patient for a in b]))
        for c in codes:
            seq = [1 if c in el else 0 for el in patient[::-1]]  # already rev
            decay = [1 / 2 ** i for i in range(1, len(patient) + 1)]
            # print(c,seq,decay,np.multiply(seq,decay),sum(np.multiply(seq,decay)))
            fdp[c] = sum(np.multiply(seq, decay))
    else:
        codes = list(set([a for b in patient[:-1] for a in b]))
        for c in codes:
            seq = [1 if c in el else 0 for el in patient[:-1][::-1]]  # already rev
            decay = [1 / 2 ** i for i in range(1, len(patient))]
            # print(c,seq,decay,np.multiply(seq,decay),sum(np.multiply(seq,decay)))
            fdp[c] = sum(np.multiply(seq, decay))
    return fdp


def flat_cloud(kdp, last_visit=True, all_codes=None):
    local_flat = [flat_dp(n, last_visit=last_visit) for n in kdp]
    if not all_codes:
        all_codes = sorted(list(set([a for b in [list(lf.keys()) for lf in local_flat] for a in b])))
    feat_mx = np.full((len(kdp), len(all_codes)), 0.)
    for row, lf in enumerate(local_flat):
        for (k, v) in lf.items():
            feat_mx[row][all_codes.index(k)] = v
    return feat_mx, all_codes


def unflat_cloud(cloud, code_list, max_past, past_mean, past_std):
    res = []
    for dp in cloud:  # iterate over synthetic datapoints
        dp_mx = np.full((len(code_list), max_past), 0)  # codes * visits
        for col, temp_val in enumerate(dp):
            check = .5
            for visit_index in range(max_past - 1, -1, -1):
                if temp_val >= check:
                    dp_mx[col, visit_index] = 1
                    temp_val -= check
                check /= 2
            # dp_mx matrix filled with values
        raw_seq = [[code_list[c] for c in range(dp_mx.shape[0]) if dp_mx[c, v] == 1] for v in range(dp_mx.shape[1])]  #
        num_visit = max(1, int(np.ceil(np.random.normal(past_mean, past_std))))
        res.append(raw_seq[-num_visit:])
    return res


def normal_perturbation_on_flat(flat_neigh, seed, size=1000):
    np.random.seed(seed)

    num_codes = flat_neigh.shape[1]
    flat_syn_neigh = np.full((size, num_codes), 0.)

    for code_col in range(num_codes):
        code_mean = np.mean(flat_neigh[:, code_col])
        code_std = np.std(flat_neigh[:, code_col])
        flat_syn_neigh[:, code_col] = np.random.normal(code_mean, code_std, size)

    flat_syn_neigh[flat_syn_neigh < 0.] = 0.
    flat_syn_neigh[flat_syn_neigh > 1.] = 1.

    return flat_syn_neigh


# mask together subtrees of instances
# the root of the subtree is the ancesto ICD9 with no dot, or one of the exceptions
def ontological_perturbation_of_one_patient(patient, mask_prob, num_clones, G, seed):
    ### DA CAMBIARE PERCHÈ TROPPO SPECIFICO PER L'ONTOLOGIA ICD9
    np.random.seed(seed)
    codes = list(set([a for b in patient for a in b]))
    exceptions = ['E849.0', 'E849.3', 'E849.4', 'E849.5', 'E849.6', 'E849.7', 'E849.8', 'E849.9']
    # subtree roots
    fringe = list(set([c if c in exceptions else c.split('.')[0] for c in codes]))
    # partition
    classes = [[c for c in codes if nx.has_path(G, f, c)] for f in fringe]
    # make clones
    clones = []
    # iterate for each clone to create
    for k in range(num_clones):
        # ICD9 codes that I’m not masking
        allowed_codes = [a for b in [c for i, c in enumerate(classes) if np.random.rand() > mask_prob] for a in b]
        # duplicate patient keeping only allowed ICD9s
        new_clone = [[c for c in v if c in allowed_codes] for v in patient]
        clones.append(new_clone)
    return clones


def ontological_perturbation(real_neighbors, size, mask_prob, G, seed):
    num_clones = int(size / len(real_neighbors))
    synt_neigh = [ontological_perturbation_of_one_patient(neigh, mask_prob, num_clones, G, seed) for neigh in
                  real_neighbors]
    return [a for b in synt_neigh for a in b]


def old_dummify_labels(labels, dict_icd2int, n_unique_codes=4880):
    dummyfied_labels = np.zeros((len(labels), n_unique_codes))
    int_labels = [[dict_icd2int[k] for k in label] for label in labels]
    for i in range(len(int_labels)):
        dummyfied_labels[i][int_labels[i]] = 1
    return dummyfied_labels

def dummify_labels(labels):

    all_codes = list(set([a for b in labels for a in b]))
    dict_codes = defaultdict(list)
    for last_visit in labels:
        for code in all_codes:
            if code in last_visit:
                dict_codes[code].append(1)
            else:
                dict_codes[code].append(0)
    one_hot_labels = pd.DataFrame(dict_codes)

    return one_hot_labels.values, list(one_hot_labels.columns.values)

def dummify_label_given_columns(label, feat_names):
    dummy_label = {}
    for code in feat_names:
        if code in label:
            dummy_label[code] = 1
        else:
            dummy_label[code] = 0
    return np.array(list(dummy_label.values()))


def old_dummify_patient(patient_seq, dict_icd2int, n_unique_codes=4880):
    patient_codes = list(set([a for b in patient_seq for a in b]))
    dummyfied_patient = np.zeros(n_unique_codes)
    int_codes = [dict_icd2int[k] for k in patient_codes]
    for i in range(len(int_codes)):
        dummyfied_patient[int_codes[i]] = 1
    return dummyfied_patient


def old_dummify_patients(patient_seqs, dict_icd2int, n_unique_codes=4880):
    dummy_patients = np.array([old_dummify_patient(patient, dict_icd2int, n_unique_codes) for patient in patient_seqs])
    return dummy_patients


def rule_extractor(DT, patient2bexplained, flat_patient, flat_patient_features_names, labels_names,
                   ICD9_description_dict=None):
    """
    :param DT: sklearn Decision Tree, the DT trained on the synthetic neighborhood
    :param patient2bexplained: list of lists, the sequence of visits of the patient to be explained
    :param flat_patient: list, the numerical/flattened version of the patient to be explained
    :param flat_patient_features_names: list, the names of the features of the flattened version of the patient to be explained
    :param labels_names: list, the names of the labels
    :param ICD9_description_dict: python dict, the dictionary that maps the names of the codes to their description
    :return:
    """
    labels_names = np.array(labels_names)
    decision_path_nodes = DT.tree_.decision_path(flat_patient.astype(np.float32))
    decision_path_nodes_indices = decision_path_nodes.indices
    feature = DT.tree_.feature
    threshold = DT.tree_.threshold

    istance_string = list()
    list_split_conditions = list()
    code_names = list()

    for node_id in decision_path_nodes_indices:

        code_name = flat_patient_features_names[feature[node_id]]
        true_code_value = flat_patient[0][feature[node_id]]
        code_names.append(code_name)

        # print(f'{code_name} = {true_code_value}')
        #node_string = f'{code_name} = {true_code_value} ->'
        if ICD9_description_dict:
            node_string = f'{code_name} = "{ICD9_description_dict[code_name]}", '
        else:
            node_string = f'{code_name}, '

        if true_code_value > 0:

            list_visits = [i + 1 for i, visit in enumerate(patient2bexplained) if code_name in visit]

            if len(list_visits) == 1:
                #print(f'{code_name} was observed in visit {list_visits[0]}')
                node_string = node_string + f' was observed in visit {list_visits[0]}'

            elif len(list_visits) == 2:
                #print(f'{code_name} was observed in visit {list_visits[0]} and {list_visits[1]}')
                node_string = node_string + f' was observed in visit {list_visits[0]} and {list_visits[1]}'

            elif len(list_visits) > 2:
                first_visits = ', '.join([str(i) for i in list_visits[:-1]])
                first_visits += f' and {list_visits[-1]}'
                #print(f'{code_name} was observed in visits {first_visits}')
                node_string = node_string + f' was observed in visits {first_visits}'

        else:

            #print(f'{code_name} was not observed')
            node_string = node_string + f'was not observed'

        istance_string.append(node_string)

        if flat_patient[0][feature[node_id]] <= threshold[node_id]:
            threshold_sign = " <= "  # "dopo la visita"
        else:
            threshold_sign = " > "  # "prima della visita "

        list_split_conditions.append(
            f'{flat_patient_features_names[feature[node_id]]}{threshold_sign}{threshold[node_id]}')
        #print(f'{code_name}{threshold_sign}{round(threshold[node_id], 3)}')
        decision_rule = ', '.join(list_split_conditions)

    conclusion = labels_names[DT.predict(flat_patient)[0] != 0]
    conclusion = [str(x) for x in conclusion]
    predicted_labels = ', '.join(conclusion)
    decision_rule += f' -> {predicted_labels}'

    return istance_string, list_split_conditions, code_names, decision_rule



def hamming_score(y_true, ypred):
    hamming = 1 - hamming_loss(y_true, ypred)
    return hamming


def tuned_tree(X, y, param_distributions, scoring='f1_micro', cv=5):
    """
    This function performs an hyperpatameter tuning using a randomized search (see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
    and returns a tuned decision tree

    :param X: array-like or sparse matrix, shape = [n_samples, n_features], the training input sample, synthetic neighborhood features
    :param y: array-like, shape = [n_samples] or [n_samples, n_outputs], the target values (class labels) as integers or strings, the labels of the black box
    :param param_distributions:  dict, dictionary with parameters names (string) as keys and distributions or lists of parameters to try.
    :param scoring: string, callable, list/tuple, dict or None, default 'f1_micro'
    :param cv:  int (number of folds), cross-validation generator or an iterable, optional, it determines the cross-validation splitting strategy, default=5
    :return: sklearn DecisionTreeClassifier
    """

    hamming_scorer = make_scorer(hamming_score)

    tree = DecisionTreeClassifier(class_weight="balanced")
    sop = np.prod([len(v) for k, v in param_distributions.items()])
    n_iter_search = min(30, sop)
    random_search = RandomizedSearchCV(tree,
                                       param_distributions=param_distributions,
                                       scoring=hamming_scorer,
                                       n_iter=n_iter_search,
                                       cv=cv)  # scoring='f1_micro')
    random_search.fit(X, y)
    best_params = random_search.best_params_
    tree.set_params(**best_params)
    tree.fit(X, y)

    return tree


def highlight_codes(visit, color_map, code_names):
    label = ''
    for code in visit:
        if code in code_names:
            label += f'<b style="color: {color_map[code]};">{code}</b><br>'
        else:
            label += f'{code}<br>'
    return label


def extract_explanation_from_rule(patient2bexplained, code_names, prediction, save=False, path='./explanation.png'):
    prediction_annotation = '<br>'.join(prediction)
    # print(f'prediction bb: {prediction_annotation}')

    colors = cm.rainbow(np.linspace(0, 1, len(code_names)))
    colors = [to_hex(color) for color in colors]
    color_map = {code: color for code, color in zip(code_names, colors)}

    min_x = 0
    width = 4
    max_x = width * len(patient2bexplained)

    x_ticks = np.linspace(min_x, max_x, len(patient2bexplained))
    y_ticks = [95 for i in range(len(patient2bexplained))]

    fig = go.Figure()

    fig.layout = go.Layout(
        title="Patient's visits",
        yaxis=dict(showgrid=False, ticks=None, showticklabels=False),
        xaxis=dict(showgrid=False, ticks=None, showticklabels=False),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        autosize=False,
        width=1000,
        height=700,
    )

    for i, x in enumerate(x_ticks):
        fig.add_annotation(
            x=x_ticks[i],
            y=y_ticks[i],
            text=f"visit {i + 1}",
            ax=0,
            ay=-40
        )

    texts_annotations = []
    for i, x in enumerate(x_ticks):
        codes = highlight_codes(patient2bexplained[i], color_map, code_names)
        texts_annotations.append(codes)

    fig.add_trace(go.Scatter(
        x=x_ticks,
        y=y_ticks,
        mode="lines+text",
        name="Patient's visit",
        text=texts_annotations,
        textposition="bottom center",
        hoverinfo='skip'

    ))

    fig.add_annotation(
        x=x_ticks[-1] + width / 2,
        y=100,
        text=f'Prediction <br> for visit {len(patient2bexplained) + 1}:',
        ax=0,
        ay=0,
        font=dict(
            size=16,
            color="red"
        ),

    )

    fig.add_trace(go.Scatter(
        x=[x_ticks[-1] + width],
        y=[95],
        mode="lines+markers+text",
        text=[prediction_annotation],
        textposition="bottom center",
        # hoverinfo='skip'

    ))

    # fig.add_annotation(
    #    x=x_ticks[-1] + width,
    #    y=90,
    #    text=prediction_annotation,
    #    ax=0,
    #    ay=40,
    #    bordercolor="#ffbb99",
    #    borderwidth=2,
    #    borderpad=4,
    #    bgcolor="#ffbb99",
    #    opacity=1

    # )

    fig.update_xaxes(range=[0, [x_ticks[-1] + width]])
    fig.update_yaxes(range=[0, 100])
    if save:
        fig.write_image(path)

    fig.show()
    return

