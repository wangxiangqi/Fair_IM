import torch
import pickle
import networkx as nx
import numpy as np
# Define your file path
file_path = './pokec_n.pt'

# Load the model
model = torch.load(file_path)
"""
print(model)
print(model.keys())
print(model['num_nodes'])
print(model['num_edges'])
#这里有sensitive label了，那就直接用。
print(model['adjacency_matrix'])
print(len(model['node_features'][0]))
print(len(model['labels']))
print(model['sensitive_labels'])
"""
"""
def locate_column(matrix, array):
    # Convert the matrix and array to numpy arrays
    matrix = np.array(matrix).T
    array = np.array(array)
    print(matrix.shape)
    print(array.shape)
    # Check if any column in the matrix is strictly the same as the array
    matches = np.all(matrix == array, axis=0)
    coutnn=0
    record=0
    for line in matrix:
        #print(len(line))
        if line.all()==array.all():
            record=coutnn
        coutnn+=1

    return record

key_num=locate_column(model['node_features'],model['sensitive_labels'])
print(key_num)
"""
graph = nx.DiGraph()

#print(type(model['adjacency_matrix'][0]))

# Get the non-zero indices
rows, cols = model['adjacency_matrix'].nonzero()

# Convert to list of tuples
indices = list(zip(rows, cols))

print(indices)
for edge in indices:
    print(list(edge))
    source, target = int(list(edge)[0]), int(list(edge)[1])
    graph.add_edge(source, target, weight=1)

print(graph[6184])
print([6184,5918] in graph.edges())
#graph.nodes()=sorted(graph.nodes())
with open('./pokec_relationship.G', 'wb') as f:
    pickle.dump(indices, f)

# Calculate the probability of each edge
prob_dict = {}
for source, target in graph.edges():
    probability = graph[source][target]['weight'] / sum(graph[source][neighbor]['weight'] for neighbor in graph.successors(source))
    #print(f"The probability of edge ({source}, {target}) is {probability}")
    prob_dict[(source, target)] = probability

prob_dict = {(int(k[0]), int(k[1])): float(v) for k, v in prob_dict.items()}

with open('./prob.dic', 'wb') as f:
    pickle.dump(str(prob_dict), f)

my_dict = {}

import numpy as np

def normalize_rows(matrix):
    normalized_matrix = []
    for row in matrix:
        min_val = min(row)
        max_val = max(row)
        normalized_row = [(val - min_val) / (max_val - min_val) for val in row]
        normalized_matrix.append(normalized_row)
    return normalized_matrix

def normalize_columns(lst):
    lst_1=np.array(lst)
    lst_2=np.array(normalize_rows(lst_1.T))
    lst_3=lst_2.T
    return lst_3.tolist()

my_dict = {}
my_dict_att = {}
Node_features=model['node_features']
coutnn=0
# Subsample the node feature of the node sets
N = graph.number_of_nodes() + 0
mapping = dict(zip(graph.nodes(), range(0, N)))

for row in Node_features:
    #print(row[-1])
    values=np.append(row[:7],row[-1])
    #print(values)
    values_1 = [values[i:i+4] for i in range(0, len(values), 4)]
    values_1 = np.array(values_1, dtype=float)
    values_1=values_1.tolist()
    if coutnn in list(mapping.keys()):
        #print(mapping[key])
        my_dict[mapping[coutnn]] = values_1
        #print(values_1[1][0])
        my_dict_att[mapping[coutnn]]=values_1[0][1]
    
    coutnn+=1

#Over here we give feature as 2
new_dict = {}
for key, value in my_dict.items():
    new_dict[int(key)] = [[float(i) for i in sublist] for sublist in value]
my_dict=new_dict
#print(my_dict)
with open('./parameter.dic', 'wb') as f:
    pickle.dump(str(my_dict), f)

new_dict_att = {}
for key, value in my_dict_att.items():
    new_dict_att[int(key)] = int(value)
my_dict_att=new_dict_att
#print(my_dict_att)
with open('./gender.dic', 'wb') as f:
    pickle.dump(str(my_dict_att), f)

import numpy as np

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def dice_ratio(a, b):
    intersection = len(set(a) & set(b))
    return 2 * intersection / (len(a) + len(b))

def jaccard_similarity(a, b):
    intersection = len(set(a).intersection(set(b)))
    union = len(set(a).union(set(b)))
    if union == 0:
        return 1
    else:
        return intersection / union

def pearson_correlation(a, b):
    return np.corrcoef(a, b)[0, 1]

import pandas as pd

# read the csv file into a dataframe
reader=pd.DataFrame(model['node_features'])


Edge_feature_dic={}
# Add the edges to the graph

To_load=[]
from sklearn import preprocessing
import numpy as np

for edge in graph.edges():
    source, target = list(edge)[0], list(edge)[1]
    #reader = reader.apply((reader[reader.iloc[:, 0] == source],reader[reader.iloc[:, 0] == target]), axis=1)
    empty_arr=[]
    empty_arr.append(cosine_similarity(model['node_features'][int(source)][1:7], model['node_features'][int(target)][1:7]))
    empty_arr.append(dice_ratio(model['node_features'][int(source)][1:7], model['node_features'][int(target)][1:7]))
    empty_arr.append(jaccard_similarity(model['node_features'][int(source)][1:7], model['node_features'][int(target)][1:7]))
    empty_arr.append(pearson_correlation(model['node_features'][int(source)][1:7], model['node_features'][int(target)][1:7]))
    To_load.append(empty_arr)
#print(To_load)
To_load=normalize_columns(To_load)
#print("after normalized to_load",To_load)
record=0
for edge in graph.edges():
    source, target = list(edge)[0], list(edge)[1]
    if source in mapping.keys() and target in mapping.keys():
        source=mapping[source]
        target=mapping[target]
        Edge_feature_dic[(source, target)]=To_load[record]
        record+=1
#print(Edge_feature_dic)
Edge_feature_dic = {(int(k[0]), int(k[1])): [float(x) for x in v] for k, v in Edge_feature_dic.items()}
with open('./edge_feature.dic', 'wb') as f:
    pickle.dump(str(Edge_feature_dic), f)









