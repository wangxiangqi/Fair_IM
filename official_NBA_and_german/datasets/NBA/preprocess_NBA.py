import networkx as nx
import pickle
import numpy as np
# Create an empty ded graph
from sklearn import preprocessing
graph = nx.DiGraph()

# Open the text file and read the edges
with open('nba_relationship.txt', 'r') as f:
    edges = f.readlines()

# Add the edges to the graph
for edge in edges:
    source, target = edge.strip().split()
    graph.add_edge(source, target, weight=1)

# Save the graph in .G file format
N = graph.number_of_nodes() + 0
mapping = dict(zip(graph.nodes(), range(0, N)))
#print(list(mapping.keys()))
graph= nx.convert_node_labels_to_integers(graph, label_attribute='pid')
with open('nba_relationship.G', 'wb') as f:
    pickle.dump(graph, f)

# Calculate the probability of each edge
prob_dict = {}
for source, target in graph.edges():
    probability = graph[source][target]['weight'] / sum(graph[source][neighbor]['weight'] for neighbor in graph.successors(source))
    #print(f"The probability of edge ({source}, {target}) is {probability}")
    prob_dict[(source, target)] = probability

prob_dict = {(int(k[0]), int(k[1])): float(v) for k, v in prob_dict.items()}

with open('prob.dic', 'wb') as f:
    pickle.dump(str(prob_dict), f)

import csv

# Open the .csv file and read the rows
with open('nba.csv', 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)

rows=rows[1:]
# Create a dictionary from the rows
my_dict = {}



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

my_dict_att = {}
for row in rows:
    key = row[0]
    #print(mapping[key])
    values = row[4:8]+list(row[37])+row[9:12]
    #print(row[37])
    #print(values)
    values_1 = [values[i:i+4] for i in range(0, len(values), 4)]
    #print(values_1)
    values_1 = np.array(values_1, dtype=float)
    #print(values_1[1][0])
    #scaler = preprocessing.StandardScaler()
    # 使用StandardScaler对象拟合并转换矩阵
    #values_1 = scaler.fit_transform(values_1)
    #values_1=normalize_columns(values_1)
    values_1=values_1.tolist()
    #print("key is",key)
    #print(list(mapping.keys()))
    #print(mapping[key])
    #print(key in list(mapping.keys()))
    if key in list(mapping.keys()):
        #print(mapping[key])
        my_dict[mapping[key]] = values_1
        #print(values_1[1][0])
        my_dict_att[mapping[key]]=values_1[1][0]

#print(my_dict_att)
#print(my_dict)
# Save the dictionary to a .dic file
new_dict = {}
for key, value in my_dict.items():
    new_dict[int(key)] = [[float(i) for i in sublist] for sublist in value]
my_dict=new_dict
#print(my_dict)
with open('parameter.dic', 'wb') as f:
    pickle.dump(str(my_dict), f)

new_dict_att = {}
for key, value in my_dict_att.items():
    new_dict_att[int(key)] = int(value)
my_dict_att=new_dict_att
#print(my_dict_att)
with open('country.dic', 'wb') as f:
    pickle.dump(str(my_dict_att), f)


## 现在的主要的问题是如何构造edge features，这里的edge features应该基于relationship和 attribute来构造。
## 目前的想法是可以利用两个node在csv构造的vector计算相似度，可以选取多种相似度，比如cosine, jaccard, pearson, Euclidean
# Open the text file and read the edges
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

with open('nba_relationship.txt', 'r') as f:
    edges = f.readlines()

import pandas as pd

# read the csv file into a dataframe
reader = pd.read_csv('nba.csv')


Edge_feature_dic={}
# Add the edges to the graph

To_load=[]
from sklearn import preprocessing
import numpy as np

for edge in edges:
    source, target = edge.strip().split()
    #reader = reader.apply((reader[reader.iloc[:, 0] == source],reader[reader.iloc[:, 0] == target]), axis=1)
    empty_arr=[]
    empty_arr.append(cosine_similarity(reader[reader.iloc[:, 0] == int(source)].stack()[1:7], reader[reader.iloc[:, 0] == int(target)].stack()[1:7]))
    empty_arr.append(dice_ratio(reader[reader.iloc[:, 0] == int(source)].stack()[1:7], reader[reader.iloc[:, 0] == int(target)].stack()[1:7]))
    empty_arr.append(jaccard_similarity(reader[reader.iloc[:, 0] == int(source)].stack()[1:7], reader[reader.iloc[:, 0] == int(target)].stack()[1:7]))
    empty_arr.append(pearson_correlation(reader[reader.iloc[:, 0] == int(source)].stack()[1:7], reader[reader.iloc[:, 0] == int(target)].stack()[1:7]))
    To_load.append(empty_arr)
#print(To_load)
To_load=normalize_columns(To_load)
#print("after normalized to_load",To_load)
record=0
for edge in edges:
    source, target = edge.strip().split()
    if source in mapping.keys() and target in mapping.keys():
        source=mapping[source]
        target=mapping[target]
        Edge_feature_dic[(source, target)]=To_load[record]
        record+=1
#print(Edge_feature_dic)
Edge_feature_dic = {(int(k[0]), int(k[1])): [float(x) for x in v] for k, v in Edge_feature_dic.items()}
with open('edge_feature.dic', 'wb') as f:
    pickle.dump(str(Edge_feature_dic), f)



