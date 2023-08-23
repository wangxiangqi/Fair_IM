''' Independent cascade model for influence propagation
'''
__author__ = 'ivanovsergey'
from copy import deepcopy
from random import random
import numpy as np

def runICmodel_n (G, S, class_feature, P):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    class_feature -- class feature to implement group fairness
    Output: T -- defined reward of Fair_IM
    '''
    reward = 0
    #这里加入一个subset_G,并且subset_G的class_feature对应的class
    T = deepcopy(S) # copy already selected nodes
    E = {}
    Estimation_of_each_class={}
    Number_of_each_class={}
    for v in G.nodes():
        Number_of_each_class.setdefault(class_feature[v],[])
        Estimation_of_each_class.setdefault(class_feature[v],[])
    for v in G.nodes():
        Number_of_each_class[class_feature[v]].append('1')
    #print("Number_of_each_class",len(Number_of_each_class))
    # ugly C++ version
    i = 0
    while i < len(T):
        for v in G[T[i]]: # for neighbors of a selected node                
            w = G[T[i]][v]['weight'] # count the number of edges between two nodes
            if random() <= 1 - (1-G[T[i]][v]['weight'])**w: # if at least one of edges propagate influence
                    # print T[i], 'influences', v
                if v not in T: # if it wasn't selected yet
                    Estimation_of_each_class[class_feature[v]].append('1')
                    T.append(v)
                if (T[i], v) in E:
                    E[(T[i], v)] += 1
                else:
                    E[(T[i], v)] = 1
        i += 1
    reward=999
    record=0
    #print("Estimate",Estimation_of_each_class)
    for key,value in Estimation_of_each_class.items():
        estimate=len(Estimation_of_each_class[key])
        all_num=len(Number_of_each_class[key])
        #print("key, value",key, estimate/all_num)
        if estimate/all_num != 0 and estimate/all_num < reward:
            reward = estimate/all_num
            record=estimate
    
    #reward = int(reward)
    return reward, T, E

def runICmodel (G, S, P):
    ## I also modified this model.
    #因为在上述的reward中，得到的reward为当前类影响节点的期望除以当前类的大小
    #所以在现在的ICmodel中，同样未来归一化，应该描述
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''

    T = deepcopy(S) # copy already selected nodes
    E = {}

    # ugly C++ version
    i = 0
    while i < len(T):
        for v in G[T[i]]: # for neighbors of a selected node
            if v not in T: # if it wasn't selected yet
                w = G[T[i]][v]['weight'] # count the number of edges between two nodes
                if random() <= 1 - (1-P[T[i]][v]['weight'])**w: # if at least one of edges propagate influence
                    # print T[i], 'influences', v
                    T.append(v)
                    E[(T[i], v)] = 1
        i += 1

    return len(T), T, E

def runIC (G, S, p = .01):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''
    from copy import deepcopy
    from random import random
    T = deepcopy(S) # copy already selected nodes
    E = {}

    # ugly C++ version
    i = 0
    while i < len(T):
        for v in G[T[i]]: # for neighbors of a selected node
            if v not in T: # if it wasn't selected yet
                w = G[T[i]][v]['weight'] # count the number of edges between two nodes
                if random() <= 1 - (1-p)**w: # if at least one of edges propagate influence
                    # print T[i], 'influences', v
                    T.append(v)
                    E[(T[i], v)] = 1
        i += 1

    # neat pythonic version
    # legitimate version with dynamically changing list: http://stackoverflow.com/a/15725492/2069858
    # for u in T: # T may increase size during iterations
    #     for v in G[u]: # check whether new node v is influenced by chosen node u
    #         w = G[u][v]['weight']
    #         if v not in T and random() < 1 - (1-p)**w:
    #             T.append(v)

    return len(T), T, E

def runIC2(G, S, p=.01):
    ''' Runs independent cascade model (finds levels of propagation).
    Let A0 be S. A_i is defined as activated nodes at ith step by nodes in A_(i-1).
    We call A_0, A_1, ..., A_i, ..., A_l levels of propagation.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''
    from copy import deepcopy
    import random
    T = deepcopy(S)
    Acur = deepcopy(S)
    Anext = []
    i = 0
    while Acur:
        values = dict()
        for u in Acur:
            for v in G[u]:
                if v not in T:
                    w = G[u][v]['weight']
                    if random.random() < 1 - (1-p)**w:
                        Anext.append((v, u))
        Acur = [edge[0] for edge in Anext]
        print(i, Anext)
        i += 1
        T.extend(Acur)
        Anext = []
    return T
    
def avgSize(G,S,p,iterations):
    avg = 0
    for i in range(iterations):
        avg += float(len(runIC(G,S,p)))/iterations
    return avg
