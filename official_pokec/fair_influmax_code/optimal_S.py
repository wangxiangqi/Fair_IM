import networkx as nx
import numpy as np
import pickle
from utils import greedy
from icm import sample_live_icm, make_multilinear_objective_samples_group, make_multilinear_gradient_group
from algorithms import algo, maxmin_algo, make_normalized, indicator
import math

def multi_to_set(f, n = None):
    '''
    Takes as input a function defined on indicator vectors of sets, and returns
    a version of the function which directly accepts sets
    '''
    if n == None:
        n = len(g)
    def f_set(S):
        return f(indicator(S, n))
    return f_set

def valoracle_to_single(f, i):
    def f_single(x):
        return f(x, 100)[i]
    return f_single

budget = 20
print('Budget: {}'.format(budget))


#whether fair influence share is calculated by forming a subgraph consisting
#only of nodes in a given subgroup -- leave this to True for the setting in 
#the paper
succession = True

#what method to use to solve the inner maxmin LP. This can be either 'md' to
#use stochastic saddlepoint mirror descent, or 'gurobi' to solve the LP explicitly
#the the gurobi solver (requires an installation and license).
#MD is better asymptotically for large networks but may require many iterations
#and is typically slower than gurobi for small/medium sized problems. You can
#tune the stepsize/batch size/number of iterations for MD by editing algorithms.py
solver = 'md'

#attribute to examine fairness wrt
#attributes = ['birthsex', 'gender', 'sexori', 'race']
attributes = ['key5']


#network -> attribute -> n_runs * n_values
gr_values = {}
#network -> attribute -> n_runs * n_values
group_size = {}
#algorithm -> network -> attribute -> n_runs * n_values
alg_values = {}

num_runs = 1
algorithms = ['Greedy', 'GR', 'MaxMin-Size']
for alg in algorithms:
    alg_values[alg] = {}

num_graphs = 1
graphnames = ['Flickr']
print(graphnames)

for graphname in graphnames:

    g = pickle.load(open('networks/{}.pickle'.format(graphname), 'rb'),encoding='latin1') 
    print("already loaded graph")
    #remove nodes without demographic information
    #这一部分可以不去处理了
    
    #propagation probability for the ICM
    p = 0.1
    for u,v in g.edges():
        g[u][v]['p'] = p
    
    #g = nx.convert_node_labels_to_integers(g, label_attribute='pid')
    #with open('./networks/NBA.pickle', 'wb') as f:
    #    pickle.dump(g, f) 
    gr_values[graphname] = {}
    group_size[graphname] = {}
    for alg in algorithms:
        alg_values[alg][graphname] = {}
    #for attribute in attributes:
            #assign a unique numeric value for nodes who left the attribute blank
            #这一部分在Flickr数据集下也可以不用考虑。
    """
            nvalues = len(np.unique([g.node[v][attribute] for v in g.nodes()]))
            if 'spa' not in graphname:
                for v in g.nodes():
                    if np.isnan(g.node[v][attribute]):
                        g.node[v][attribute] = nvalues
            nvalues = len(np.unique([g.node[v][attribute] for v in g.nodes()]))
            gr_values[graphname][attribute] = np.zeros((num_runs, nvalues))
            group_size[graphname][attribute] = np.zeros((num_runs, nvalues))
            for alg in algorithms:
                alg_values[alg][graphname][attribute] = np.zeros((num_runs, nvalues))
    """

    
    fair_vals_attr = np.zeros((num_runs, len(attributes)))
    greedy_vals_attr = np.zeros((num_runs, len(attributes)))
    pof = np.zeros((num_runs, len(attributes)))
    
    include_total = False
    print("ready to run")
    for run in range(num_runs):
        print(graphname, run)
        live_graphs = sample_live_icm(g, 5)
        #print("loaded live graph")
        print(live_graphs)
        group_indicator = np.ones((len(g.nodes()), 1))
        
        val_oracle = make_multilinear_objective_samples_group(live_graphs, group_indicator,  list(g.nodes()), list(g.nodes()), np.ones(len(g)))
        grad_oracle = make_multilinear_gradient_group(live_graphs, group_indicator,  list(g.nodes()), list(g.nodes()), np.ones(len(g)))
        print("loaded oracle")
        def f_multi(x):
            return val_oracle(x, 5).sum()
        
        
        #f_multi = make_multilinear_objective_samples(live_graphs, list(g.nodes()), list(g.nodes()), np.ones(len(g)))
        f_set = multi_to_set(f_multi)
        print("run optimal set")
        #find overall optimal solution
        S, obj = greedy(list(range(len(g))), budget, f_set)
        with open('optimal_set.pkl', 'wb') as f:
            # serialize and save set to file
            pickle.dump(S, f)