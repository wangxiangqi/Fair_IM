import numpy as np
import sys
sys.path.append('./fair_influmax_code')
sys.path.append('./Oracle')
from degreeDiscount import degreeDiscountIC, degreeDiscountIC2, degreeDiscountIAC, degreeDiscountIAC2, degreeDiscountStar, degreeDiscountIAC3
from generalGreedy import generalGreedy
from utils import greedy
from icm import sample_live_icm, make_multilinear_objective_samples_group, make_multilinear_gradient_group
from algorithms import algo, maxmin_algo, make_normalized, indicator, make_welfare

group_size = {}
def Fair_IM_oracle(G,K,attributes,P=0.1):

    for attribute in attributes:
        nvalues = len(np.unique([G.node[v]['node_type'][attribute] for v in G.nodes()]))
        group_size[attribute] = np.zeros((1, nvalues))

    live_graphs = sample_live_icm(G, 30)
    group_indicator = np.ones((len(G.nodes()), 1))
        
    val_oracle = make_multilinear_objective_samples_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
    grad_oracle = make_multilinear_gradient_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
        
    def multi_to_set(f, n = None):
        if n == None:
            n = len(G)
        def f_set(S):
            return f(indicator(S, n))
        return f_set


    def f_multi(x):
        return val_oracle(x, 5).sum()
    
    fair_vals_attr = np.zeros((1, len(attributes)))
    greedy_vals_attr = np.zeros((1, len(attributes)))
    pof = np.zeros((1, len(attributes)))
    include_total = False
    #live_graphs = sample_live_icm(G, 40)
    
    group_indicator = np.ones((len(G.nodes()), 1))
        
    val_oracle = make_multilinear_objective_samples_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
    grad_oracle = make_multilinear_gradient_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
        
    def f_multi(x):
        return val_oracle(x, 5).sum()
        
    #f_multi = make_multilinear_objective_samples(live_graphs, list(g.nodes()), list(g.nodes()), np.ones(len(g)))
    f_set = multi_to_set(f_multi)

    def valoracle_to_single(f, i):
        def f_single(x):
            return f(x, 5)[i]
        return f_single
    #print("about to run greedy")
    S_opop, obj = greedy(list(range(len(G))), K, f_set)
    print("S_opop is",S_opop)
    #print("successfully output S_opop")
    set_to_fair=[]
    for attr_idx, attribute in enumerate(attributes):
        print("attribute",attribute)    
        values = np.unique([G.node[v]['node_type'][attribute] for v in G.nodes()])
        print(values)
        nodes_attr = {}
        for vidx, val in enumerate(values):
            nodes_attr[val] = [v for v in G.nodes() if G.node[v]['node_type'][attribute] == val]
            group_size[attribute][0, vidx] = len(nodes_attr[val])
        #print(nodes_attr[val])

        group_indicator = np.zeros((len(G.nodes()), len(values)))
        for val_idx, val in enumerate(values):
                group_indicator[nodes_attr[val], val_idx] = 1

        val_oracle = make_multilinear_objective_samples_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
        grad_oracle = make_multilinear_gradient_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))

        f_attr = {}
        f_multi_attr = {}
        for vidx, val in enumerate(values):
            nodes_attr[val] = [v for v in G.nodes() if G.node[v]['node_type'][attribute] == val]
            f_multi_attr[val] = valoracle_to_single(val_oracle, vidx)
            f_attr[val] = multi_to_set(f_multi_attr[val])

        S_attr = {}
        opt_attr = {}

        print("another greedy?")
        print(len(values))
        for val in values:
            #S_attr=generalGreedy(G,K,0.1)
            S_attr[val], opt_attr[val] = greedy(list(range(len(G))), int(len(nodes_attr[val])/len(G) * K), f_attr[val])

        all_opt = np.array([opt_attr[val] for val in values])

        solver="md"
        threshold = 0.4
        targets = [opt_attr[val] for val in values]
        #print("S_att is",S_attr)
        targets = np.array(targets)
        print('ready to run fair algorithm')
        #The algo overhere is to to run two different algorithm, one focused on Diversity Fairness, another on Maximin fairness

        """
        #The first one is on diversity constraint
        fair_x = algo(grad_oracle, val_oracle, threshold, K, group_indicator, np.array(targets), 5, solver)[1:]
        #print("fair_x output")
        #print(fair_x)
        fair_x = fair_x.mean(axis=0)
        for sublist in fair_x:
            m=val_oracle(sublist,20).mean()
            if isinstance(m,float) and 0<=m and m<len(G.nodes()):
                set_to_fair.append(int(m))
        fair_vals=val_oracle(fair_x,20)
        for m in fair_vals:
        #    #print(val_oracle(sublist, 400).mean())
        #    #m= val_oracle(sublist, 20).mean()
        #    #print(m)
            if isinstance(m, float) and 0 <= m and m<len(G.nodes()):
                set_to_fair.append(int(m))

        
        # The second is on the maximin constraint
        """
        print("ready to run maximin oracle")
        grad_oracle_normalized = make_normalized(grad_oracle, group_size[attribute][0])
        val_oracle_normalized = make_normalized(val_oracle, group_size[attribute][0])
        minmax_x = maxmin_algo(grad_oracle_normalized, val_oracle_normalized, threshold, K, group_indicator, 10, 5, 0.1, solver)
        minmax_x = minmax_x.mean(axis=0)
        fair_vals=val_oracle(minmax_x,20)
        for m in fair_vals:
            #print(val_oracle(sublist, 400).mean())
        #    #m= val_oracle(sublist, 20).mean()
        #    #print(m)
            #m=val_oracle(sublist,20).mean()
            if isinstance(m, float) and 0 <= m and m<len(G.nodes()):
                set_to_fair.append(int(m))
        
        
        
        #for sublist in minmax_x:
        #    m=val_oracle(sublist,20).mean()
        #    if isinstance(m,float) and 0<=m and m<len(G.nodes()):
        #        set_to_fair.append(int(m))

        
        #This is another to calculate fair_set

        #This is on the welfare
        #print("ready to run welfare oracle")
        
        

        
    set_to_fair=np.unique(set_to_fair)
    #set_to_fair.drop(2000)
    #print("set_to_fair is",set_to_fair)
    #We have parameter to the number of graph
    set_to_fair = [item for item in set_to_fair if 0 <= item <= len(G.nodes())]
    #set_to_fair=set_to_fair.tolist()
    set_to_fair.extend([item for item in S_opop if item not in set_to_fair])
    set_to_fair=set_to_fair[:K]
    #print("set_to_fair is",set_to_fair)
    print("set_to_fair is",set_to_fair)
    return set_to_fair

group_size = {}
def Fair_IM_oracle_wel(G,K,attributes,P=0.1,alpha=0.5):
    for attribute in attributes:
        nvalues = len(np.unique([G.node[v]['node_type'][attribute] for v in G.nodes()]))
        group_size[attribute] = np.zeros((1, nvalues))

    live_graphs = sample_live_icm(G, 30)
    group_indicator = np.ones((len(G.nodes()), 1))
        
    val_oracle = make_multilinear_objective_samples_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
    grad_oracle = make_multilinear_gradient_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
        
    def multi_to_set(f, n = None):
        if n == None:
            n = len(G)
        def f_set(S):
            return f(indicator(S, n))
        return f_set


    def f_multi(x):
        return val_oracle(x, 5).sum()
    
    fair_vals_attr = np.zeros((1, len(attributes)))
    greedy_vals_attr = np.zeros((1, len(attributes)))
    pof = np.zeros((1, len(attributes)))
    include_total = False
    #live_graphs = sample_live_icm(G, 40)
    
    group_indicator = np.ones((len(G.nodes()), 1))
        
    val_oracle = make_multilinear_objective_samples_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
    grad_oracle = make_multilinear_gradient_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
        
    def f_multi(x):
        return val_oracle(x, 10).sum()
        
    #f_multi = make_multilinear_objective_samples(live_graphs, list(g.nodes()), list(g.nodes()), np.ones(len(g)))
    f_set = multi_to_set(f_multi)

    def valoracle_to_single(f, i):
        def f_single(x):
            return f(x, 10)[i]
        return f_single
    #print("about to run greedy")
    S_opop, obj = greedy(list(range(len(G))), K, f_set)
    print("S_opop is",S_opop)
    #print("successfully output S_opop")
    set_to_fair=[]
    for attr_idx, attribute in enumerate(attributes):
        print("attribute",attribute)    
        values = np.unique([G.node[v]['node_type'][attribute] for v in G.nodes()])
        print(values)
        nodes_attr = {}
        for vidx, val in enumerate(values):
            nodes_attr[val] = [v for v in G.nodes() if G.node[v]['node_type'][attribute] == val]
            group_size[attribute][0, vidx] = len(nodes_attr[val])
        #print(nodes_attr[val])

        group_indicator = np.zeros((len(G.nodes()), len(values)))
        for val_idx, val in enumerate(values):
                group_indicator[nodes_attr[val], val_idx] = 1

        val_oracle = make_multilinear_objective_samples_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
        grad_oracle = make_multilinear_gradient_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))

        f_attr = {}
        f_multi_attr = {}
        for vidx, val in enumerate(values):
            nodes_attr[val] = [v for v in G.nodes() if G.node[v]['node_type'][attribute] == val]
            f_multi_attr[val] = valoracle_to_single(val_oracle, vidx)
            f_attr[val] = multi_to_set(f_multi_attr[val])

        S_attr = {}
        opt_attr = {}

        print("another greedy?")
        print(len(values))
        for val in values:
            #S_attr=generalGreedy(G,K,0.1)
            S_attr[val], opt_attr[val] = greedy(list(range(len(G))), int(len(nodes_attr[val])/len(G) * K), f_attr[val])

        all_opt = np.array([opt_attr[val] for val in values])

        solver="md"
        threshold = 100
        targets = [opt_attr[val] for val in values]
        #print("S_att is",S_attr)
        targets = np.array(targets)
        print('ready to run welfare fair algorithm')
        #The algo overhere is to to run two different algorithm, one focused on Diversity Fairness, another on Maximin fairness
        #grad_oracle_welfare = make_welfare(grad_oracle, group_size[attribute][0],alpha)
        #val_oracle_welfare = make_welfare(val_oracle, group_size[attribute][0])
        grad_oracle=make_welfare(grad_oracle, group_size[attribute][0], alpha)
        grad_oracle=make_welfare(grad_oracle,group_size[attribute][0], alpha)
        wel_x = algo(grad_oracle, val_oracle, threshold, K, group_indicator, np.array(targets), 5, solver)[1:]
        wel_x = wel_x.mean(axis=0)
        for sublist in wel_x:
            #print(val_oracle(sublist, 400).mean())
        #    #m= val_oracle(sublist, 20).mean()
        #    #print(m)
            m=val_oracle(sublist,20).mean()
            if isinstance(m, float) and 0 <= m and m<len(G.nodes()):
                set_to_fair.append(int(m))

    set_to_fair=np.unique(set_to_fair)
    #set_to_fair.drop(2000)
    #print("set_to_fair is",set_to_fair)
    #We have parameter to the number of graph
    set_to_fair = [item for item in set_to_fair if 0 <= item <= len(G.nodes())]
    #set_to_fair=set_to_fair.tolist()
    set_to_fair.extend([item for item in S_opop if item not in set_to_fair])
    set_to_fair=set_to_fair[:K]
    #print("set_to_fair is",set_to_fair)
    print("set_to_fair is",set_to_fair)
    return set_to_fair

group_size = {}
def optimal_gred(G,K,attributes,P=0.1,alpha=0.5):
    for attribute in attributes:
        nvalues = len(np.unique([G.node[v]['node_type'][attribute] for v in G.nodes()]))
        group_size[attribute] = np.zeros((1, nvalues))

    live_graphs = sample_live_icm(G, 30)
    group_indicator = np.ones((len(G.nodes()), 1))
        
    val_oracle = make_multilinear_objective_samples_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
    grad_oracle = make_multilinear_gradient_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
        
    def multi_to_set(f, n = None):
        if n == None:
            n = len(G)
        def f_set(S):
            return f(indicator(S, n))
        return f_set


    def f_multi(x):
        return val_oracle(x, 20).sum()
    
    fair_vals_attr = np.zeros((1, len(attributes)))
    greedy_vals_attr = np.zeros((1, len(attributes)))
    pof = np.zeros((1, len(attributes)))
    include_total = False
    #live_graphs = sample_live_icm(G, 40)
    
    group_indicator = np.ones((len(G.nodes()), 1))
        
    val_oracle = make_multilinear_objective_samples_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
    grad_oracle = make_multilinear_gradient_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
        
    def f_multi(x):
        return val_oracle(x, 10).sum()
        
    #f_multi = make_multilinear_objective_samples(live_graphs, list(g.nodes()), list(g.nodes()), np.ones(len(g)))
    f_set = multi_to_set(f_multi)

    def valoracle_to_single(f, i):
        def f_single(x):
            return f(x, 10)[i]
        return f_single
    #print("about to run greedy")
    S_opop, obj = greedy(list(range(len(G))), K, f_set)
    return list(S_opop)
