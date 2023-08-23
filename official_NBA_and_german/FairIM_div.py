import time
import os
import pickle 
import datetime
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from conf import *
#from preprocess import *
from Tool.utilFunc import *

from BanditAlg.CUCB import UCB1Algorithm
from BanditAlg.DILinUCB import N_LinUCBAlgorithm
from BanditAlg.greedy import eGreedyAlgorithm
from BanditAlg.IMFB import MFAlgorithm
#from BanditAlg.IMLinUCB import N_LinUCBAlgorithm
from IC.IC import runIC, runICmodel, runICmodel_n
from IC.runIAC  import weightedEp, runIAC, runIACmodel, randomEp, uniformEp
import sys
sys.path.append('E:/summer_intern/Hua_zheng_Wang/Fair_IM/IMFB-KDD2019-master/IMFB-KDD2019-master/fair_influmax_code')
from utils import greedy
from icm import sample_live_icm, make_multilinear_objective_samples_group, make_multilinear_gradient_group
from algorithms import algo, maxmin_algo, make_normalized, indicator

alpha=-1


def multi_to_set(f, n = None):
    '''
    Takes as input a function defined on indicator vectors of sets, and returns
    a version of the function which directly accepts sets
    '''
    if n == None:
        n = len(G)
    def f_set(S):
        return f(indicator(S, n))
    return f_set

def Get_reward(G,attributes,seed_size,live_nodes,live_edges):
    #Implement reward under maximin restriction:
    """
    Maximin_reward=[]
    for attribute in attributes:
        values = np.unique([G.node[v]['node_type'][attribute] for v in G.nodes()])
        nodes_attr = {}
        for vidx, val in enumerate(values):
            nodes_attr[val] = [v for v in G.nodes() if G.node[v]['node_type'][attribute] == val]
        temp_arr=[]
        for key,item in nodes_attr.items():
            set_of_values=set(item)
            set_of_live_nodes=set(live_nodes)
            intersection = set_of_values.intersection(set_of_live_nodes)
            # Convert the intersection set back to a list (if needed)
            intersection_list = list(intersection)
            temp_arr.append(len(intersection_list)/len(set_of_values))
        Maximin_reward.append(temp_arr)
    Maximin_reward=[item for sublist in Maximin_reward for item in sublist]
    Maximin_reward=min(Maximin_reward)

    #Implement reward under diversity restriction:
    """
    Diversity_reward=[]
    for attribute in attributes:
        values = np.unique([G.node[v]['node_type'][attribute] for v in G.nodes()])
        nodes_attr = {}
        for vidx, val in enumerate(values):
            nodes_attr[val] = [v for v in G.nodes() if G.node[v]['node_type'][attribute] == val]
        temp_arr=[]
        for key,item in nodes_attr.items():
            set_of_values=set(item)
            set_of_live_nodes=set(live_nodes)
            intersection = set_of_values.intersection(set_of_live_nodes)
            k_i=int(seed_size*len(nodes_attr[key])/(len(G.nodes())))
            #Number of all influenced
            #In diversity constraint, we need to extract a subgraph of it
            subGraph=G.subgraph(item)
            chosen_item=random.sample(item,k_i)
            optimal_reward_1, live_nodes_1, live_edges_1 = runIC(subGraph, chosen_item)
            set_of_live_nodes_1=set(live_nodes_1)
            intersection_1 = set_of_values.intersection(set_of_live_nodes_1)
            # Here add diversity constraint:
            if len(intersection_1)>len(intersection):
                temp_arr.append(0)
            else:
                temp_arr.append(len(intersection)/len(set_of_values))
        Diversity_reward.append(temp_arr)
    Diversity_reward=[item for sublist in Diversity_reward for item in sublist]
    Diversity_reward=sum(Diversity_reward) / len(Diversity_reward)
    #sum_array = [a + b for a, b in zip(Maximin_reward, Diversity_reward)]
    
    return Diversity_reward

def Get_reward_wel(G,attributes,seed_size,live_nodes,live_edges):
    #Implement reward under maximin restriction:
    Welfare_reward=[]
    for attribute in attributes:
        values = np.unique([G.node[v]['node_type'][attribute] for v in G.nodes()])
        nodes_attr = {}
        for vidx, val in enumerate(values):
            nodes_attr[val] = [v for v in G.nodes() if G.node[v]['node_type'][attribute] == val]
        temp_arr=[]
        for key,item in nodes_attr.items():
            set_of_values=set(item)
            set_of_live_nodes=set(live_nodes)
            intersection = set_of_values.intersection(set_of_live_nodes)
            # Convert the intersection set back to a list (if needed)
            intersection_list = list(intersection)
            influence_ratio=len(intersection_list)/len(set_of_values)
            temp_arr.append(influence_ratio)
        for ratio,(key,item) in zip(temp_arr,nodes_attr.items()):
            Welfare_reward.append(len(item)*(ratio**alpha)/alpha)
    reward=sum(Welfare_reward)
    #Implement reward under diversity restriction:
    #sum_array = [a + b for a, b in zip(Maximin_reward, Diversity_reward)]
    return reward


class simulateOnlineData:
    def __init__(self, G, P, oracle,class_feature, attributes, seed_size, iterations, dataset):
        self.G = G
        self.TrueP = P
        self.seed_size = seed_size
        self.oracle = oracle
        self.iterations = iterations
        self.class_feature = class_feature
        self.attributes = attributes
        self.dataset = dataset
        self.startTime = datetime.datetime.now()
        self.BatchCumlateReward = {}
        self.AlgReward = {}
        self.result_oracle = []

    def runAlgorithms(self, algorithms):
        self.tim_ = []
        for alg_name, alg in list(algorithms.items()):
            self.AlgReward[alg_name] = []
            self.BatchCumlateReward[alg_name] = []

        self.resultRecord()
        optS = self.oracle(self.G, self.seed_size,self.attributes,0.1)
        #print(optS)
        with open('original_set.pkl', 'wb') as f:
            # serialize and save set to file
            pickle.dump(optS, f)
        print("successfully dumped data")
        UCB1=[]
        IMFB=[]
        e_gred=[]
        #DILinUCB=[]
        optimal_reward, live_nodes, live_edges = runICmodel(G, optS,self.TrueP)
        optimal_reward=Get_reward(G,self.attributes,self.seed_size,live_nodes,live_edges)
        #total_optimal=self.iterations*optimal_reward
        total_influence_UCB=[]
        total_influence_IMFB=[]
        total_influence_egred=[]
        for iter_ in range(self.iterations):
            optimal_reward, live_nodes, live_edges = runICmodel(G, optS,self.TrueP)
            optimal_reward=Get_reward(G,self.attributes,self.seed_size,live_nodes,live_edges)
            self.result_oracle.append(optimal_reward)
            print('oracle', optimal_reward)
            
            for alg_name, alg in list(algorithms.items()): 
                S = alg.decide() 
                reward, live_nodes, live_edges = runICmodel(G, S,self.TrueP)
                reward=Get_reward(G,self.attributes,self.seed_size,live_nodes,live_edges)
                print("reward gap",optimal_reward-reward)
                if iter_==0 and ('{}'.format(alg_name))=='UCB1':
                    UCB1.append(optimal_reward-reward)
                    total_influence_UCB.append(live_nodes)
                elif ('{}'.format(alg_name))=='UCB1':
                    UCB1.append(UCB1[iter_-1]+optimal_reward-reward)
                    total_influence_UCB.append(live_nodes)
                
                if iter_==0 and ('{}'.format(alg_name))=='IMFB':
                    IMFB.append(optimal_reward-reward)
                    total_influence_IMFB.append(live_nodes)
                elif ('{}'.format(alg_name))=='IMFB':
                    IMFB.append(IMFB[iter_-1]+optimal_reward-reward)
                    total_influence_IMFB.append(live_nodes)
                
                if iter_==0 and ('{}'.format(alg_name))=='egreedy_0.1':
                    e_gred.append(optimal_reward-reward)
                    total_influence_egred.append(live_nodes)
                elif ('{}'.format(alg_name))=='egreedy_0.1':
                    e_gred.append(e_gred[iter_-1]+optimal_reward-reward)
                    total_influence_egred.append(live_nodes)

                alg.updateParameters(S, live_nodes, live_edges, iter_)

                self.AlgReward[alg_name].append(reward)

            self.resultRecord(iter_)
        for alg_name, alg in list(algorithms.items()): 
            plt.plot(UCB1)
            with open('UCB1_set_pokec_div_fair.pkl', 'wb') as f:
            # serialize and save set to file
                pickle.dump(UCB1, f)
            plt.plot(IMFB)
            with open('IMFB_set_pokec_div_fair.pkl', 'wb') as f:
            # serialize and save set to file
                pickle.dump(IMFB, f)
            plt.plot(e_gred)
            with open('egred_set_pokec_div_fair.pkl', 'wb') as f:
            # serialize and save set to file
                pickle.dump(e_gred, f)
            #plt.plot(DILinUCB)
            #with open('DILInCUB_set_100_div.pkl', 'wb') as f:
            # serialize and save set to file
            #    pickle.dump(DILinUCB, f)
        plt.xlabel('Time Steps')
        plt.ylabel('Regret')
        plt.title('Regret in Bandit Problem')
        plt.show()
        self.showResult()

    def resultRecord(self, iter_=None):
        # if initialize
        if iter_ is None:
            timeRun = self.startTime.strftime('_%m_%d_%H_%M_%S') 
            fileSig = '_seedsize'+str(self.seed_size) + '_iter'+str(self.iterations)+'_'+str(self.oracle.__name__)+'_'+self.dataset
            self.filenameWriteReward = os.path.join(save_address, 'AccReward' + timeRun + fileSig + '.csv')

            with open(self.filenameWriteReward, 'w') as f:
                f.write('Time(Iteration)')
                f.write(',' + ','.join( [str(alg_name) for alg_name in algorithms.keys()]))
                f.write('\n') 
        else:
            # if run in the experiment, save the results
            print("Iteration %d" % iter_, " Elapsed time", datetime.datetime.now() - self.startTime)
            self.tim_.append(iter_)
            for alg_name in algorithms.keys():
                self.BatchCumlateReward[alg_name].append(sum(self.AlgReward[alg_name][-1:]))
            with open(self.filenameWriteReward, 'a+') as f:
                f.write(str(iter_))
                f.write(',' + ','.join([str(self.BatchCumlateReward[alg_name][-1]) for alg_name in algorithms.keys()]))
                f.write('\n')

    def showResult(self):
        print('average reward for oracle:', np.mean(self.result_oracle))
        
        f, axa = plt.subplots(1, sharex=True)
        for alg_name in algorithms.keys():  
            axa.plot(self.tim_, self.BatchCumlateReward[alg_name],label = alg_name)
            print('%s: %.2f' % (alg_name, np.mean(self.BatchCumlateReward[alg_name])))
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Reward")
        axa.set_title("Average Reward")
        plt.savefig('./SimulationResults/AvgReward' + str(self.startTime.strftime('_%m_%d_%H_%M'))+'.pdf')
        plt.show()
        # plot accumulated reward
        f, axa = plt.subplots(1, sharex=True)
        for alg_name in algorithms.keys():  
            result = [sum(self.BatchCumlateReward[alg_name][:i]) for i in range(len(self.tim_))]
            axa.plot(self.tim_, result, label = alg_name)
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Reward")
        axa.set_title("Accumulated Reward")
        plt.savefig('./SimulationResults/AcuReward' + str(self.startTime.strftime('_%m_%d_%H_%M'))+'.pdf')
        plt.show()

        for alg_name in algorithms.keys():  
            try:
                loss = algorithms[alg_name].getLoss()
            except:
                continue
            print("loss is",loss)
            f, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel('Loss of Probability', color=color)
            ax1.plot(self.tim_, loss[:, 0], color=color, label='Probability')
            ax1.tick_params(axis='y', labelcolor=color)

            #ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            #ax2.set_ylabel('Loss of Theta and Beta', color='tab:blue')  # we already handled the x-label with ax1
            #ax2.plot(self.tim_, loss[:, 1], color='tab:blue', linestyle=':', label = r'$\theta$')
            #ax2.plot(self.tim_, loss[:, 2], color='tab:blue', linestyle='--', label = r'$\beta$')
            #ax2.tick_params(axis='y', labelcolor='tab:blue')
            #ax2.legend(loc='upper left',prop={'size':9})
            #f.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.savefig('./SimulationResults/Loss' + str(self.startTime.strftime('_%m_%d_%H_%M'))+'.pdf')
            plt.show()
            np.save('./SimulationResults/Loss-{}'.format(alg_name) + str(self.startTime.strftime('_%m_%d_%H_%M'))+'.npy', loss)

import ast     
if __name__ == '__main__':
    start = time.time()
    G = pickle.load(open(graph_address, 'rb'), encoding='latin1')
    
    parameter = pickle.load(open(param_address, 'rb'), encoding='latin1')
    parameter = ast.literal_eval(parameter)
    Node_attr={}
    for key in parameter:
        if key in G.nodes():
            value=parameter[key]
            merged_list = [val for sublist in value for val in sublist]
            my_dict = {f'key{i+1}': val for i, val in enumerate(merged_list)}
            my_dict = {key: my_dict}
            Node_attr.update(my_dict)
    import networkx as nx
    #print(Node_attr)
    nx.set_node_attributes(G, values = Node_attr, name='node_type') 


    N = G.number_of_nodes() + 0
    mapping = dict(zip(G.nodes(), range(0, N)))
    G = nx.convert_node_labels_to_integers(G, label_attribute='pid')
    #prob = pickle.load(open(prob_address, 'rb'), encoding='latin1')
    #prob = ast.literal_eval(prob)
    transformed_dict = {}
    for key in parameter.keys():
        mapped_x1 = mapping.get(key)
        transformed_dict[mapped_x1] = parameter[key]
    parameter=transformed_dict
    feature_dic = pickle.load(open(edge_feature_address, 'rb'), encoding='latin1')
    feature_dic = ast.literal_eval(feature_dic)
    transformed_dict = {}
    for key in feature_dic.keys():
        x1, x2 = str(key).split(',')
        mapped_x1 = mapping.get(x1)
        mapped_x2 = mapping.get(x2)
        transformed_key = f"{mapped_x1},{mapped_x2}"
        transformed_dict[transformed_key] = feature_dic[key]
    feature_dic=transformed_dict
    class_feature = pickle.load(open(class_feature_address, 'rb'),encoding='latin1')
    class_feature = ast.literal_eval(class_feature)
    transformed_dict = {}
    for key in class_feature.keys():
        mapped_x1 = mapping.get(key)
        transformed_dict[mapped_x1] = class_feature[key]
    class_feature=transformed_dict
    prob = pickle.load(open(prob_address, 'rb'), encoding='latin1')
    prob=ast.literal_eval(prob)
    prob_1={}
    for key, value in prob.items():
        #print(key)
        #print(key[0])
        #print(key[1])
        #if key[0] and key[1] in mapping.keys():
        prob_1[(mapping[key[0]],mapping[key[1]])]=value
    prob=prob_1
    attributes=["key2"]
    P = nx.DiGraph()

    #对于Fair Oracle在graph里面要加入node_type成分。
    for (u,v) in G.edges():
        #print(u,v)
        P.add_edge(u, v, weight=0.5)
        G[u][v]['weight']=0.5
        #G[u][v]['p']=0.1
    print('nodes:', len(G.nodes()))
    print('edges:', len(G.edges()))
    print('Done with Loading Feature')
    print('Graph build time:', time.time() - start)
    
    simExperiment = simulateOnlineData(G, P, oracle, class_feature,attributes, seed_size, iterations, dataset)
    algorithms = {}
    algorithms['UCB1'] = UCB1Algorithm(G, P, parameter, seed_size, oracle,attributes)
    algorithms['egreedy_0.1'] = eGreedyAlgorithm(attributes,G, seed_size, oracle, 0.1)
    #algorithms['LinUCB'] = N_LinUCBAlgorithm(attributes, G, P, parameter, seed_size, oracle, dimension*dimension, alpha_1, lambda_, feature_dic)
   # algorithms['CUCB'] = CUCB(G, P, parameter, seed_size, oracle, dimension)
    algorithms['IMFB'] = MFAlgorithm(G, P, parameter, seed_size, oracle, dimension,attributes)

    simExperiment.runAlgorithms(algorithms)