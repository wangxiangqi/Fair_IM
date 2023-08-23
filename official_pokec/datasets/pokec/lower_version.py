import networkx as nx
import pickle

# 从源文件中读取.G文件
with open('pokec_relationship.G', 'rb') as f:
    source_graph = pickle.load(f)

#print(source_graph)
# 使用networkx创建一个新的.G文件
new_graph = nx.Graph()

# 将源图中的节点和边复制到新图中
for line in source_graph.edges():
    #print(line)
    node1, node2 = line[0],line[1]
    print(line[0], line[1])
    new_graph.add_edge(node1, node2)

#print(new_graph[6184])
#print(type(new_graph.edges()))
#print(sorted(new_graph.edges())[-1])
#print(new_graph.has_edge(6184,5918))
# 将新图保存到目标文件中
with open('pokec_relationship.G', 'wb') as f:
    pickle.dump(new_graph, f)