# from source import
import networkx as nx
import matplotlib.pyplot as plt

G = {
    'R1': {'S': 1, 'R2': 1, 'R3': 1},
    'R2': {'R1': 2},
    'R3': {'R1': 3, 'R4': 3},
    'R4': {'S': 4, 'R3': 4},
    'S': {'R1': 5, 'R4': 5}
}
root = 'S'
# g = mst(root, G)
# print(g)
#
# nx.draw(g)
# plt.show()
F = nx.Graph()
T = nx.DiGraph()
nodes = ['S', 'R1', 'R2', 'R3', 'R4']
T.add_nodes_from(nodes)

for n in nodes:
    for v in G[n]:  # form v to G[n]
        F.add_edge(n,v)
        # F.add_edge(v,n, weight=G[v][n])
        if v != 'S':  # do not add the edge to the source node
            T.add_edge(n, v, weight=G[n][v])

# edges = [('s',1),('s',4)]
m = nx.algorithms.tree.branchings.minimum_spanning_arborescence(T)  # find the min arborescence spanning tree

random_graph=nx.fast_gnp_random_graph(10,1/2)
# for e in random_graph.edges:
#     print(e)
# nx.draw(random_graph)
# plt.show()
pos = nx.shell_layout(m)
for e in T.edges('S'):
    print(e)
print(T.edges(data=True))
for n in T.nodes:
    for e in T.edges(n):
        T.edges[e]['weight'] = 1
print(T.edges(data=True))
nx.draw(F,pos=pos)
nx.draw_networkx_labels(F, pos=pos)
# nx.draw_networkx_edge_labels(F, pos=pos)
plt.show()


edge_labels = {}
for e in m.edges(data=True):
    edge_labels.update({(e[0], e[1]): '$'+str(e[2]['weight'])})
nx.draw(m, pos=pos)
nx.draw_networkx_labels(m, pos=pos)
nx.draw_networkx_edge_labels(m, pos=pos, edge_labels=edge_labels)
plt.show()
