from source import *

N = 10 #number of receivers
p = 0.5  # probability of having edges
price = [10,2,4,5,6,7,5,6,8,2,4]
underlay_graph, color_map, node_labels = build_general_graph(N, p, price, 1)

# visualize the neighborhood relationship (underlay Graph)
pos = nx.spring_layout(underlay_graph)
nx.draw(underlay_graph, node_color=color_map, pos=pos, labels=node_labels)
plt.show()

t = nx.algorithms.tree.branchings.minimum_spanning_arborescence(underlay_graph)  # use the function of NetworX
edge_labels = {}
for e in t.edges(data=True):
    edge_labels.update({(e[0], e[1]): '$'+str(e[2]['weight'])})
pos = nx.shell_layout(t)
nx.draw(t, pos=pos,node_color=color_map, labels=node_labels)
nx.draw_networkx_edge_labels(t, pos=pos, edge_labels=edge_labels)
plt.show()