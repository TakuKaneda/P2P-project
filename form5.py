from source import *
import networkx as nx
import matplotlib.pyplot as plt

# parameters
N = 10  # number of receivers
p = 0.5  # probability of having edges
zeta = 0.1  # value for optimality
eps = 1 - 1 / np.sqrt(1 + zeta)  # rate for updating price
delta = (1 + eps) / (((1 + eps) * (N + 1)) ** (1 / eps))  # value for the initial price Without helper

# build an general underlay graph (randomly)
underlay_graph, color_map, node_labels = build_general_graph(N, p, delta, 0)

# visualize the neighborhood relationship (underlay Graph)
pos = nx.spring_layout(underlay_graph)
nx.draw(underlay_graph, node_color=color_map, pos=pos, labels=node_labels, with_labels=True)
plt.show()

### algorithm
output_capacity, num_loop, min_price_tree, edge_labels = primal_dual_form_5(underlay_graph, eps)

# visualization
nx.draw(min_price_tree, pos=pos, node_color=color_map, labels=node_labels)
nx.draw_networkx_edge_labels(min_price_tree, pos=pos, edge_labels=edge_labels)
plt.show()
