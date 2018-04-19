from source import *

import networkx as nx
import matplotlib.pyplot as plt
G = nx.DiGraph()
G.add_node("S",pos=(2,2)), G.add_node(1,pos=(1,1)), G.add_node(2,pos=(1,0)),G.add_node(3,pos=(0,1)),G.add_node(4,pos=(0,0))
pos1=nx.get_node_attributes(G,'pos')
G.add_edges_from([("S", 1),(1, 2), (1, 3),(2, 4)])
color_map_G = ['red','blue','blue','blue','blue']

F = nx.DiGraph()
F.add_node("S",pos=(2,2)), F.add_node(3,pos=(1,1)), F.add_node(4,pos=(1,0)),F.add_node(2,pos=(0,1)),F.add_node(1,pos=(0,0))
pos2=nx.get_node_attributes(F,'pos')
F.add_edges_from([("S", 3),(3, 4), (4, 2),(4, 1)])
color_map_F = ['red','blue','blue','blue','blue']

plt.subplot(121)
nx.draw(G,pos1,node_color=color_map_G, with_labels=True,font_weight='bold')
plt.subplot(122)
nx.draw(F,pos2,node_color=color_map_F,with_labels=True,font_weight='bold')
plt.show()


N = 5  # number of receivers
Capacity = 20
inf = float('inf')
# initialization of nodes
server = Node('s', Capacity, inf, None, 4)
receivers = []
for i in range(N):
    receivers.append(Node('r', Capacity, inf, i, i + 1))  # initialize the receivers
graph_1 = Graph(1, server, receivers)
# plot_graph(graph_1, True)  # show full mesh underlay graph
#
# # find the SPT
# t, price = smallest_price_tree1(graph_1)
# plot_tree(t, True)  # plot
# # it will be 2- hop tree
#
# # when the price of the server is cheapest
# server.price = 0.5
# t, price = smallest_price_tree1(graph_1)
# plot_tree(t, True)  # plot
# # 1-hop tree, Obvious ...... :)

helpers = []
H = 3
for i in range(H):
    helpers.append(Node('h', Capacity, inf, i, i + 1))

graph_2 = Graph(2, server, receivers, helpers)
plot_graph(graph_2, True)

# compute SPT
server.price = 0.5
t2, price2 = smallest_price_tree2(graph_2)
plot_tree(t2,True) # plot
print('Price : ${0}'.format(price2))
## helpers are not in the SPT

#  the price of cheapest helper <  that of receivers:
# but it will not be selected...-> because measure in effective price!
helpers[0].price = 0.3
receivers[0].price = 0.31
t2, price2 = smallest_price_tree2(graph_2)
plot_tree(t2,True) # plot
print('Price : ${0:.3f}'.format(price2))
# 2-hop tree without helpers, node 'H0' is the cheapest but not selected
# because of the efficient price

# when a helper is the cheapest
server.price = 5
receivers[0].price = 1
t2, price2 = smallest_price_tree2(graph_2)
plot_tree(t2,True) # plot
print('Price : ${0}'.format(price2))
# 2-hop tree