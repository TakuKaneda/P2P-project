import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from source import *

# Set Parameters
N = 10  # number of receivers
H = 5  # number of helpers
M = 2  # Degree bound
# M = N + 1  # NO Degree bound here
V = N + H + 1  # number of nodes (+ server) with helper
C = distribution(N)  # Decide the value of capacity for RECEIVERS
Ch = distribution(H)  # Decide the value of capacity for HELPERS
zeta = 0.1  # value for optimality
eps = 1 - 1 / np.sqrt(1 + zeta)  # rate for updating price
delta = (1 + eps) / (((1 + eps) * (N+1)) ** (1 / eps))  # value for the initial price Without helper
delta_h = (1 + eps) / (((1 + eps) * V) ** (1 / eps))  # value for the initial price With helper

# initialization of nodes
server = Node('s', 768, N + H + 1)  # fix the capacity to 768Kbps
receivers = []
helpers = []
for i in range(N):
    receivers.append(Node('r', C[i], N + H + 1, i))  # initialize the receivers
for i in range(H):
    helpers.append(Node('h', Ch[i], N + H + 1, i))  # initialize the helpers
graph_1 = Graph(1, server, receivers)              # Form 1: Full mesh, No degree bound, No helper
graph_2 = Graph(2, server, receivers, helpers)     # Form 2: Full mesh, No degree bound, With helper
graph_3 = Graph(3, server, receivers, None, M)     # Form 3: Full mesh, With degree bound, No helper
graph_4 = Graph(4, server, receivers, helpers, M)  # Form 4: Full mesh, With degree bound, With helper

# initialize the price for all Forms
graph_1.ini_price(delta)
graph_2.ini_price(delta_h)
graph_3.ini_price(delta)
graph_4.ini_price(delta_h)
# solution of multi-casting, only works for Form 1
print('Capacity of server: {0}'.format(server.capacity))
print('Average capacity  : {0}'.format((sum(C) + server.capacity) / N))

###################
###  Algorithm  ###
###################

# output_capacity, num_loop, min_price_tree = primal_dual_single_session(graph_1, eps)
# output_capacity, num_loop, min_price_tree = primal_dual_single_session(graph_2, eps)
# output_capacity, num_loop, min_price_tree = primal_dual_single_session(graph_3, eps)
output_capacity, num_loop, min_price_tree = primal_dual_single_session(graph_4, eps)

plot_tree(min_price_tree)  # plot the min price tree
