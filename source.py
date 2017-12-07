import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# import sys
import time


class Node:
    """Class Node refers to a peer of the P2P network"""

    def __init__(self, type, capacity, degree_bound, id=0, price=0, degree=0, flow=0, child=[]):
        self.capacity = capacity  # capacity (parameter)
        self.degree_bound = degree_bound  # degree bound (parameter)
        self.degree = degree  # number of degree
        self.price = price
        self.flow = flow
        if type == 's':
            self.type = 'server'
            self.id = 'S'
        elif type == 'h':
            self.type = 'helper'
            self.id = 'H' + str(id)
        elif type == 'r':
            self.type = 'receiver'
            self.id = 'R' + str(id)
        self.child = child  # pointer to the children node ** list!!

    def __lt__(self, other):
        return self.price < other.price

    def __le__(self, other):
        return self.price <= other.price

    def __gt__(self, other):
        return other.__lt__(self)

    def __ge__(self, other):
        return other.__le__(self)

    def __str__(self):
        return 'Type: ' + self.type + ', ID: ' + str(self.id) + ', Price: ' + str(self.price)


class Tree:
    def __init__(self, internal, leaves=None):
        self.internal = internal  # internal nodes (i.e. use the uplink capacity)
        self.leaves = leaves  # leaf nodes (i.e. not use the uplink capacity)

    def all_nodes(self):
        return self.internal + self.leaves


class Graph:
    """Class Graph refers to the whole graph of the net work"""

    def __init__(self, form, server, receivers, helpers=None, degree_bound=0):
        self.form = form  # form of the graph: possibly 1 to 16
        self.server = server  # a single node
        self.receivers = receivers  # list of receivers
        self.helpers = helpers  # there might not be helpers
        self.set_degree_bound(degree_bound)  # set the degree bound

    def all_nodes(self):
        """return the list of all nodes (regardless of the types)"""
        if self.helpers:
            return [self.server] + self.receivers + self.helpers
        else:
            return [self.server] + self.receivers
    def number_nodes(self):
        """return number of nodes"""
        if self.helpers:
            return 1 + len(self.receivers) + len(self.helpers)
        else:
            return 1 + len(self.receivers)

    def set_degree_bound(self, M):
        """set the degree bound of all nodes of the graph to M"""
        for v in self.all_nodes():
            v.degree_bound = M

    def ini_degree(self):
        """Initialize the degree of all nodes to zero"""
        for v in self.all_nodes():
            v.degree = 0

    def ini_child(self):
        """Initialize the child of all nodes to []"""
        for v in self.all_nodes():
            v.child = []

    def ini_price(self, delta):
        """itinialize the price of the nodes according to the parameter delta"""
        for v in self.all_nodes():
            v.price = delta / v.capacity


def plot_graph(graph):
    """visualize the graph for full mesh topology"""
    G = nx.Graph()
    color_map = []
    node_labels = {}
    to_nodes = graph.all_nodes()
    for v in graph.all_nodes():
        to_nodes.remove(v)
        node_labels[v] = v.id + "-" + str.format('{0}Kbps', v.capacity) + ": $" + str.format('{0:.3f}', v.price)
        if v.type == 'server':
            color_map.append('red')
        if v.type == 'receiver':
            color_map.append('blue')
        if v.type == 'helper':
            color_map.append('green')
        for w in to_nodes:
            G.add_edge(v, w)
    pos = nx.spring_layout(G)
    nx.draw(G, node_color=color_map, pos=pos, labels=node_labels, with_labels=True)
    plt.show()
    return


def plot_tree(tree):
    """visualize the tree with the value of the flow
    assume the underlay graph is Full Mesh"""
    T = nx.DiGraph()
    color_map = []
    node_labels = {}
    for v in tree.all_nodes():
        T.add_node(v.id)  # add all node
        node_labels[v.id] = v.id + "-" + str.format('{0}Kbps', v.capacity) + ": $" + str.format('{0:.3f}', v.price)
        # ": $" + str.format('{0:.2f}', v.price)
        if v.type == 'server':
            color_map.append('red')
        if v.type == 'receiver':
            color_map.append('blue')
        if v.type == 'helper':
            color_map.append('green')
        if len(v.child) != 0:  # if the node has child nodes
            for c in v.child:
                T.add_edge(v.id, c.id)  # add edge between the node and its children
    # generate positions for the nodes
    # pos = nx.spring_layout(T, weight=None)
    pos = nx.spring_layout(T, k=1)
    nx.draw(T, node_color=color_map, pos=pos, labels=node_labels, with_labels=True)
    plt.show()
    return


# distribution of the capacities
dist = [(2.8, 64), (14.3, 128), (4.3, 256), (23.3, 384), (55.3, 768)]
cdf = [.028, .171, .214, 0.447, 1.0]  # cdf


def distribution(N):
    """To decide the capacity of the nodes"""
    rand = np.random.random(N)
    Capacity = [None] * N

    for i in range(N):
        k = 0
        while rand[i] > cdf[k]:
            k += 1
        Capacity[i] = dist[k][1]
    return Capacity


def smallest_price_tree1(graph):
    """Compute the smallest price tree st:
    -Full Mesh Graph
    -No Helpers
    -No Degree Bound"""
    server = graph.server
    receivers = graph.receivers
    sorted_receivers = sorted(receivers)
    min_node = min(server, sorted_receivers[0])
    R = len(receivers)
    # initialize the degree and child of the tree
    graph.ini_degree()
    graph.ini_child()
    # compute the total price
    total_price = server.price + (R - 1) * min_node.price
    if min_node.type == 'server':  # if the cheapest node is server -> 1-hop tree
        server.degree = R
        server.child = receivers
        return Tree([server], receivers), total_price
    elif min_node.type == 'receiver':  # if the cheapest node is receiver -> 2-hop tree
        min_node.degree = R - 1
        min_node.child = sorted_receivers[1:]
        server.degree = 1
        server.child = [min_node]
        return Tree([server, sorted_receivers[0]], sorted_receivers[1:]), total_price


def smallest_price_tree2(graph):
    """Compute the smallest price tree st:
    -Full Mesh Graph
    -With Helpers
    -No Degree Bound"""
    # initialize the degree and child of the tree
    graph.ini_degree()
    graph.ini_child()

    server = graph.server
    receivers = graph.receivers
    helpers = graph.helpers

    R = len(receivers)  # number of receivers
    sorted_receivers = sorted(receivers)  # sort in price
    sorted_helpers = sorted(helpers)  # sort in price
    min_helper = sorted_helpers[0]  # find min price helper
    # for helper in sorted_helpers:
    #     helper.price = helper.price * R / (R - 1)

    # compute the total price
    total_price = server.price + min((R - 1) * server.price, (R - 1) * sorted_receivers[0].price,
                                     R * min_helper.price)
    min_helper.price = min_helper.price * R / (R - 1)  # apply effective price for min helper
    min_node = min(server, sorted_receivers[0], min_helper)  # min price node
    min_helper.price = min_helper.price * (R - 1) / R  # retrieve the actual price
    if min_node.type == 'server':  # if the cheapest node is server -> 1-hop tree
        server.degree = R
        server.child = receivers
        return Tree([server], receivers + helpers), total_price
    elif min_node.type == 'receiver':  # if the cheapest node is receiver -> 2-hop tree
        min_node.degree = R - 1
        min_node.child = sorted_receivers[1:]
        server.degree = 1
        server.child = [min_node]
        return Tree([server, min_node], sorted_receivers[1:] + helpers), total_price
    elif min_node.type == 'helper':  # if the cheapest node is helper -> 2-hop tree
        min_node.degree = R
        min_node.child = receivers
        server.degree = 1
        server.child = [min_node]
        return Tree([server, min_node], receivers + sorted_helpers[1:]), total_price


def smallest_price_tree3(graph):
    """Compute the smallest price tree st:
    -Full Mesh Graph
    -No Helper
    -With Degree Bound"""
    # initialize the degree and child of the tree
    graph.ini_degree()
    graph.ini_child()

    server = graph.server
    receivers = graph.receivers
    R = len(receivers)  # number of receivers
    sorted_receivers = sorted(receivers)  # sort in price
    min_receiver = sorted_receivers[0]  # the min price receiver
    A = [server, min_receiver]  # server + min price receiver
    B = sorted_receivers[1:]  # the rest of receivers
    server.child.append(min_receiver)  # server -> min price receiver
    server.degree = 1  # degree of server is 1
    # # set the degree of all receivers to zero
    # for r in receivers:
    #     r.degree = 0

    tree_price = server.price  # total price of the tree
    internal = [server]  # set of internal nodes
    while len(A) != R + 1:  # while size of A is not equal to the number of nodes
        A_tilde = []
        for v in A:
            if v.degree < v.degree_bound:  # m(v) < M(V)
                A_tilde.append(v)
        min_node = min(A_tilde)
        degree_temp = min(len(B), int(min_node.degree_bound - min_node.degree))
        D = B[:degree_temp]  # take m_temp smallest price nodes form B
        del B[:degree_temp]  # remove the nodes from B
        A = A + D  # add D to set A
        # for v in D:
        #     min_node.child.append(v)
        min_node.degree = min_node.degree + degree_temp  # update the degree of the min price node
        min_node.child = min_node.child + D  # test

        tree_price += min_node.price * degree_temp  # update the price of the tree
        if min_node.type != 'server':
            internal.append(min_node)  # add the min price node to the internal node
    # find leaf nodes
    leaves = []
    for v in receivers:
        if len(v.child) == 0:
            leaves.append(v)
    del A, B, R, sorted_receivers, min_receiver, D
    return Tree(internal, leaves), tree_price


def min_cap_per_degree(nodes):
    """find the possible rate of the saturated node(s)"""
    y = 100000
    for n in nodes:
        z = n.capacity / n.degree
        if z < y:
            min_node = n
            y = z
    return y, min_node


def max_flow(nodes):
    """find the scaling factor (alpha in the paper) to scale the flows to the actual rate"""
    a = 0
    for node in nodes:
        if a < node.flow / node.capacity:
            a = node.flow / node.capacity
    return a


def smallest_price_tree4(graph, Mp):
    """Compute the smallest price tree st:
    -Full Mesh Graph
    -With Helper
    -With Degree Bound"""
    # initialize the degree and child of the tree
    graph.ini_degree()
    graph.ini_child()

    server = graph.server
    receivers = graph.receivers
    helpers = graph.helpers

    #### special settings ####
    graph.set_degree_bound(Mp)  # receivers and helpers have same degree bound Mp
    server.degree_bound = float('inf')  # servers degree bound is infinity
    ##########################

    R = len(receivers)  # number of receivers
    m = min(Mp, R)
    for h in helpers:  # Effective price for the helpers
        h.price = h.price * (m / (m - 1))
    v = min(receivers + helpers)  # find the min node with effective price
    for h in helpers:  # Retrieve the actual price for the helpers
        h.price = h.price * (m - 1) / m

    A = [server, v]
    B = receivers + helpers
    B.remove(v)  # set of receivers and helpers except v

    server.child.append(v)  # server -> min effective price node v
    server.degree = 1  # degree of server is 1
    while intersect(B, receivers) != []:
        n = 0
        for v in A:
            diff = v.degree_bound - v.degree
            n = n + diff
        if n >= len(intersect(B, receivers)):
            break
        m = min(Mp, len(intersect(B, receivers)) - n)
        for h in helpers:  # Update of the effective price for the helpers
            h.price = h.price * (m / (m - 1))
        u = min(B)  # pick the min effective price node in B
        if u.price >= server.price:
            for h in helpers:  # retrieve the actual price for the helpers
                h.price = h.price * ((m - 1) / m)
            break
        A_tilde = []
        for v in A:
            if v.degree < v.degree_bound:  # m(v) < M(V)
                A_tilde.append(v)
        v_a = min(A_tilde)
        for h in helpers:  # retrieve the actual price for the helpers
            h.price = h.price * ((m - 1) / m)

        v_a.child.appned(u)  # v_a -> u
        v_a.degree = v_a.degree + 1  # update the degree
        A.append(u)  # add u to A
        B.remove(u)  # remove u from B

    B = intersect(B, receivers)
    while B:  # while B has an element
        A_tilde = []
        for v in A:
            if v.degree < v.degree_bound:  # m(v) < M(V)
                A_tilde.append(v)
        v_a = min(A_tilde)  # pick min node in A_tilde
        m_prime = min(len(B), v_a.degree_bound - v_a.degree)
        B.sort()  # sort by price
        D = B[:m_prime]  # take m_temp smallest price nodes form B
        A = A + D  # add D to set A
        del B[:m_prime]  # remove the nodes from B

        v_a.child = v_a.child + D  # add D as child of v_a
        v_a.degree = v_a.degree + m_prime  # update the degree of the min price node

    # find internal and leaf nodes
    leaves = []
    internal = [server]
    tree_price = server.price * server.degree  # tree price
    for v in receivers + helpers:
        if len(v.child) == 0:
            leaves.append(v)
        else:
            internal.append(v)
            tree_price = tree_price + v.price * v.degree
    del A, B, R, D
    return Tree(internal, leaves), tree_price


def intersect(a, b):
    return list(set(a) & set(b))


#####################################
###  Algorithm for single session ###
#####################################
def primal_dual_single_session(graph, eps):
    Y = 0  # tracking variable for y_t
    D = 0  # min tree price

    j = 0  # counter of the algorithm
    tic = time.time()
    if graph.form == 4:
        Mp = graph.receivers[0].degree_bound
    while D < 1:  # while the price of the min tree is less than 1
        if graph.form == 1:
            t, tree_price = smallest_price_tree1(graph)  # Form 1
        elif graph.form == 2:
            t, tree_price = smallest_price_tree2(graph)  # Form 2
        elif graph.form == 3:
            t, tree_price = smallest_price_tree3(graph)  # Form 3
        elif graph.form == 4:
            t, tree_price = smallest_price_tree4(graph, Mp)  # Form 4
        y, min_node = min_cap_per_degree(t.internal)  # y is the flow sent on the tree
        for node in t.internal:
            node.flow = node.flow + y * node.degree  # update the flow for internal nodes
        Y = Y + y  # update the whole flow

        # update of the price
        min_tree_price = 0  # total price of the minimum tree
        dual_obj = 0  # dual objective = total price of the minimum price
        for node in graph.all_nodes():
            node.price = node.price * (1 + eps * node.degree * y / node.capacity)
            min_tree_price += node.price * node.degree
            dual_obj += node.price * node.capacity
        D = min_tree_price
        j += 1
        if j % 1000 == 0:
            toc = time.time()
            print('Loop: {0}, Time: {1:.3f}, Min Tree Price: {2:.6f}'.format(j, toc - tic, D))
    alpha = max_flow(graph.all_nodes())
    output_capacity = Y / alpha
    print('Algorithm Finished\nNumber of loop: {0}, Total time: {1:.3f}\n'.format(j, time.time() - tic))
    print('Price of the min tree   : ${0:.3f}'.format(D))
    print('Value of dual objective : ${0:.3f}'.format(dual_obj))
    print('Approx capacity rate    : ${0:.3f}'.format(output_capacity))
    # plot_tree(t)  # plot the final tree (min price tree)
    min_price_tree = t
    return output_capacity, j, min_price_tree


#################################
### Form 5: A General Graph  ####
#################################

def build_general_graph(N, p, delta, check):
    """build a general graph with
    -No helper
    -No degree bound

    Arguments - N: number of receivers
              - p: prob to create edges
              - delta: real delta or list of price
              - check: 0 - for P-D algorithm, initialization wrt delta
                     : 1 - you can set your own price: delta should be the list of price |R| + 1 length

    """
    server = Node('s', 768, N + 1)  # fix the capacity to 768Kbps
    receivers = []
    C = distribution(N)  # Decide the value of capacity for RECEIVERS
    for i in range(N):
        receivers.append(Node('r', C[i], N + 1, i))  # initialize the receivers

    graph_5 = Graph(5, server, receivers)
    if check == 0:  # for primal dual
        graph_5.ini_price(delta)  # initialize price
    elif len(delta) != graph_5.number_nodes():
        return None
    elif check == 1:  # your choice
        server.price = delta[0]
        for i in range(len(receivers)):
            receivers[i].price= delta[i+1]
    else:
        return None

    # build a connected random graph
    random_graph = nx.erdos_renyi_graph(N + 1, p)  # create random graph -> create the graph from this
    while not nx.is_connected(random_graph):  # until find one with all nodes are connected
        random_graph = nx.erdos_renyi_graph(N + 1, p)

    color_map = []  # color map for visualization
    node_labels = {}  # label for visualization
    underlay_graph = nx.DiGraph()  # directed graph
    # graph = nx.Graph()  # if you want non directed graph
    for n in random_graph.nodes:
        if n == 0:  # let it be the server, we ignore edges from receives to server (meaning less)
            for l in random_graph.edges(0):
                ind = l[1]  # destination node form n
                underlay_graph.add_edge(server, receivers[ind - 1], weight=server.price)  # give an initial price
            color_map.append('red')
            node_labels[server] = server.id + "-" + str.format('{0}Kbps', server.capacity) + ": $" + str.format(
                '{0:.3f}', server.price)
        else:  # receivers
            for l in random_graph.edges(n):
                ind = l[1]  # destination node from n
                # give an initial price (both direction)
                underlay_graph.add_edge(receivers[n - 1], receivers[ind - 1], weight=receivers[n - 1].price)
                underlay_graph.add_edge(receivers[ind - 1], receivers[n - 1], weight=receivers[ind - 1].price)
            color_map.append('blue')
            rec = receivers[n - 1]
            node_labels[rec] = rec.id + "-" + str.format('{0}Kbps', rec.capacity) + ": $" + str.format('{0:.3f}',
                                                                                                       rec.price)
    return underlay_graph, color_map, node_labels


def min_cap_per_degree_form5(tree):
    """find the possible rate of the saturated node(s) for form 5"""
    y = 100000000
    inner = []  # inner nodes
    for n in tree.nodes:
        n.degree = tree.out_degree(n)
        if tree.out_degree(n) > 0:
            inner.append(n)
            z = n.capacity / tree.out_degree(n)
            if z < y:
                min_node = n
                y = z

    return y, min_node, inner


def primal_dual_form_5(graph, eps):
    """Primal-Dual algorithm for general graph (form 5)"""
    Y = 0  # tracking variable for y_t
    D = 0  # min tree price
    j = 0  # counter of the algorithm
    tic = time.time()
    while D < 1:  # while the price of the min tree is less than 1
        # t = smallest_price_tree4(graph, Mp)  # Form 4
        t = nx.algorithms.tree.branchings.minimum_spanning_arborescence(graph)  # min spanning arborescence problem
        y, min_node, inner = min_cap_per_degree_form5(t)  # y is the flow sent on the tree
        for node in inner:
            node.flow = node.flow + y * node.degree  # update the flow for internal nodes
        Y = Y + y  # update the whole flow

        # update of the price
        tree_price = 0  # total price of the minimum tree
        dual_obj = 0  # dual objective = total price of the minimum price
        for node in graph.nodes:
            node.price = node.price * (1 + eps * node.degree * y / node.capacity)
            tree_price += node.price * node.degree
            dual_obj += node.price * node.capacity

        for n in graph.nodes:
            for e in graph.edges(n):
                graph.edges[e]['weight'] = graph.edges[e]['weight'] * (1 + eps * n.degree * y / n.capacity)

        D = tree_price
        j += 1
        if j % 1000 == 0:
            toc = time.time()
            print('Loop: {0}, Time: {1:.3f}, Min Tree Price: {2:.6f}'.format(j, toc - tic, D))
    alpha = max_flow(graph.nodes)
    output_capacity = Y / alpha
    print('Algorithm Finished\nNumber of loop: {0}, Total time: {1:.3f}\n'.format(j, time.time() - tic))
    print('Price of the min tree   : ${0:.3f}'.format(D))
    print('Value of dual objective : ${0:.3f}'.format(dual_obj))
    print('Approx capacity rate    : ${0:.3f}'.format(output_capacity))
    # plot_tree(t)  # plot the final tree (min price tree)
    min_price_tree = t
    edge_labels = {}
    for e in min_price_tree.edges(data=True):
        # print(e)
        edge_labels.update({(e[0], e[1]): '$' + str.format('{0:.3f}', e[2]['weight'])})
    return output_capacity, j, min_price_tree, edge_labels
