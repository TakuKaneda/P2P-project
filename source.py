import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
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
            self.id = 's'
        elif type == 'h':
            self.type = 'helper'
            self.id = 'h' + str(id)
        elif type == 'r':
            self.type = 'receiver'
            self.id = 'r' + str(id)
        self.child = child  # pointer to the children node ** list!!

    def __eq__(self, other):
        if self.price == other:
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

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


def plot_graph(graph):
    abstract


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


def smallest_price_tree4(graph):
    """Compute the smallest price tree st:
    -Full Mesh Graph
    -With Helper
    -With Degree Bound"""
# initialize the degree and child of the tree
    graph.ini_degree()
    graph.ini_child()
    graph.set_degree_bound()

    server = graph.server
    receivers = graph.receivers
    helpers = graph.helpers

    R = len(receivers)  # number of receivers
    H = len(helpers)    # number of helpers

    sorted_receivers = sorted(receivers)  # sort in price
    sorted_helpers = sorted(helpers)      # sort in price

    min_receiver = sorted_receivers[0]  # the min price receiver
    min_helper = sorted_helpers[0]      # the min price helper

    A = [server, min_receiver]  # server + min price receiver
    B = receivers + helpers - min_receiver    # the rest of receivers and helpers

    server.child.append(min_receiver)  # server -> min price receiver
    server.degree = 1  # degree of server is 1

    mp = 0
    m = min(mp, R)  # !!!! Not sure of what Mp means... !!!

    helpers.price = helpers.price * (1 / (m - 1))  # Effective price for the helpers
    # The effective price of the receiver is equal to the real price --> nothing changes

    while intersect(B, R) != 0:
        for v in A:
            diff = v.degree_bound - v.degree
            n = n + diff
            if n >= len(intersect(B, R)):
                break
            m = min(mp, len(intersect(B, R)) - n)
            helpers.price = helpers.price * (1 / (m - 1))  # Update of the effective price for the helpers
            min_receiver = sorted_receivers[0]             # Update of the min price receiver
            if min_receiver <= min_helper:
                u = [receivers, min_receiver]
            if min_receiver > min_helper:
                u = [helpers, min_helper]

            if u.price >= server.price:
                break

        A_tilde = []
        for v in A:
            if v.degree < v.degree_bound:  # m(v) < M(V)
                A_tilde.append(v)
        min_node = min(A_tilde)
        min_node.price = min_node.price + 1
        A = A + u
        B = B - u

    tree_price = server.price  # total price of the tree
    internal = [server]  # set of internal nodes
    B = intersect(B, R)
    while B != 0:
        A_tilde = []
        for v in A:
            if v.degree < v.degree_bound:  # m(v) < M(V)
                A_tilde.append(v)
        min_node = min(A_tilde)
        mprime = min(len(B), int(min_node.degree_bound - min_node.degree))
        D = B[:mprime]  # take m_temp smallest price nodes form B
        A = A + D  # add D to set A
        del B[:mprime]  # remove the nodes from B

        min_node.degree = min_node.degree + mprime  # update the degree of the min price node
        min_node.child = min_node.child + D  # test

        # COPY AND PASTE FROM YOUR CODE ... I don't know how to end...
        tree_price += min_node.price * mprime  # update the price of the tree
        if min_node.type != 'server':
            internal.append(min_node)  # add the min price node to the internal node

    # find leaf nodes
    leaves = []
    for v in receivers:
        if len(v.child) == 0:
            leaves.append(v)
    del A, B, R, sorted_receivers, min_receiver, D
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
    while D < 1:  # while the price of the min tree is less than 1
        if graph.form == 1:
            t, tree_price = smallest_price_tree1(graph)  # Form 1
        elif graph.form == 2:
            t, tree_price = smallest_price_tree2(graph)  # Form 2
        elif graph.form == 3:
            t, tree_price = smallest_price_tree3(graph)  # Form 3
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
            print('Loop: {0}, Time: {1:.3f}, Price: {2:.6f}'.format(j, toc - tic, D))
    alpha = max_flow(graph.all_nodes())
    output_capacity = Y / alpha
    print('Number of loop: {0}, Total time: {1:.3f}\n'.format(j, time.time() - tic))
    print('Price of the min tree   : ${0:.3f}'.format(D))
    print('Value of dual objective : ${0:.3f}'.format(dual_obj))
    print('Approx capacity rate    : ${0:.3f}'.format(output_capacity))
    plot_tree(t)  # print the final tree (min price tree)
    return output_capacity, j
