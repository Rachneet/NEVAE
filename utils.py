import os
import pickle
from numpy import *
import numpy as np
import networkx as nx
import tensorflow as tf
import ast
import queue as Q
from numpy.linalg import svd, qr, norm
import glob

def slerp(p0, p1, t):
    omega = arccos(dot(p0/norm(p0), p1/norm(p1)))
    so = sin(omega)
    #print "Debug", p0, p1, omega, so,  sin((1.0-t)*omega)/so,  sin((1.0-t)*omega)/so *np.array(p0)
    return sin((1.0-t)*omega) / so * np.array(p0) + sin(t*omega)/so * np.array(p1)

def lerp(p0, p1, t):
    return np.add(p0, t * np.subtract(p1,p0))

def degree(A):
    return np.zeros()


def construct_feed_dict(lr,dropout, k, n, d, decay, placeholders):
    # construct feed dictionary
    feed_dict = dict()


    #feed_dict.update({placeholders['features']: features})
    #feed_dict.update({placeholders['adj']: adj})
    feed_dict.update({placeholders['lr']: lr})
    feed_dict.update({placeholders['dropout']: dropout})
    feed_dict.update({placeholders['decay']: decay})
    #feed_dict.update({placeholders['input']:np.zeros([k,n,d])})
    return feed_dict


def get_shape(tensor):
    '''return the shape of tensor as list'''
    return tensor.get_shape().as_list()

def basis(adj, atol=1e-13, rtol=0):
    """Estimate the basis of a matrix.


    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length n will be treated
        as a 2-D with shape (1, n)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    b : ndarray
        The basis of the columnspace of the matrix.

    See also
    --------
    numpy.linalg.matrix_rank
    """

    A = degree(adj) - adj

    A = np.atleast_2d(A)
    s = svd(A, compute_uv=False)
    tol = max(atol, rtol * s[0])
    rank = int((s >= tol).sum())
    q, r = qr(A)
    return q[:rank]

def print_vars(string):
    '''print variables in collection named string'''
    print("Collection name %s"%string)
    print("    "+"\n    ".join(["{} : {}".format(v.name, get_shape(v)) for v in tf.get_collection(string)]))

def get_basis(mat):
    basis = np.zeros(1,1)
    return basis

def create_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# def get_edges(adj):
#     G.edges()
#     return

def pickle_load(path):
    '''Load the picke data from path'''
    with open(path, 'rb') as f:
        loaded_pickle = pickle.load(f)
    return loaded_pickle

def load_embeddings(fname):
    embd = []
    with open(fname) as f:
        for line in f:
            embd.append(ast.literal_eval(line))
    return embd


def pick_connected_component_new(G):
    print('in pick connected component new')
    #print(G.number_of_nodes())
    print(type(G))
    #adj_list = G.adjacency_list()
    #print(adj_list)
    #print(len(adj_list))
    for id,adj in G.adjacency():
        # print('adj:', adj)
        # print('in for of pcc')
        id_min = min(adj)
        # print('id_min: ', id_min)
        if id<id_min and id>=1:
        # if id<id_min and id>=4:
            break
    node_list = list(range(id)) # only include node prior than node "id"
    print(type(node_list))
    G = G.subgraph(node_list)
    G = max(nx.connected_component_subgraphs(G), key=len)
    return G

# load a list of graphs
def load_graph_list(fname,is_real=True):
    # print('in load graph list')
    with open(fname, "rb") as f:
        graph_list = pickle.load(f)
        #print(type(graph_list))
    for i in range(len(graph_list)):
        #print('in for')
        #print(type(graph_list[i]))
        edges_with_selfloops = list(graph_list[i].selfloop_edges())
        # print(len(edges_with_selfloops))

        if len(edges_with_selfloops)>0:
            #print('pass 1')
            graph_list[i].remove_edges_from(edges_with_selfloops)
        if is_real:
            #print('is real')
            graph_list[i] = max(nx.connected_component_subgraphs(graph_list[i]), key=len)
            graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
        else:
            graph_list[i] = pick_connected_component_new(graph_list[i])

    return graph_list

def load_data(filename, num=0, bin_dim=3, mode= "train"):
    path = filename
    adjlist = []
    featurelist = []
    weightlist = []
    weight_binlist = []
    edgelist = []
    hdelist = []
    # for fname in sorted(glob.iglob(path+"/*")):
    if mode == 'train':
        print("In train")
        fname = path + "/GraphRNN_RNN_community2_multi_4_128_train_0.dat"
    elif mode == 'test':
        fname = path + "/GraphRNN_RNN_community2_multi_4_128_test_0.dat"
    g_list = load_graph_list(fname)
    # print(len(g_list))
    for G in g_list:
        # f = open(fname, 'rb')
        # print(graph.nodes())
        # try:
        #
        #     G=nx.read_edgelist(graph, nodetype=int)
        #     print("nodes", G.nodes())
        #
        # except:
        #     # f = open(fname, 'rb')
        #     # lines = f.read()
        #     # linesnew = lines.replace('{', '{\'weight\':').split('\n')
        #     # G = nx.parse_edgelist(linesnew, nodetype=int)
        #     continue

        # f.close()
        n = num
        for i in range(n):
            if i not in G.nodes():
                G.add_node(i)
        degreemat = np.zeros((n,1), dtype=np.float)
        # count = np.zeros(n)

        # for u in G.nodes():
        #     degreemat[int(u)][0] = (G.degree(u) * 1.0) / (n - 1)
        #     count[G.degree(u)] += 1
        # hde = (2 * count[4] + 2 + count[3] - count[1]) / 2
        # hdelist.append(hde)

        for u in G.nodes():
            degreemat[int(u)][0] = (G.degree(u) * 1.0) / (n - 1)
            edgelist.append(G.edges())
        try:
            adjlist.append(np.array(nx.adjacency_matrix(G).todense()))
            featurelist.append(degreemat)
        except:
            continue

    return (adjlist, featurelist, edgelist)


def pickle_save(content, path):
    '''Save the content on the path'''
    with open(path, 'wb') as f:
        pickle.dump(content, f)



#new functions

def get_candidate_edges(n):
    list_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            # list_edges.append((i,j))
            list_edges.append((i, j, 1))
            list_edges.append((i, j, 2))
            list_edges.append((i, j, 3))
    return list_edges

from numpy.linalg import svd, qr, norm


def normalise_h(prob, weight, bin_dim, indicator, edge_mask, indexlist):
    n = len(prob[0])
    temp = np.ones([n, n])
    p_rs = np.exp(np.minimum(np.multiply(prob, edge_mask), 10 * temp))

    temp = np.ones([n, n, bin_dim])
    w_rs = np.exp(np.minimum(weight, 10 * temp))
    combined_problist = []

    problist = []
    for i in indexlist:
        for j in range(i + 1, n):
            problist.append(p_rs[i][j])
            indi = np.multiply(indicator[i], indicator[j])
            denom = sum(np.multiply(w_rs[i][j], indi))
            if denom == 0:
                denom = 1
                del problist[-1]
            w_rs[i][j] = np.multiply(w_rs[i][j], indi) / denom
            combined_problist.extend(p_rs[i][j] * w_rs[i][j])
    problist = np.array(problist)

    return combined_problist / problist.sum()



def get_weighted_edges_connected(indicator, prob, edge_mask, w_edge, n_edges, node_list, degree_mat, start):
    i = 0
    candidate_edges = []
    q = Q.Queue()
    q.put(start)
    p = []
    list_edges = []
    list_nodes = [0]
    n = indicator.shape[0]

    list_edges = get_candidate_edges(n)
    try:
        while i < n_edges:

            p = normalise_h(prob, w_edge, 3, indicator, edge_mask, range(n))

            if np.count_nonzero(p) == 0:
                break
            candidate_edges.extend([list_edges[k] for k in
                                    np.random.choice(range(len(list_edges)), [1], p=p, replace=False)])

            (u, v, w) = candidate_edges[i]
            degree_mat[u] += w
            degree_mat[v] += w

            edge_mask[u][v] = 0
            edge_mask[v][u] = 0

            if (node_list[u] - degree_mat[u]) == 0:
                indicator[u][0] = 0
            if (node_list[u] - degree_mat[u]) <= 1:
                indicator[u][1] = 0
            if (node_list[u] - degree_mat[u]) <= 2:
                indicator[u][2] = 0

            if (node_list[v] - degree_mat[v]) == 0:
                indicator[v][0] = 0
            if (node_list[v] - degree_mat[v]) <= 1:
                indicator[v][1] = 0
            if (node_list[v] - degree_mat[v]) <= 2:
                indicator[v][2] = 0
            i += 1
            # print("Debug candidate nodes", len(candidate_edges))
    except:
        candiadate_edges = []
    return candidate_edges


def load_data_new(filename, num, node_sample, edge_sample, bin_dim=3):
    path = filename + "/*"
    adjlist = []
    featurelist = []
    featurelist1 = []
    weightlist = []
    weight_binlist = []
    edgelist = []
    smiles = []
    filenumber = int(len(glob.glob(path)) * 1.0)
    filenumber = 1

    neg_edgelist = []
    for fname in sorted(glob.glob(path))[:filenumber]:
        f = open(fname, 'r')
        try:
            G = nx.read_edgelist(f, nodetype=int)
        except:
            f = open(fname, 'r')
            lines = f.read()
            linesnew = lines.replace('{', '{\'weight\':').split('\n')
            G = nx.parse_edgelist(linesnew, nodetype=int)

        # if guess_correct_molecules(fname, 'temp.txt', num, 1):
        #     m = Chem.MolFromMol2File('temp.txt')
        #     if m != None:
        #         smiles.append(Chem.MolToSmiles(m))
        #     else:
        #         smiles.append('')
        #         continue

        f.close()
        n = num
        for i in range(n):
            if i not in G.nodes():
                G.add_node(i)

        # We assume there are only 4 types of atoms
        degreemat = np.zeros((n, 4), dtype=np.float)
        count = np.zeros(4)
        degree_1 = []
        # print("Debug degree", G.degree().values())
        # np.zeros(n)
        for u in G.nodes():
            if G.degree(u) == 3 or G.degree(u) >= 5:
                index = 2
            else:
                index = G.degree(u) - 1
            degreemat[int(u)][index] = 1
            degree_1.append(index)
        e = len(G.edges())

        # try:
        weight = np.array(nx.adjacency_matrix(G).todense())
        adj = np.zeros([n, n])
        weight_bin_list = []
        count = 0
        pos_count = 0
        neg_edges = []

        edge_list = get_edge_list_BFS(weight, G, node_sample, edge_sample, "max")
        # print("Debug edge_list", edge_list)
        for edges in edge_list:
            count_pos = 0
            weight_bin = []
            # np.zeros([bin_dim])
            for (i, j) in edges:
                temp = np.zeros([bin_dim])
                temp[weight[i][j] - 1] = 1
                weight_bin.append(temp)
                # weight_bin[pos_count][weight[i][j]-1] = 1
                count_pos += 1
            weight_bin_list.append(weight_bin)

        for i in range(n):
            for j in range(i + 1, n):
                if weight[i][j] > 0:
                    adj[i][j] = 1
                else:
                    neg_edges.append((i, j))
                # count += 1

        adjlist.append(adj)
        weightlist.append(weight)
        weight_binlist.append(weight_bin_list)
        featurelist.append(degreemat)
        featurelist1.append(degree_1)
        edgelist.append(edge_list)
        neg_edgelist.append(neg_edges)

    return (adjlist, weightlist, weight_binlist, featurelist, edgelist, neg_edgelist, featurelist1, smiles)

import collections
from collections import defaultdict


def breadth_first_search(graph, degree, root):
    visited, queue = set(), collections.deque([root])
    # degree = graph.degree()
    # print "Degree", degree
    bfs_nodes = []
    bfs_edges = []
    # print "Graph", graph
    while queue:
        vertex = queue.popleft()
        bfs_nodes.append(vertex)
        if degree[vertex] > 0:
            neighbourlist = np.random.choice(graph[vertex], len(graph[vertex]), replace=False)
            for neighbour in neighbourlist:
                if neighbour not in visited:
                    degree[vertex] -= 1
                    degree[neighbour] -= 1
                    if degree[neighbour] == 0:
                        visited.add(neighbour)
                    if degree[vertex] == 0:
                        visited.add(vertex)
                    bfs_edges.append((vertex, neighbour))
                    if degree[neighbour] > 0:
                        queue.append(neighbour)
    return bfs_nodes, bfs_edges


def get_degree_distribution(deg):
    total_deg = sum(deg.values())
    degree_dist = []
    for i in range(len(deg.keys())):
        degree_dist.append(deg[i] * 1.0 / total_deg)
    return degree_dist


def get_edge_list_BFS(A, G, n, m, choice=None):
    #A = np.array(nx.adjacency_matrix(G).todense())
    adjdict = defaultdict()
    for node in G.nodes():
        adjdict[node] = np.nonzero(A[node])[0].tolist()
    degree_dist = get_degree_distribution(G.degree())
    max_deg = max(degree_dist)
    if choice == 'max':
        candidate_nodes = [x for x in G.nodes() if degree_dist[x] == max_deg]
        #np.argmax(degree_dist, axis=0).tolist()
        nodes = np.random.choice(candidate_nodes, min(n, len(candidate_nodes)), replace=False)
    else:
        nodes = np.random.choice(G.nodes(), n, p=degree_dist, replace=False)
    bfs_edge_list = []
    for node in nodes:
        for i in range(m):
            bfs_n, bfs_e = breadth_first_search(adjdict, G.degree(), node)
            bfs_edge_list.append(bfs_e)
            print("Debug BFS", bfs_e)
    return bfs_edge_list