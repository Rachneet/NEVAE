import sys
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import glob
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.layout import *
from random import choice
from collections import defaultdict
import os



def draw_graph(G, name, pos):
    pos = graphviz_layout(G)
    nx.draw_networkx(G, node_size=400, pos=pos, with_labels=True)
    # nx.draw_networkx(G, pos=circular_layout(G),with_labels= True)
    plt.axis('off')
    # plt.title(title)
    if not os.path.exists('predicted_graphs'):
        os.makedirs('predicted_graphs')
    plt.savefig("predicted_graphs_community/" + name + '.pdf')
    plt.gcf().clear()


if __name__ == "__main__":

    path = "sample/*"
    # path = "/home/rachneet/PycharmProjects/graph_generation/baselines/graphvae/graphs/"
    for fname in sorted(glob.glob(path)):
        print(fname)
        f = open(fname, 'rb')
        G = nx.read_edgelist(f, nodetype=int)
        # = np.argsort(degree_seq)
        # getlikelihood(G, n, m)
        n = 32
        # len(G.nodes())

        pos = defaultdict()
        row = 0
        col = 0
        for i in range(n):
            pos[i] = (row, col)
            row += 1
            if (i + 1) % 4 == 0:
                col += 1
                row = 0

        name = fname.split('/')[-1].split('.')[0]
        draw_graph(G, name, pos)