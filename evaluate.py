import glob
import pickle
import networkx as nx
from random import shuffle
from eval.stats import *
from utils import load_graph_list

# class ArgsEvaluate():
#     def __init__(self):
#         # to-do


def findNearestIdx(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def clean_graphs(graph_real, graph_pred):
    ''' Selecting graphs generated that have the similar sizes.
    It is usually necessary for GraphRNN-S version, but not the full GraphRNN model.
    '''
    shuffle(graph_real)
    shuffle(graph_pred)

    # get length
    real_graph_len = np.array([len(graph_real[i]) for i in range(len(graph_real))])
    pred_graph_len = np.array([len(graph_pred[i]) for i in range(len(graph_pred))])

    # select pred samples
    # The number of nodes are sampled from the similar distribution as the training set
    pred_graph_new = []
    pred_graph_len_new = []
    for value in real_graph_len:
        pred_idx = findNearestIdx(pred_graph_len, value)
        pred_graph_new.append(graph_pred[pred_idx])
        pred_graph_len_new.append(pred_graph_len[pred_idx])

    return graph_real, pred_graph_new


def evaluation_epoch(graph_real, graph_pred, graph_validate, fname_output, is_clean= True):

    with open(fname_output, 'w+') as f:
        f.write(
            'approach,degree_validate,clustering_validate,orbits4_validate,degree_test,clustering_test,orbits4_test\n')

    # will follow the same methodology as graph rnn for consistency purpose

    # split test graphs to train test and validation
    # hold out sets
    #     graph_test_len = len(graph_real)
    #     graph_train = graph_real[0:int(0.8 * graph_test_len)]  # train
    #     graph_validate = graph_real[0:int(0.2 * graph_test_len)]  # validate
    #     graph_test = graph_real[int(0.8 * graph_test_len):]  # test on a hold out test set
        graph_test = graph_real

        # average number of nodes in test graphs
        graph_test_aver = 0
        for graph in graph_test:
            graph_test_aver += graph.number_of_nodes()
        graph_test_aver /= len(graph_test)
        print('test average nodes', graph_test_aver)

        # select graphs of similar sizes from both sets
        if is_clean:
            graph_test, graph_pred = clean_graphs(graph_test, graph_pred)

        else:
            # select same no. of graphs as in test set
            shuffle(graph_pred)
            graph_pred = graph_pred[0:len(graph_test)]

        print('len graph_test', len(graph_test))
        print('len graph_validate', len(graph_validate))
        print('len graph_pred', len(graph_pred))

        graph_pred_aver = 0
        for graph in graph_pred:
            graph_pred_aver += graph.number_of_nodes()
        graph_pred_aver /= len(graph_pred)
        print('pred average nodes', graph_pred_aver)



        # evaluate MMD test
        mmd_degree = degree_stats(graph_test, graph_pred)
        mmd_clustering = clustering_stats(graph_test, graph_pred)
        try:
            mmd_4orbits = orbit_stats_all(graph_test, graph_pred)
        except:
            mmd_4orbits = -1
        # evaluate MMD validate
        mmd_degree_validate = degree_stats(graph_validate, graph_pred)
        mmd_clustering_validate = clustering_stats(graph_validate, graph_pred)
        try:
            mmd_4orbits_validate = orbit_stats_all(graph_validate, graph_pred)
        except:
            mmd_4orbits_validate = -1

        # write results
        f.write("approach_1" + ',' +
                str(mmd_degree_validate) + ',' +
                str(mmd_clustering_validate) + ',' +
                str(mmd_4orbits_validate) + ',' +
                str(mmd_degree) + ',' +
                str(mmd_clustering) + ',' +
                str(mmd_4orbits) + '\n')
        print('degree', mmd_degree, 'clustering', mmd_clustering, 'orbits', mmd_4orbits)


def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)



if __name__ == "__main__":

    # load test graphs to test against
    test_graphs = []
    test_path = "graph/GraphRNN_RNN_community2_multi_4_128_test_0.dat"
    validate_path = "graph/GraphRNN_RNN_community2_multi_4_128_validate_0.dat"
    # test_path = "graph/GraphRNN_RNN_barabasi_small_4_64_test_0.dat"
    test_graphs = load_graph_list(test_path)
    v_graphs = load_graph_list(validate_path)
    # load predicted graphs and add them to a list
    # path = "sample/*"
    # path = "/home/rachneet/PycharmProjects/graph_generation/baselines/graphvae/graphs/"
    path = "graph/nevae_community_pred.dat"


    #for i in range(2):
    # for fname in sorted(glob.glob(path)):
    #     pred_graphs = []
    #     print(fname)
    #     if "community_vae" in fname:
        #     with open(fname,'rb') as f:
        #         graph = nx.read_edgelist(f, nodetype=int)
        #         pred_graphs.append(graph)

    # save_graph_list(pred_graphs,'./graph/nevae_community_pred.dat')
    #         pred_graphs = load_graph_list(path+"community_vae_"+str(i)+".dat", is_real=True)
    #         print("Length of test and pred graphs set: ", len(test_graphs), len(pred_graphs))
    #
    #         fname_output = "output/" + "community_vae_results" + '.csv'
    #         evaluation_epoch(test_graphs, pred_graphs, fname_output)

    # pred_graphs = load_graph_list(path + "community_vae_" + str(i) + ".dat", is_real=True)
    pred_graphs = load_graph_list(path)

    print("Length of test and pred graphs set: ", len(test_graphs), len(pred_graphs))

    fname_output = "output/" + "nevae_community_new_results" + '.csv'
    evaluation_epoch(test_graphs, pred_graphs, v_graphs, fname_output)










