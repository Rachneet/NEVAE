from utils import *
#create_dir, pickle_save, print_vars, load_data, get_shape, proxy
from config import SAVE_DIR, VAEGConfig
from datetime import datetime
from cell import VAEGCell
from model import VAEG

import tensorflow as tf
import numpy as np
import logging
import pickle
import os
import argparse

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

FLAGS = None
placeholders = {
    'dropout': tf.placeholder_with_default(0., shape=()),
    'lr': tf.placeholder_with_default(0., shape=()),
    'decay': tf.placeholder_with_default(0., shape=())
    }
def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # network
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--decay_rate", type=float, default=1.0, help="decay rate")
    parser.add_argument("--dropout_rate", type=float, default=0.00005, help="dropout rate")
    parser.add_argument("--log_every", type=int, default=5, help="write the log in how many iterations")
    parser.add_argument("--sample_file", type=str, default="sample", help="directory to store the sample graphs")

    parser.add_argument("--random_walk", type=int, default=3, help="random walk depth")
    parser.add_argument("--z_dim", type=int, default=7, help="z_dim")

    parser.add_argument("--graph_file", type=str, default="graph",
                        help="The directory where the training graph structure is saved")
    parser.add_argument("--z_dir", type=str, default="z_dir",
                        help="The z values will be stored file to be stored")
    parser.add_argument("--sample", type=bool, default=True, help="True if you want to sample")

    parser.add_argument("--mask_weight", type=bool, default=True, help="True if we want to mask")

    parser.add_argument("--out_dir", type=str, default="model_checkpoint",
                        help="Store log/model files.")
    parser.add_argument("--edges", type=int, default=140, help="Number of edges to sample.")
    
    parser.add_argument("--nodes", type=int, default=34, help="Number of nodes to sample.")
    parser.add_argument("--offset", type=int, default=0, help="offset of sample.")



def create_hparams(flags):
  """Create training hparams."""
  return tf.contrib.training.HParams(
      # Data
      graph_file=flags.graph_file,
      out_dir=flags.out_dir,
      z_dir=flags.z_dir,
      sample_file=flags.sample_file,
      z_dim=flags.z_dim,

      # training
      learning_rate=flags.learning_rate,
      decay_rate=flags.decay_rate,
      dropout_rate=flags.dropout_rate,
      num_epochs=flags.num_epochs,
      random_walk=flags.random_walk,
      log_every=flags.log_every,
      mask_weight=flags.mask_weight,
      
      #sample
      sample=flags.sample,
      edges=flags.edges,
      nodes=flags.nodes,
      offset=flags.offset
      )

if __name__ == '__main__':
    nmt_parser = argparse.ArgumentParser()
    add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()
    hparams = create_hparams(FLAGS)
    
    # loading the data from a file
    adj, features, edges = load_data(hparams.graph_file, hparams.nodes)

    num_nodes = adj[0].shape[0]
    
    #Test code
    #''' interpolation
    
    model2 = VAEG(hparams, placeholders, hparams.nodes, 1, edges)
    # print(hparams.out_dir)
    model2.restore(hparams.out_dir)
    #hparams.sample = True
    
    i = 0
    '''
    # getting embeddings
    sample_1 = model2.getembeddings(hparams, placeholders, adj[i], features[i])
    '''
    '''
    sample_1 = model2.getembeddings(hparams, placeholders, adj[0], features[0]) 
    sample_2 = model2.getembeddings(hparams, placeholders, adj[1], features[1])
    
    while i < 1:
        model2.sample_graph_slerp(hparams, placeholders, i,sample_1, sample_2, "slerp", (i+1)*0.1, num=70)
        model2.sample_graph_slerp(hparams, placeholders, i,sample_1, sample_2, "lerp", (i+1)*0.1, num=70)
        i+=1
    '''
    #''' sampling
    #while i < 100:
    model2.sample_graph(hparams, placeholders, i+hparams.offset, hparams.nodes, hparams.edges)
    i += 1
