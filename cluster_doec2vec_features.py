"""
    CLuster documnet vecotrs to find topics.
"""

import logging
import sys
import pickle
import VecCluster as vc

# num clusters
n_clusters = 100
# number of data point selected randomly for clustering
# we cant train on whole dataset due to limittd compute power.
n_random_data = 1000
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

VC = vc.VecCluster(n_clusters, "tmp/dialogs4.dict")
VC.cluster("tmp/id2feat.txt", n_random_data, "tmp/id2doc.txt")

# save the model
fo = open("tmp/vec_cluster.pk", "wb")
pickle.dump(VC, fo)
fo.close()

