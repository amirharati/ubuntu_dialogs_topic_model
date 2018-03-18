"""
Display top 10 topics for doc2vec model
"""
import pickle

fo = open("tmp/vec_cluster.pk", "rb")
VC = pickle.load(fo)
fo.close()

print("TOP TOPICS")
for i in range(10):
    tid = VC.main_topics_id[i]
    print(tid, " --- ", VC.cluster_names[tid])
