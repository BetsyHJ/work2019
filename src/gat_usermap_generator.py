'''
Created on May 10th by Jin Huang.
Note!!!
The part of the code is copied from 
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import os
import numpy as np
from time import time
import scipy.sparse as sp
import random as rd
import pandas as pd
import multiprocessing
import pickle
from collections import Counter

from gcn_generator import Data_generator

class GAT_usermap_generator(Data_generator):
    def __init__(self, data, adj_type='bi'): # data get from util.py 'class Dataset'
        # ------------ load basic data needed ------------ #
        super(GAT_usermap_generator, self).__init__(data)

        # ------------ deal data ------------ #
        if self.load_relation_dict_u2e(data.path) == False:
            print "generate relation_dict"
            start_time = time()
            self.relation_dict_u2e = self._get_relational_dict('u2e')
            self.save_u2e(self.relation_dict_u2e)
            print "relation_dict_u2e generate done. time:", time() - start_time
        self.relation_dict_i2e = self._get_relational_dict('i2e')
        
        self.relation_dict_u2e = self._relational_dict_id2idx(self.relation_dict_u2e, self.user2idx, self.en2idx)
        self.relation_dict_i2e = self._relational_dict_id2idx(self.relation_dict_i2e, self.head2idx, self.en2idx)
        
        # ------------ prepare the data for gat embedding -------------- #
        self.adj_edge_ue = self._get_relational_adj_edge(self.relation_dict_u2e, self.n_users, self.n_entities)
        self.adj_edge_ie = self._get_relational_adj_edge(self.relation_dict_i2e, self.n_items, self.n_entities)

    def load_relation_dict_u2e(self, path):
        self.relation_dict_savefile = path + '/GCNOut/u2e.pkl'
        if os.path.exists(self.relation_dict_savefile) == True:
            print "load relation_dict from", self.relation_dict_savefile
            f = open(self.relation_dict_savefile)
            self.relation_dict_u2e = pickle.load(f)
            f.close()
            return True
        return False

    # ------------- for generate needed A_in --------------- #
    def _relational_dict_id2idx(self, relation_dict, lnode2idx, rnode2idx):
        relation_dict_new = {}
        for r in relation_dict:
            u_e_tuples = relation_dict[r]
            xid_eid_tuples = []
            for (x, e) in u_e_tuples:
                xid_eid_tuples.append((lnode2idx[x], rnode2idx[e]))
            relation_dict_new[self.rel2idx[r]] = xid_eid_tuples
        return relation_dict_new

    def _get_relational_dict(self, flag='u2e'):
        ## here it is to get u->en according to u->i and i->en, and drop item i
        ## And for each relation, the u will get a special id.
        def tf_idf_score_filter(u_e_count, e_all_count, e_us, n_neighbors=20):
            u2e = []
            for u in u_e_count:
                e_count = u_e_count[u] #{e:count}
                '''
                e_tf_idf = {} # {e: tf-idf} given user u
                for e in e_count:
                    tf = e_count[e] * 1.0 / e_all_count[e]
                    idf = math.log(self.n_users * 1.0 / (len(e_us[e])) + 1.0) # add 1 for smoothing
                    e_tf_idf[e] = tf * idf
                e_tf_idf = sorted(e_tf_idf.items(), key=lambda d: d[1], reverse=True) # from big to small
                '''
                e_tf_idf = sorted(e_count.items(), key=lambda d: d[1], reverse=True)
                for i in range(len(e_tf_idf)):
                    if i >= 100:
                        break
                    e = e_tf_idf[i][0]
                    u2e += [(u, e)] * e_count[e]
            return u2e

        def _get_dict_u2e(train, kg):
            # get {u:[heads]}
            upis = train['ItemId'].groupby(train['UserId'])
            u2hs = dict()
            for u, group in upis:
                pis = group.values # the interacted users
                u2hs[u] = [self.item2en[x] for x in pis]
            
            hpts = kg['tail'].groupby(kg['head'])
            h2ts = dict()
            u_e_count = {} # {u:{e:count}}
            e_all_count = Counter()
            e_us = {}
            for h, group in hpts:
                h2ts[h] = group.values #pts = group.values
            u2e = list()
            for u in u2hs:
                for h in u2hs[u]:
                    if h not in h2ts: # be careful, this is because of the loss attribute
                        continue
                    ts = h2ts[h]
                    e_all_count.update(ts)
                    e_count = Counter(ts) #{item:count}
                    if len(e_count) > 0:
                        u_e_count[u] = e_count
                    for t in ts:
                        e_us.setdefault(t, set())
                        e_us[t].add(u)
                    for t in h2ts[h]:
                        u2e.append((u, t))
            u2e = tf_idf_score_filter(u_e_count, e_all_count, e_us)
            return u2e

        def _get_dict_i2e(kg):
            h2e = list()
            for (h, t, _) in kg.values:
                h2e.append((h, t))
            return h2e

        # ------ user2en graph (u2e) & item2en (i2e) -> gcn input format ------ #
        relation_dict = dict()
        rels = list(self.kg['rel'].unique())
        #start_time = time()
        for r in rels:
            kg = self.kg[self.kg['rel'] == r]
            if flag == 'u2e':
                x2e = _get_dict_u2e(self.train, kg)
            else:
                x2e = _get_dict_i2e(kg)
            relation_dict[r] =x2e
        return relation_dict
    
    def save_u2e(self, relation_dict):
        filename = self.relation_dict_savefile
        fp = open(filename, 'wb')
        pickle.dump(relation_dict, fp, -1)
        fp.close()

    def _get_relational_adj_edge(self, relation_dict, n_lnode, n_rnode):
        rows, cols = [], []
        start_time = time()
        def _np_mat2sp_adj(rid, np_mat):
            n_all = n_lnode * self.n_rels + n_rnode
            
            a_rows = np_mat[:, 0] * self.n_rels + rid
            a_cols = np_mat[:, 1] + n_lnode * self.n_rels

            b_rows = a_cols
            b_cols = a_rows

            row = np.concatenate([a_rows, b_rows], 0)
            col = np.concatenate([a_cols, b_cols], 0)

            return row, col
        
        for rid in relation_dict.keys():
            row, col = _np_mat2sp_adj(rid, np.array(relation_dict[rid]))
            rows.append(row)
            cols.append(col)
        rows = np.concatenate(rows, 0)
        cols = np.concatenate(cols, 0)
        adj_edge = np.array([rows, cols], dtype=np.int32)
        print "convert relation triples into adj mat done. time:", time() - start_time
        return adj_edge