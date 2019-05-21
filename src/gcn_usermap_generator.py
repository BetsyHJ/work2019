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

class GCN_usermap_generator(Data_generator):
    def __init__(self, data, adj_type='bi'): # data get from util.py 'class Dataset'
        # ------------ load basic data needed ------------ #
        super(GCN_usermap_generator, self).__init__(data)
        self.adj_type= adj_type

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
        
        # ------------ prepare the data for gcn embedding -------------- #
        self.adj_list_u2e = self._get_relational_adj_list(self.relation_dict_u2e, self.n_users, self.n_entities)
        self.lap_list_u2e = self._get_relational_lap_list(self.adj_list_u2e)

        self.adj_list_i2e = self._get_relational_adj_list(self.relation_dict_i2e, self.n_items, self.n_entities)
        self.lap_list_i2e = self._get_relational_lap_list(self.adj_list_i2e)


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

    def _get_relational_adj_list(self, relation_dict, n_lnode, n_rnode):
        adj_mat_list, adj_r_list = [], []
        start_time = time()
        def _np_mat2sp_adj(rid, np_mat):
            n_all = n_lnode * self.n_rels + n_rnode
            
            a_rows = np_mat[:, 0] * self.n_rels + rid
            a_cols = np_mat[:, 1] + n_lnode * self.n_rels
            a_vals = [1.] * len(a_rows)

            b_rows = a_cols
            b_cols = a_rows
            b_vals = [1.] * len(b_rows)

            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
            b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))
            return a_adj, b_adj

        for rid in relation_dict.keys():
            K, K_inv = _np_mat2sp_adj(rid, np.array(relation_dict[rid]))
            adj_mat_list.append(K)
            #adj_r_list.append(rid)
            adj_mat_list.append(K_inv)
            #adj_r_list.append(self.n_rels + rid)

        print "convert relation triples into adj mat done. time:", time() - start_time
        return adj_mat_list

    def _get_relational_lap_list(self, adj_list):
        def _bi_norm_lap(adj):
            # add diag
            l = np.arange(adj.shape[0], dtype=np.int32)
            v = [1.] * adj.shape[0]
            add_diag = sp.coo_matrix((v, (l, l)), shape=adj.shape)
            adj = adj + add_diag

            rowsum = np.array(adj.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def _si_norm_lap(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.adj_type == 'bi':
            lap_list = [_bi_norm_lap(adj) for adj in adj_list]
            print '\tgenerate bi-normalized adjacency matrix.'
        else:
            lap_list = [_si_norm_lap(adj) for adj in adj_list]
            print '\tgenerate si-normalized adjacency matrix.'
        return lap_list

    def load_item_factors(self, data):
        rel_en_emb, dim = data.load_u2e_LINE()
        item_factors = np.zeros((self.n_items, self.n_rels, dim), dtype=np.float32)
        head_rel_ens = {} # format: {item:{rel:[ens]}}
        for [h, t, r] in self.kg.values:
            if h in self.heads and r in self.rels:
                head_rel_ens.setdefault(h, {})
                head_rel_ens[h].setdefault(self.rel2idx[r], [])
                head_rel_ens[h][self.rel2idx[r]].append(t)

        for h in head_rel_ens:
            factor = []
            rel_ens = head_rel_ens[h]
            for rid in range(self.n_rels):
                if rid not in rel_ens:
                    ens_vec = np.zeros(dim, dtype=np.float32)
                else: # if has related entities, use add/ et. to deal with
                    ens, ens_emb = rel_ens[rid], rel_en_emb[self.rels[rid]] # later: {en:emb}
                    ens_vec = []
                    for en in ens:
                        if en in ens_emb:
                            ens_vec.append(ens_emb[en])
                    if len(ens_vec) == 0:
                        ens_vec = np.zeros(dim, dtype=np.float32)
                    else:
                        ens_vec = np.array(ens_vec, dtype=np.float32).mean(0)
                factor.append(ens_vec)
            #print item_factors.shape, np.array(factor, dtype=np.float32).shape, self.head2idx[h]
            item_factors[self.head2idx[h]] = np.array(factor, dtype=np.float32)
        return item_factors
    
    def load_en_factors(self, data):
        rel_en_emb, dim = data.load_u2a_LINE()
        en_factors = np.zeros((self.n_entities, dim), dtype=np.float32)
        en_vecs = {}
        for r in rel_en_emb:
            en_emb = rel_en_emb[r]
            for en in en_emb:
                en_vecs.setdefault(en, [])
                en_vecs[en].append(en_emb[en])
        for en in en_vecs:
            if en not in self.entities:
                continue
            en_factors[self.en2idx[en]] = np.array(en_vecs[en], dtype=np.float32).mean(0)
        return en_factors



