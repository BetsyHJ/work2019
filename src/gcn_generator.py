'''
Created on May 7th by Jin Huang.
Note!!!
The part of the code is copied from 
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import os
import numpy as np
import math
import pickle
from time import time
import scipy.sparse as sp
import random as rd
import pandas as pd
import multiprocessing
from collections import Counter

class Data_generator(object):
    def __init__(self, data): # data get from util.py 'class Dataset'
        # ------------ load basic data needed ------------ #
        self.train = data.train
        self.kg = data.kg
        self.item2en = data.item2en
        self.test_u_cans = data.load_negative()

        self.users = self.train['UserId'].unique()
        self.items = self.train['ItemId'].unique()
        self.heads = self.kg['head'].unique()
        self.entities = self.kg['tail'].unique()
        self.rels = self.kg['rel'].unique()

        # align the item and linked kg_en, so they can use one idx
        items = list(set(self.items) & set(self.item2en.index.values))
        item2en = self.item2en[items] # filter the items not appended in RS dataset
        heads = list(set(item2en.values) & set(self.heads))
        item2en = item2en[item2en.isin(heads)] # filter the heads whose related items not append in RS dataset
        self.item2en = item2en
        self.items, self.heads = self.item2en.index.values, self.item2en.values # items and heads have aligned
        self.kg = self.kg[self.kg['head'].isin(self.heads) & self.kg['tail'].isin(self.entities)]
        self.train = self.train[self.train['ItemId'].isin(self.items)]

        # ------------ id2idx processing ------------ #

        self.n_users = len(self.users)
        self.user2idx = pd.Series(data=np.arange(self.n_users), index=self.users)

        self.n_items, self.n_heads = len(self.items), len(self.heads)
        assert self.n_items == self.n_heads
        assert self.item2en[self.items[-1]] == self.heads[-1]
        self.item2idx = pd.Series(data=np.arange(self.n_items), index=self.items)
        self.head2idx = pd.Series(data=np.arange(self.n_heads), index=self.heads)

        self.n_entities = len(self.entities)
        self.en2idx = pd.Series(data=np.arange(self.n_entities), index=self.entities)

        self.n_rels = len(self.rels)
        self.rel2idx = pd.Series(data=np.arange(self.n_rels), index=self.rels)

        u_iidxs = {} #u_is = {}
        pos = self.train['ItemId'].groupby(self.train['UserId'])
        for u, group in pos:
            #u_is[u] = list(group.values)
            u_iidxs[u] = [self.item2idx[x] for x in list(group.values) if x in self.items]
        self.u_iidxs = u_iidxs

    # ------------- for generating input data: u i+ i- ... --------------- #
    def generate_rs_test_data(self):
        n_copy = self.num_neg + 1
        u_test_l, u_test_r = {}, {}
        for u in self.test_u_cans:
            gtItem, negs = self.test_u_cans[u]
            gtItem = gtItem[0]
            if gtItem not in self.items:# or data.item2en[gtItem[0]] in special_items:
                continue
            cans = [x for x in negs if x in self.items] + [gtItem]
            uidx = self.user2idx[u]
            cansidx = [self.item2idx[x] for x in cans]
            test_l, test_r = [], []
            for iidx in cansidx:
                test_l.append([uidx])
                test_r.append([iidx] * n_copy)
            test_l = np.array(test_l, dtype=np.int32)
            test_r = np.array(test_r, dtype=np.int32)
            u_test_l[u] = test_l
            u_test_r[u] = test_r
        return u_test_l, u_test_r
            
            
    def generate_rs_data_fast(self, num_neg=4):
        self.num_neg = num_neg
        # ----------- prepare the data in all set ------------- #
        #canidxs = self.item2idx.values
        
        # ----------- deal data in multiple process ------------ #
        train_l, train_r = self._generate_mutiprocess(self.train, num_neg)
        order = np.arange(len(train_l))
        np.random.shuffle(order)
        #train_l, train_r, train_l_Ks, train_r_Ks = train_l[order], train_r[order], train_l_Ks[order], train_r_Ks[order]
        train_l, train_r = train_l[order], train_r[order]
        return train_l, train_r #, train_l_Ks, train_r_Ks
    
    def _generate_mutiprocess(self, train_set, num_neg, num_thread=10):
        # ----------- deal data in multiple process ------------ #
        thread_size = len(train_set) / num_thread
        q = multiprocessing.Queue()
        thread_ps = []
        for thread in range(num_thread):
            if thread == num_thread - 1:
                train_set_thread = train_set[thread*thread_size:]
            else:
                train_set_thread = train_set[thread*thread_size: (thread+1)*thread_size]
            p = multiprocessing.Process(target=self._generate_input_data, args=(train_set_thread, num_neg, q))
            p.start()
            thread_ps.append(p)
        # ----------- merge all the data generate from multiprocessing ----------- #
        train_l, train_r = [], []
        for thread in range(num_thread):
            [l, r] = q.get()
            train_l.append(l)
            train_r.append(r)
            #train_l_Ks.append(l_Ks)
            #train_r_Ks.append(r_Ks)
            #print "get queue data", l.shape, r.shape
        for p in thread_ps:
            p.join()
        train_l = np.concatenate(train_l, axis=0)
        train_r = np.concatenate(train_r, axis=0)
        #train_l_Ks = np.concatenate(train_l_Ks, axis=0)
        #train_r_Ks = np.concatenate(train_r_Ks, axis=0)
        print "after merging", train_l.shape, train_r.shape#, train_l_Ks.shape, train_r_Ks.shape
        return train_l, train_r

    def _generate_input_data(self, batch_data, num_neg, q):
        train_l, train_r = [], []
        #train_l_Ks, train_r_Ks = [], []
        n_cans = self.n_items #len(canidxs)
        for (u, i, _) in batch_data.values:
            if u not in self.users or i not in self.items:
                continue
            uidx, iidx = self.user2idx[u], self.item2idx[i]
            iidx_pos = self.u_iidxs[u]
            negs = []
            while len(negs) < num_neg:
                t_neg = np.random.randint(n_cans) # for canidxs is item2idx.value [0...n_item-1]
                if t_neg not in negs and t_neg not in iidx_pos:
                    negs.append(t_neg)
            train_l.append([uidx])
            train_r.append([iidx] + negs)
            '''
            l_Ks = np.array([uidx], dtype=np.int32) + self.copyK_u #[self.copyK_u + x for x in [uidx]]
            train_l_Ks.append(l_Ks)
            r_Ks = np.expand_dims(np.array(negs+[iidx], dtype=np.int32), axis=1) + self.copyK_i # [self.copyK_i + x for x in negs+[iidx]]
            train_r_Ks.append(r_Ks)
            '''
        train_l = np.array(train_l, dtype=np.int32)
        train_r = np.array(train_r, dtype=np.int32)
        q.put([train_l, train_r])
    

class BPR_generator(Data_generator):
    def __init__(self, data): # data get from util.py 'class Dataset'
        # ------------ load basic data needed ------------ #
        super(BPR_generator, self).__init__(data)
        

class CKE_generator(Data_generator):
    def __init__(self, data): # data get from util.py 'class Dataset'
        # ------------ load basic data needed ------------ #
        super(CKE_generator, self).__init__(data)

    def generate_rs_data_fast(self, num_neg=4):
        # ----------- prepare the data in all set ------------- #
        uidx_iidxs = {}
        pos = self.train['ItemId'].groupby(self.train['UserId'])
        for u, group in pos:
            uidx_iidxs[self.user2idx[u]] = [self.item2idx[x] for x in list(group.values) if x in self.items]
        canidxs = self.item2idx.values
        # ----------- deal data in multiple process ------------ #
        train_users, train_pos_items, train_neg_items = self._generate_mutiprocess(self.train, uidx_iidxs, self.user2idx, self.item2idx, canidxs, num_neg)
        order = np.arange(len(train_users))
        np.random.shuffle(order)
        train_users, train_pos_items, train_neg_items = train_users[order], train_pos_items[order], train_neg_items[order]
        return train_users, train_pos_items, train_neg_items

    def generate_kg_data_fast(self, num_neg=4):
        # ----------- prepare the data in all set ------------- #
        ridx_hidx_eidxs = {}
        ridx_canidxs = {}
        for r in self.rels:
            kg = self.kg[self.kg['rel'] == r]
            pos = kg['tail'].groupby(kg['head'])
            rid = self.rel2idx[r]
            h_ens, canidxs = {}, set()
            for h, group in pos:
                eidxs = [self.en2idx[x] for x in list(group.values)]
                h_ens[self.head2idx[h]] = eidxs
                canidxs = canidxs | set(eidxs)
            ridx_hidx_eidxs[rid] = h_ens
            ridx_canidxs[rid] = list(canidxs)
        # ----------- deal data in multiple process ------------ #
        kg_heads, kg_pos_tails, kg_neg_tails, kg_rels = [], [], [], []
        for r in self.rels:
            rid, kg = self.rel2idx[r], self.kg[self.kg['rel']==r]
            h, t1, t2 = self._generate_mutiprocess(kg, ridx_hidx_eidxs[rid], self.head2idx, self.en2idx, ridx_canidxs[rid], num_neg)
            r = np.array([rid]*len(h), dtype=np.int32)
            kg_heads.append(h)
            kg_pos_tails.append(t1)
            kg_neg_tails.append(t2)
            kg_rels.append(r)

        kg_heads = np.concatenate(kg_heads, axis=0)
        kg_pos_tails = np.concatenate(kg_pos_tails, axis=0)
        kg_neg_tails = np.concatenate(kg_neg_tails, axis=0)
        kg_rels = np.concatenate(kg_rels, axis=0)
        order = np.arange(len(kg_heads))
        np.random.shuffle(order)
        kg_heads, kg_pos_tails, kg_neg_tails, kg_rels = kg_heads[order], kg_pos_tails[order], kg_neg_tails[order], kg_rels[order]
        return kg_heads, kg_pos_tails, kg_neg_tails, kg_rels
    
    def _generate_mutiprocess(self, train_set, n1idx_n2idxs, n12idx, n22idx, canidxs, num_neg, num_thread=10):
        # ----------- deal data in multiple process ------------ #
        thread_size = len(train_set) / num_thread
        q = multiprocessing.Queue()
        thread_ps = []
        for thread in range(num_thread):
            if thread == num_thread - 1:
                train_set_thread = train_set[thread*thread_size:]
            else:
                train_set_thread = train_set[thread*thread_size: (thread+1)*thread_size]
            p = multiprocessing.Process(target=self._generate_input_data, args=(train_set_thread, n1idx_n2idxs, n12idx, n22idx, canidxs, num_neg, q))
            p.start()
            thread_ps.append(p)
        # ----------- merge all the data generate from multiprocessing ----------- #
        train_l, train_r1, train_r2 = [], [], []
        for thread in range(num_thread):
            [l, r1, r2] = q.get()
            train_l.append(l)
            train_r1.append(r1)
            train_r2.append(r2)
            #print "get queue data", l.shape, r.shape
        for p in thread_ps:
            p.join()
        train_l = np.concatenate(train_l, axis=0)
        train_r1 = np.concatenate(train_r1, axis=0)
        train_r2 = np.concatenate(train_r2, axis=0)
        print "after merging", train_l.shape, train_r1.shape, train_r2.shape
        return train_l, train_r1, train_r2

    def _generate_input_data(self, batch_data, n1idx_n2idxs, n12idx, n22idx, canidxs, num_neg, q):
        train_l, train_r1, train_r2 = [], [], []
        n_cans = len(canidxs)
        for (n1, n2, _) in batch_data.values:
            if n1 not in n12idx.index or n2 not in n22idx.index:
                continue
            n1idx, n2idx = n12idx[n1], n22idx[n2]
            n2_pos = n1idx_n2idxs[n1idx]
            negs = []
            while len(negs) < num_neg:
                t_neg = canidxs[np.random.randint(n_cans)]
                if t_neg not in negs and t_neg not in n2_pos:
                    negs.append(t_neg)
                    train_l.append(n1idx)
                    train_r1.append(n2idx)
                    train_r2.append(t_neg)
        q.put([train_l, train_r1, train_r2])
    

class GCN_generator(Data_generator):
    def __init__(self, data, adj_type='bi'): # data get from util.py 'class Dataset'
        # ------------ load basic data needed ------------ #
        super(GCN_generator, self).__init__(data)
        self.adj_type= adj_type

        # ------------ deal data ------------ #
        self.relation_dict_savefile = data.path + '/GCNOut/u2e.pkl'
        if os.path.exists(self.relation_dict_savefile) == True:
            print "load relation_dict from", self.relation_dict_savefile
            f = open(self.relation_dict_savefile)
            self.relation_dict = pickle.load(f)
            f.close()
        else:
            print "generate relation_dict"
            self.relation_dict = self._get_relational_dict()

        self.relation_dict = self._relational_dict_id2idx()

        # ------------ prepare the data for gcn embedding -------------- #
        self.adj_list = self._get_relational_adj_list()
        self.lap_list = self._get_relational_lap_list()

    def _relational_dict_id2idx(self):
        relation_dict_new = {}
        for r in self.relation_dict:
            u_e_tuples = self.relation_dict[r]
            uid_eid_tuples = []
            for (u, e) in u_e_tuples:
                if u in self.users and e in self.entities:
                    uid_eid_tuples.append((self.user2idx[u], self.en2idx[e]))
            relation_dict_new[self.rel2idx[r]] = uid_eid_tuples
        return relation_dict_new

    def _get_relational_dict(self):
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

        def _get_dict(train, kg):
            # get {u:[heads]}
            upis = train['ItemId'].groupby(train['UserId'])
            u2hs = dict()
            for u, group in upis:
                pis = group.values # the interacted users
                u2hs[u] = [self.item2en[x] for x in pis]
            
            hpts = kg['tail'].groupby(kg['head'])
            h2ts = dict()
            for h, group in hpts:
                h2ts[h] = group.values #pts = group.values
            u_e_count = {} # {u:{e:count}}
            e_all_count = Counter()
            e_us = {}
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
                        if t not in e_us:
                            e_us[t] = set()
                        e_us[t].add(u)

            u2e = tf_idf_score_filter(u_e_count, e_all_count, e_us)
            return u2e

        relation_dict = dict()
        rels = list(self.kg['rel'].unique())
        start_time = time()
        for r in rels:
            print r, "start"
            st1 = time()
            kg = self.kg[self.kg['rel'] == r]
            u2e = _get_dict(self.train, kg)
            relation_dict[r] = u2e
            print "for relation", r, ", we get (u, e) tuple", len(u2e), time() - st1
        self.save_u2e(relation_dict)
        print "relation_dict generate done. time:", time() - start_time

        return relation_dict

    def save_u2e(self, relation_dict):
        filename = self.relation_dict_savefile
        fp = open(filename, 'wb')
        pickle.dump(relation_dict, fp, -1)
        fp.close()

    def _get_relational_adj_list(self):
        adj_mat_list, adj_r_list = [], []
        start_time = time()
        def _np_mat2sp_adj(rid, np_mat):
            n_all = self.n_users * self.n_rels + self.n_entities
            
            a_rows = np_mat[:, 0] + rid * self.n_users
            a_cols = np_mat[:, 1] + self.n_users * self.n_rels
            a_vals = [1.] * len(a_rows)

            b_rows = a_cols
            b_cols = a_rows
            b_vals = [1.] * len(b_rows)

            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
            b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))
            return a_adj, b_adj
        
        for rid in self.relation_dict.keys():
            K, K_inv = _np_mat2sp_adj(rid, np.array(self.relation_dict[rid]))
            adj_mat_list.append(K)
            #adj_r_list.append(rid)
            adj_mat_list.append(K_inv)
            #adj_r_list.append(self.n_rels + rid)
        print "convert relation triples into adj mat done. time:", time() - start_time
        return adj_mat_list

    def _get_relational_lap_list(self):
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
            lap_list = [_bi_norm_lap(adj) for adj in self.adj_list]
            print '\tgenerate bi-normalized adjacency matrix.'
        else:
            lap_list = [_si_norm_lap(adj) for adj in self.adj_list]
            print '\tgenerate si-normalized adjacency matrix.'
        return lap_list
