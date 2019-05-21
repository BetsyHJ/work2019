import numpy as np
import logging, math, os
import time
import random
import argparse

import pandas as pd
import multiprocessing

import tensorflow as tf

from utils import Dataset
from evaluate import eval_one_rating
from gcn_generator import GCN_generator

#tf.set_random_seed(517)
def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--pretrain', type=bool, default=False, help='pretrain or not')
    parser.add_argument('--path', type=str, default='../data/music/', help='dataset path')
    parser.add_argument('--gpu', type=str, default='0', help='the gpu id')
    parser.add_argument('--dim', type=int, default=16, help='dimension of embedding')
    print "some input parameter"
    args = parser.parse_args()
    return args

class Ourmodel(object):
    def __init__(self, n_items, n_users, embedding, learning_rate, num_neg, A_in, n_fold=100):
        self.num_neg = num_neg
        self.n_items = n_items
        self.n_users = n_users
        self.embedding = embedding
        self.initial_learning_rate = learning_rate
        self.lambda_bilinear = 1e-5
        self.A_in = A_in
        self.n_fold = n_fold
        self.n_layers = 1 # for hop-1 aggregation
        self.weight_size_list = [self.embedding, self.embedding]

    def _create_placeholders(self):
        with tf.name_scope('input_data'):
            self.user_input = tf.placeholder(tf.int32, shape=[None], name='user_input')
            self.item_input = tf.placeholder(tf.int32, shape=[None], name='item_input')
            self.label = tf.placeholder(tf.float32, shape=[None])
            self.mess_dropout = tf.Variable(0, trainable=False) #tf.placeholder(tf.float32, shape=[1], name = 'mess_dropout')
    
    def _create_variables(self):
        with tf.name_scope('embedding'):
            self.user_embed = tf.Variable(tf.truncated_normal(shape=[self.n_users, self.weight_size_list[0]], mean=0.0, stddev=0.01),name='user_embedding', dtype=tf.float32)
            self.item_embed = tf.Variable(tf.truncated_normal(shape=[self.n_items, self.weight_size_list[0]], mean=0.0, stddev=0.01),name='item_embedding', dtype=tf.float32)

        self.gcn_weights = {}
        for k in range(self.n_layers):
            self.gcn_weights['W_gc_'+str(k)] = tf.Variable(tf.truncated_normal(shape=[self.weight_size_list[k], self.weight_size_list[k+1]], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.weight_size_list[k] + self.weight_size_list[k+1]))),name='W_gc_'+str(k), dtype=tf.float32)
            self.gcn_weights['b_gc_'+str(k)] = tf.Variable(tf.truncated_normal(shape=[1, self.weight_size_list[k+1]], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.weight_size_list[k] + self.weight_size_list[k+1]))),name='b_gc_'+str(k), dtype=tf.float32)
            
            
    def _create_graph(self):
        # get user_map and item_mapsq
        self.user_embedding, self.item_embedding = self._create_gcn_embed()
        
        self.user_emb = tf.nn.embedding_lookup(self.user_embedding, self.user_input) # (b, d)
        self.item_emb = tf.nn.embedding_lookup(self.item_embedding, self.item_input) # (b, d)
        
        # layers to merge two encoded input
        self.inner_product = tf.reduce_sum(self.user_emb*self.item_emb, axis=1) # (b,)
        self.output = -tf.reduce_mean(tf.log_sigmoid(self.label * self.inner_product))

    def _create_gcn_embed(self):
        self.mess_dropout = tf.Variable(0.0, trainable=False) #tf.placeholder(tf.float32, shape=[1], name = 'mess_dropout')
        A = self.A_in
        # Generate a set of adjacency sub-matrix.
        A_fold_hat = self._split_A_hat(A)

        embeddings = tf.concat([self.user_embed, self.item_embed], axis=0)
        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            # A_hat_drop = tf.nn.dropout(A_hat, 1 - self.node_dropout[k], [self.n_users + self.n_items, 1])
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.gcn_weights['W_gc_'+str(k)]) + self.gcn_weights['b_gc_'+str(k)])
            embeddings = tf.nn.dropout(embeddings, 1.0 - self.mess_dropout)

            # normalize the distribution of embeddings.
            norm_embeddings = tf.nn.l2_normalize(embeddings, dim=1)
            all_embeddings += [norm_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        ua_embeddings, ea_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return ua_embeddings, ea_embeddings

    def _split_A_hat(self, X):
        A_fold_hat = []
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _norm(self, embedding):
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), -1, keep_dims=True))
        return embedding / norm

    def _sim(self, input_data):
        i1, i2 = input_data[0], input_data[1] # shape: b*d, b*s*d
        i1 = tf.expand_dims(i1, axis=1) # shape: (b, 1, d)
        #i1 = tf.tile(i1, [1, self.num_neg+1, 1]) # shape: (b, s, d)
        ri12 = i1 * i2 #tf.multiply(i1, i2) # shape: (b, s, d)
        ri12 = tf.reduce_sum(ri12, axis=-1, keep_dims=False) # shape (b, s)
        return ri12

    def _create_loss(self):
        with tf.name_scope('loss'):
            self.l2_loss = tf.nn.l2_loss(self.user_emb) + tf.nn.l2_loss(self.item_emb)
            self.loss = self.output + self.lambda_bilinear * self.l2_loss

    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, global_step=self.global_step, decay_steps=1, decay_rate=0.9)
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta2=0.98).minimize(self.loss)
            #self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
            #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_graph()
        self._create_loss()
        self._create_optimizer()
        logging.info("already build the computing graph...")

def training(n_epochs=20, batch_size=1024, num_neg=4, learning_rate=0.01, resample=True):
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    embedding = int(args.dim)
    # ------------- load data ------------- #
    data = Dataset(args.path)
    data_generator = GCN_generator(data)
    A_in = sum(data_generator.lap_list)
    relation_dict = data_generator.relation_dict
    n_users, n_rels, n_entities = data_generator.n_users, data_generator.n_rels, data_generator.n_entities
    print "the number of users, rels,  entites are", n_users, n_rels, n_entities
    # ------------- prepare the needed data -------------- #
    relation_tuple = []
    for rid in relation_dict:
        relation_tuple += [(rid, uid, eid) for (uid, eid) in relation_dict[rid]]
    random.shuffle(relation_tuple)
    print "all relation tuple", len(relation_tuple)
    # ------------ split the train, valid, and test data ------------- #
    n_tuple = len(relation_tuple)
    n_train, n_valid = int(n_tuple * 0.7), int(n_tuple * 0.9)
    #train_tuple, valid_tuple, test_tuple = relation_tuple[:n_train], relation_tuple[n_train: n_valid], relation_tuple[n_valid:]
    #print "train/valid/test tuple:", len(train_tuple), len(valid_tuple), len(test_tuple)
    train_tuple, valid_tuple = relation_tuple, relation_tuple[-n_valid:]

    # ------------ build the model ------------- #
    model = Ourmodel(n_entities, n_users * n_rels, embedding, learning_rate, num_neg, A_in) # use n_users * n_rels, bacause users have k embedding for k rels
    start_time = time.time()
    train_l, train_r, train_label = generate_input_fast(train_tuple, train_tuple, n_users, n_rels, num_neg)
    print "generating data needed for 1 epoch", time.time() - start_time, 's'
    
    # ------------ train the model -------------- #
    model.build_graph()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        print "initialized"
        
        for epoch in range(n_epochs):
            losses = []
            l2_losses, lrs = [], []
            if resample == True and epoch > 0:
                train_l, train_r, train_label = generate_input_fast(train_tuple, train_tuple, n_users, n_rels, num_neg)
            start = 0
            while start < train_l.shape[0]:
                feed_dict = dict()
                feed_dict[model.user_input] = train_l[start:(start+batch_size)]
                feed_dict[model.item_input] = train_r[start:(start+batch_size)]
                feed_dict[model.label] = train_label[start:(start+batch_size)]
                feed_dict[model.mess_dropout] = 0.1
                feed_dict[model.global_step] = epoch
                _, loss, l2_loss, lr = sess.run([model.optimizer, model.loss, model.l2_loss, model.learning_rate], feed_dict)
                '''print loss'''
                start += batch_size
                losses.append(loss)
                l2_losses.append(l2_loss)
                lrs.append(lr)
                #exit(0)
            print "epoch", epoch, ", loss", np.array(losses).mean(), np.array(l2_losses).mean(), 'lr', np.array(lr).mean()
            #eval(sess, model, data, item2idx, user2idx, idx2item, users, False)
            '''if (epoch+1) % 1 == 0:
                valid_set = generate_input_fast(valid_tuple, train_tuple, n_users, n_rels, num_neg)
                eval(sess, model, valid_set)'''
        #saver.save(sess, "./model/book_gcn2emb.ckpt")

        '''print "test set result:"
        test_set = generate_input_fast(test_tuple, train_tuple, n_users, n_rels, num_neg)        
        eval(sess, model, test_set)'''
        emb_map = model.item_embedding.eval()
        save_embedding(emb_map, data_generator.en2idx, save_file=data.path + '/GCNOut/node_gcn.emb')

def generate_input_fast(relation_tuple, all_tuple, n_users, n_rels, num_neg, num_thread=10):
    r_u_ens, r_ens = {}, {}
    for (rid, uid, eid) in all_tuple:
        r_u_ens.setdefault(rid, {})
        r_u_ens[rid].setdefault(uid, [])
        r_u_ens[rid][uid].append(eid)
        r_ens.setdefault(rid, set())
        r_ens[rid].add(eid)

    # multiprocess to generate input data
    thread_size = len(relation_tuple) / num_thread
    q = multiprocessing.Queue()
    # create multiprocessing
    thread_ps = []
    for thread in range(num_thread):
        if thread == num_thread-1:
            relation_tuple_thread = relation_tuple[thread*thread_size:]
        else:
            relation_tuple_thread = relation_tuple[thread*thread_size: (thread+1)*thread_size]
        p = multiprocessing.Process(target=generate_input, args=(relation_tuple_thread, n_users, n_rels, num_neg, q, r_u_ens, r_ens, thread))
        p.start()
        thread_ps.append(p)

    # get and merge the dataset
    train_l, train_r, train_label = [], [], []
    for thread in range(num_thread):
        [l, r, label] = q.get()
        train_l.append(l)
        train_r.append(r)
        train_label.append(label)
    for p in thread_ps:
        p.join()
    train_l = np.concatenate(train_l, axis=0)
    train_r = np.concatenate(train_r, axis=0)
    train_label = np.concatenate(train_label, axis=0)
    order = np.arange(len(train_l))
    np.random.shuffle(order)
    train_l, train_r, train_label = train_l[order], train_r[order], train_label[order]
    print "after merging", train_l.shape, train_r.shape, train_label.shape
    return train_l, train_r, train_label


def generate_input(relation_tuple, n_users, n_rels, num_neg, q=None, r_u_ens=None, r_ens=None, thread_id=0):
    train_l, train_r, train_label = [], [], []
    # generate the input data
    for (rid, uid, eid) in relation_tuple:
        if uid not in r_u_ens[rid]:
            continue
        uid_new = rid * n_users + uid
        pos_ens = r_u_ens[rid][uid]
        all_ens = list(r_ens[rid])
        n_ens = len(all_ens)
        train_l.append(uid)
        train_r.append(eid)
        train_label.append(1.0)
        negs = []
        while len(negs) < num_neg:
            t_en = all_ens[np.random.randint(n_ens)]
            if t_en not in negs and t_en not in pos_ens:
                negs.append(t_en)
                train_l.append(uid)
                train_r.append(t_en)
                train_label.append(-1.0)
    train_l = np.array(train_l, dtype=np.int32)
    train_r = np.array(train_r, dtype=np.int32)
    train_label = np.array(train_label, dtype=np.float32)
    if q is not None:
        q.put([train_l, train_r, train_label])
        #print "thread over"
        exit(0)
    print "multiple threads error"


def eval(sess, model, valid_set, batch_size=256):
    valid_l, valid_r = valid_set
    start = 0
    acc = []
    while start < valid_l.shape[0]:
        feed_dict = dict()
        feed_dict[model.user_input] = valid_l[start:(start+batch_size)]
        feed_dict[model.item_input] = valid_r[start:(start+batch_size)]
        feed_dict[model.mess_dropout] = 0.0
        predictions, loss = sess.run([model.similarity, model.loss], feed_dict=feed_dict)
        start += batch_size
        for p in range(predictions.shape[0]):
            pred = predictions[p]
            if pred[0] == max(pred):
                acc.append(1)
            else:
                acc.append(0)
    print "accuracy is", np.array(acc).mean()

def save_embedding(emb_map, id2idx, save_file='node_gcn.emb'):
    print "save in file:", save_file
    idx2id = pd.Series(data=id2idx.index, index=id2idx.values)
    fp = open(save_file, 'w')
    for i in range(emb_map.shape[0]):
        vec = emb_map[i]
        fp.write(idx2id[i] + ' ' + ' '.join([str(vec[x]) for x in range(emb_map.shape[1])]) + '\n')
    fp.close()



if __name__ == "__main__":
    training()    
