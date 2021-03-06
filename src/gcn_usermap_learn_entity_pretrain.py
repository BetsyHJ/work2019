import numpy as np
import logging, math, os, time
import pickle
import argparse
import copy
import multiprocessing

import pandas as pd

import tensorflow as tf

from utils import Dataset
from evaluate import eval_one_rating
from gcn_usermap_generator import GCN_usermap_generator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', type=bool, default=False, help='pretrain or not')
    parser.add_argument('--path', type=str, default='../data/music/', help='dataset path')
    parser.add_argument('--gpu', type=str, default='0', help='the gpu id')
    print "some input parameter"
    args = parser.parse_args()
    return args

class Ourmodel(object):
    def __init__(self, n_users, n_items, n_entities, n_rels, item_factors, en_factors, embedding, learning_rate, num_neg, A_in_ue, n_fold=100):
        self.num_neg = 4
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_rels = n_rels
        self.embedding = embedding

        self.item_factors = item_factors
        self.en_factors = en_factors
        self.input_shape = (n_rels, self.embedding)

        self.initial_learning_rate = learning_rate
        self.lambda_bilinear = 1e-5 # normal: 1e-5
        print "lambda", self.lambda_bilinear
        self.weight_size = 1
        self.eta_bilinear = 1e-2

        self.n_layers = 1
        self.weight_size_list = [self.embedding, self.embedding]
        self.n_fold = n_fold
        self.A_in_ue = A_in_ue

    def _create_placeholders(self):
        with tf.name_scope('input_data'):
            self.uidx = tf.placeholder(tf.int32, shape=[None, 1], name='uidx')
            self.iidx = tf.placeholder(tf.int32, shape=[None, self.num_neg+1], name='iidx')
            #self.user_input = tf.placeholder(tf.int32, shape=[None, 1, self.n_rels], name='user_input')
            #self.item_input = tf.placeholder(tf.int32, shape=[None, self.num_neg+1, self.n_rels], name='item_input')
            
    def _create_variables(self):
        with tf.name_scope('embedding'):
            # ----------- fix some embedding ------------ #
            self.en_embed = tf.get_variable(dtype=tf.float32, initializer=self.en_factors, name='en_init_embedding', trainable=False) # (n_en, b)
            self.item_map_embedding = tf.get_variable(dtype=tf.float32, initializer=self.item_factors, name='item_map', trainable=False)
            # ----------- gcn initial embedding ---------- #
            self.user_embed = tf.Variable(tf.truncated_normal(shape=[self.n_users * self.n_rels, self.weight_size_list[0]], mean=0.0, stddev=0.01),name='user_embedding', dtype=tf.float32)         
            
            # ----------- mf-bpr initial embedding ---------- #
            self.user_mf_embed = tf.Variable(tf.truncated_normal(shape=[self.n_users, self.embedding], mean=0.0, stddev=0.01),name='user_emb', dtype=tf.float32)
            self.item_mf_embed = tf.Variable(tf.truncated_normal(shape=[self.n_items, self.embedding], mean=0.0, stddev=0.01),name='item_emb', dtype=tf.float32)
            #self.W = tf.Variable(tf.truncated_normal(shape=[self.n_factors+1, self.weight_size], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.weight_size + self.n_factors+1))),name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            #self.b = tf.Variable(tf.truncated_normal(shape=[1, self.weight_size], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.weight_size + self.n_factors+1))),name='Bias_for_MLP', dtype=tf.float32, trainable=True)
            #self.user_bias = tf.get_variable('user_bias', shape=(self.n_users, self.n_factors + 1), dtype=tf.float32, trainable=True)
        #with tf.name_scope('gcn_weight'):
            self.gcn_ue_weights = {}
            for k in range(self.n_layers):
                self.gcn_ue_weights['W_gc_'+str(k)] = tf.Variable(tf.truncated_normal(shape=[self.weight_size_list[k], self.weight_size_list[k+1]], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.weight_size_list[k] + self.weight_size_list[k+1]))),name='W_gc_'+str(k), dtype=tf.float32)
                self.gcn_ue_weights['b_gc_'+str(k)] = tf.Variable(tf.truncated_normal(shape=[1, self.weight_size_list[k+1]], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.weight_size_list[k] + self.weight_size_list[k+1]))),name='b_gc_'+str(k), dtype=tf.float32)
            

    def read_usermap_value(self):
        return self.user_gcn_embeded #.eval()

    def _create_graph(self):
        # ---------- run gcn -------------- #
        user_gcn_embeded, self.en_embedding_ue = self._create_gcn_embed(self.A_in_ue, tf.concat([self.user_embed, self.en_embed], axis=0), self.gcn_ue_weights, [self.n_users * self.n_rels, self.n_entities])
        self.user_gcn_embeded = tf.reshape(user_gcn_embeded, [self.n_users, self.n_rels, self.embedding])
        # get user_map and item_mapsq
        self.user_map = tf.nn.embedding_lookup(self.user_gcn_embeded, self.uidx) # (b, 1, k, d)
        self.item_map = tf.nn.embedding_lookup(self.item_map_embedding, self.iidx) # (b, s, k, d)
        self.user_emb = tf.nn.embedding_lookup(self.user_mf_embed, self.uidx) # (b, 1, d)
        self.item_emb = tf.nn.embedding_lookup(self.item_mf_embed, self.iidx) # (b, s, d)
        
        self.UI_simMap = tf.concat([tf.reduce_sum(self.user_map * self.item_map, axis=-1), \
                            tf.reduce_sum(self.user_emb * self.item_emb, axis=-1, keep_dims=True)], axis=-1) # (b, s, k) (b, s, 1) -> (b, s, k+1)
        self.similarity = tf.reduce_sum(self.UI_simMap, axis=-1) # (b, s)
        '''
        UI_simMap = tf.reshape(self.UI_simMap, [-1, self.n_rels+1])
        dense_out = tf.matmul(UI_simMap, self.W) + self.b #tf.nn.sigmoid(tf.matmul(UI_simMap, self.W) + self.b)
        dense_out = tf.reshape(dense_out, [-1, self.num_neg+1, self.weight_size])
        self.similarity = tf.reduce_sum(dense_out, axis=-1) # (b, s)
        '''
        self.output = self.similarity[:, 0:1] - self.similarity[:, 1:] # (b, s)

    def _create_gcn_embed(self, A, init_embeddings, gcn_weights, n_lr_nodes):
        self.mess_dropout = tf.Variable(0.0, trainable=False) #tf.placeholder(tf.float32, shape=[1], name = 'mess_dropout')
        # A = self.A_in
        # Generate a set of adjacency sub-matrix.
        A_fold_hat = self._split_A_hat(A)

        #embeddings = tf.concat([self.user_embed, self.item_embed], axis=0)
        embeddings = init_embeddings
        all_embeddings = []
        for k in range(0, self.n_layers):
            # A_hat_drop = tf.nn.dropout(A_hat, 1 - self.node_dropout[k], [self.n_users + self.n_items, 1])
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
                
            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, gcn_weights['W_gc_'+str(k)]) + gcn_weights['b_gc_'+str(k)])
            embeddings = tf.nn.dropout(embeddings, 1.0 - self.mess_dropout)

            # normalize the distribution of embeddings.
            norm_embeddings = tf.nn.l2_normalize(embeddings, dim=1)
            all_embeddings += [norm_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        ua_embeddings, ea_embeddings = tf.split(all_embeddings, n_lr_nodes, 0)
        return ua_embeddings, ea_embeddings

    def _split_A_hat(self, X):
        n_node = self.n_users * self.n_rels + self.n_entities
        A_fold_hat = []
        fold_len = n_node // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = n_node
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

    def _create_loss(self):
        with tf.name_scope('loss'):
            output = tf.reduce_sum(tf.log(tf.sigmoid(self.output)), axis=-1)
            self.l2_loss = tf.nn.l2_loss(self.user_map) + tf.nn.l2_loss(self.item_emb) + tf.nn.l2_loss(self.user_emb)
            self.loss = -tf.reduce_mean(output) + self.lambda_bilinear * self.l2_loss # + self.eta_bilinear * tf.reduce_sum(tf.square(self.W)) #+ self.lambda_bilinear * sum(reg_losses)

    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, global_step=self.global_step, decay_steps=1, decay_rate=0.9)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta2=0.98).minimize(self.loss)
            #self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_graph()
        #self._create_graph_selfAtt()
        self._create_loss()
        self._create_optimizer()
        logging.info("already build the computing graph...")

def training(n_epochs=40, batch_size=256, num_neg=4, resample=True, verbose=10, learning_rate=0.01):
    # --------------- load args() -------------- #
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # ------------- load data ------------- #
    data = Dataset(args.path)
    data_generator = GCN_usermap_generator(data)
    A_in_ue = sum(data_generator.lap_list_u2e)
    print "A_in_ue shape:", A_in_ue.shape
    n_users, n_items = data_generator.n_users, data_generator.n_items
    n_rels, n_entities = data_generator.n_rels, data_generator.n_entities
    print "the number of users, items, rels, entities are", n_users, n_items, n_rels, n_entities
    # ------------- load init embedding ------------- #
    item_factors = data_generator.load_item_factors(data)
    en_factors = data_generator.load_en_factors(data)
    embedding = en_factors.shape[-1]
    print "load the init embedding done."

    # ------------ build the model ------------- #        
    model = Ourmodel(n_users, n_items, n_entities, n_rels, item_factors, en_factors, embedding, learning_rate, num_neg, A_in_ue)
    start_time = time.time()
    train_l, train_r = data_generator.generate_rs_data_fast(num_neg)
    print "generating data needed for 1 epoch", time.time() - start_time, 's'
    u_test_l, u_test_r = data_generator.generate_rs_test_data()
    
    valid_l, valid_r = train_l[-4096:], train_r[-4096:]
    #print valid_l.shape, valid_r.shape

    # ------------ train the model -------------- #
    model.build_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #session = tf.Session(config=config, ...)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print "initialized"
        if args.pretrain == True:
            print "load the pretrain model"
            #saver.restore(sess, "./model/usermap.ckpt") 
            #eval(sess, model, data, item2idx, user2idx, idx2item, idx2user, True)
        else:
            for epoch in range(n_epochs):
                if resample == True and epoch > 0:
                    #user_negs = data.neg_sample_pop(idx2item, pop, pop_items)
                    #user_negs = data.neg_sample(idx2item)
                    train_l, train_r = data_generator.generate_rs_data_fast(num_neg)
                    valid_l, valid_r = train_l[-4096:], train_r[-4096:]
                losses, lrs, l2_losss = [], [], []
                start = 0
                while start < train_l.shape[0]:
                    feed_dict = dict()
                    feed_dict[model.uidx] = train_l[start:(start+batch_size)]
                    feed_dict[model.iidx] = train_r[start:(start+batch_size)]
                    #feed_dict[model.user_input] = train_l_Ks[start:(start+batch_size)]
                    feed_dict[model.mess_dropout] = 0.1
                    feed_dict[model.global_step] = epoch
                    _, loss, lr, l2_loss = sess.run([model.optimizer, model.loss, model.learning_rate, model.l2_loss], feed_dict)
                    start += batch_size
                    losses.append(loss)
                    lrs.append(lr)
                    l2_losss.append(l2_loss)
                print "epoch", epoch, ", loss", np.array(losses).mean(), np.array(lrs).mean(), np.array(l2_losss).mean()
                valid_eval(sess, model, valid_l, valid_r, batch_size, save=False)
                if (epoch+1) % verbose == 0:
                    eval(sess, model, u_test_l, u_test_r, save=False)
                    #eval(sess, model, data, item2idx, user2idx, idx2item, idx2user)
                    #save(model.user_bias.eval(), 'user_bias.epoch' + str(epoch+1))
            #saver.save(sess, "./model2/music_GCNEncoder_model.ckpt")

def valid_eval(sess, model, valid_l, valid_r, batch_size, save=False):
    start = 0
    acc = []
    while start < valid_l.shape[0]:
        feed_dict = dict()
        feed_dict[model.uidx] = valid_l[start:(start+batch_size)]
        feed_dict[model.iidx] = valid_r[start:(start+batch_size)]
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

def eval(sess, model, u_test_l, u_test_r, save=False):
    #ps, rs, ndcgs, mrrs, losses = [],[],[],[],[]
    hits, ndcgs, mrrs = [],[],[]
    number = len(u_test_l)
    for u in u_test_l:
        test_l, test_r = u_test_l[u], u_test_r[u]
        feed_dict = dict()
        feed_dict[model.uidx] = test_l
        feed_dict[model.iidx] = test_r
        predictions = sess.run(model.similarity, feed_dict=feed_dict)
        (hr, ndcg, mrr) = eval_one_rating(predictions[:,0]) # get predictions[:,0] because test one user by one user
        hits.append(hr)
        ndcgs.append(ndcg) 
        mrrs.append(mrr) 
        
    hr = _mean_dict(hits)
    ndcg = _mean_dict(ndcgs)
    mrr = np.array(mrrs).mean()
    print "the number of ui pairs in testset is", number
    s = ['HR@'+str(x) for x in hr] + ['NGCD@'+str(x) for x in ndcg] + ['mrr'] #+ ['loss']
    print '\t'.join(s)
    s = [str(round(hr[x], 5)) for x in hr] + [str(round(ndcg[x], 5)) for x in ndcg] + [str(round(mrr, 5))] #+ [str(round(test_loss, 5))]
    print "\t".join(s)

def _mean_dict(hits):
    mean_hits = {}
    for hit in hits: #hit is a dict{k:score}
        for k in hit:
            mean_hits.setdefault(k, [])
            mean_hits[k].append(hit[k])
    r = {}
    for k in mean_hits:
        r[k] = np.array(mean_hits[k]).mean()
    return r




if __name__ == "__main__":
    training()    
