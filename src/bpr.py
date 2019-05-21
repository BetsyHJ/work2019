import numpy as np
import logging, math, os
import time

import pandas as pd
import multiprocessing
import argparse

import tensorflow as tf
from utils import Dataset
from evaluate import eval_one_rating
from gcn_generator import BPR_generator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', type=bool, default=False, help='pretrain or not')
    parser.add_argument('--path', type=str, default='../data/music/', help='dataset path')
    parser.add_argument('--gpu', type=str, default='0', help='the gpu id')
    print "some input parameter"
    args = parser.parse_args()
    return args

class Ourmodel(object):
    def __init__(self, n_items, n_users, embedding, learning_rate, num_neg):
        self.num_neg = num_neg
        self.n_items = n_items
        self.n_users = n_users
        self.embedding = embedding
        self.initial_learning_rate = learning_rate
        self.lambda_bilinear = 1e-5

    def _create_placeholders(self):
        with tf.name_scope('input_data'):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1])
            self.item_input = tf.placeholder(tf.int32, shape=[None, self.num_neg+1])
            #self.y_pred = tf.placeholder(tf.float32, shape=[None, 1])
    
    def _create_variables(self):
        with tf.name_scope('embedding'):
            #self.user_embedding = tf.get_variable('user_embedding', shape=(self.n_users, self.embedding), dtype=tf.float32, trainable=True)
            #self.item_embedding = tf.get_variable('item_embedding', shape=(self.n_items, self.embedding), dtype=tf.float32, trainable=True)
            self.user_embedding = tf.Variable(tf.truncated_normal(shape=[self.n_users, self.embedding], mean=0.0, stddev=0.01),name='user_embedding', dtype=tf.float32)
            self.item_embedding = tf.Variable(tf.truncated_normal(shape=[self.n_items, self.embedding], mean=0.0, stddev=0.01),name='item_embedding', dtype=tf.float32)
            #self.bias = tf.Variable(tf.zeros(self.n_items, 1),name='bias')
            #self.user_bias = tf.get_variable('user_bias', shape=(self.n_users, self.n_factors), dtype=tf.float32, trainable=True)

    def _create_graph(self):
        # get user_map and item_mapsq
        self.user_emb = tf.nn.embedding_lookup(self.user_embedding, self.user_input) # (b, 1, d)
        self.item_emb = tf.nn.embedding_lookup(self.item_embedding, self.item_input) # (b, s, d)
        
        '''if True:
            self.user_emb = self._norm(self.user_emb)
            self.item_emb = self._norm(self.item_emb)'''
        # layers to merge two encoded input
        #self.similarity = self._sim([self.user_emb, self.item_emb])# + bias_i # shape: b*s
        self.similarity = tf.reduce_sum(self.user_emb * self.item_emb, axis=-1, keep_dims=False)
        self.output = self.similarity[:, 0:1] - self.similarity[:, 1:] # (b, s-1)
        #self.output = tf.reduce_sum(tf.sigmoid(output[:,1:]), axis=-1, keep_dims=False)
    
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
            #output = tf.reduce_sum(tf.log(tf.sigmoid(self.output[:,1:])), axis=-1, keep_dims=False) # (b, 1)
            output = tf.reduce_sum(tf.log(tf.sigmoid(self.output)), axis=-1)
            #self.l2_loss = tf.reduce_mean(tf.reduce_sum(self.user_emb * self.user_emb, axis=[1,2]))
            #self.l2_loss = tf.nn.l2_loss(self.user_embedding) + tf.nn.l2_loss(self.item_embedding)
            self.l2_loss = tf.nn.l2_loss(self.user_emb) + tf.nn.l2_loss(self.item_emb)
            #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            self.loss = -tf.reduce_mean(output) + self.lambda_bilinear * self.l2_loss

    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, global_step=self.global_step, decay_steps=1, decay_rate=0.9)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta2=0.98).minimize(self.loss)
            #self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
            #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_graph()
        self._create_loss()
        self._create_optimizer()
        logging.info("already build the computing graph...")

def training(n_epochs=40, batch_size=256, num_neg = 4, learning_rate=0.01, verbose=10, embedding=32, resample=True):
    # --------------- load args() -------------- #
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # ------------- load data ------------- #
    data = Dataset(args.path)
    data_generator = BPR_generator(data)
    n_users, n_items = data_generator.n_users, data_generator.n_items
    print "the number of users, items, rels, entities are", n_users, n_items
    # ------------ build the model ------------- #
    model = Ourmodel(n_items, n_users, embedding, learning_rate, num_neg)
    start_time = time.time()
    train_l, train_r = data_generator.generate_rs_data_fast(num_neg)
    print "generating data needed for 1 epoch", time.time() - start_time, 's'
    u_test_l, u_test_r = data_generator.generate_rs_test_data()

    valid_l, valid_r = train_l[-4096:], train_r[-4096:]
    # ------------ train the model -------------- #
    model.build_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print "initialized"
        valid_eval(sess, model, valid_l, valid_r, batch_size, save=False)
        for epoch in range(n_epochs):
            losses = []
            l2_losses, lrs = [], []
            if resample == True and epoch > 0:
                train_l, train_r = data_generator.generate_rs_data_fast(num_neg)
                valid_l, valid_r = train_l[-4096:], train_r[-4096:]
            start = 0
            while start < train_l.shape[0]:
                feed_dict = dict()
                feed_dict[model.user_input] = train_l[start:(start+batch_size)]
                feed_dict[model.item_input] = train_r[start:(start+batch_size)]
                feed_dict[model.global_step] = epoch
                '''print epoch, start, start+batch_size
                print train_l[start:(start+batch_size)]
                print train_r[start:(start+batch_size)]'''
                _, loss, l2_loss, lr = sess.run([model.optimizer, model.loss, model.l2_loss, model.learning_rate], feed_dict)
                '''print loss'''
                start += batch_size
                losses.append(loss)
                l2_losses.append(l2_loss)
                lrs.append(lr)
                #exit(0)
            print "epoch", epoch, ", loss", np.array(losses).mean(), np.array(l2_losses).mean(), 'lr', np.array(lr).mean()
            #eval(sess, model, data, item2idx, user2idx, idx2item, users, False)
            valid_eval(sess, model, valid_l, valid_r, batch_size, save=False)
            if (epoch+1) % verbose == 0:
                eval(sess, model, u_test_l, u_test_r, save=False)

def valid_eval(sess, model, valid_l, valid_r, batch_size, save=False):
    start = 0
    acc = []
    while start < valid_l.shape[0]:
        feed_dict = dict()
        feed_dict[model.user_input] = valid_l[start:(start+batch_size)]
        feed_dict[model.item_input] = valid_r[start:(start+batch_size)]
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
        feed_dict[model.user_input] = test_l
        feed_dict[model.item_input] = test_r
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
