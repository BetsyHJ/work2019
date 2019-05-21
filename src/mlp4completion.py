import numpy as np
import logging, math, os
import pickle
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

import pandas as pd
import copy

import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Lambda, TimeDistributed, Dropout
from keras.layers import Dot, Concatenate, Add, Multiply, Subtract
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
import keras.backend as K
K.set_image_dim_ordering('tf')

from keras_self_attention import SeqSelfAttention
from utils import Dataset

class Ourmodel(object):
    def __init__(self, head_factors, rel_factors, tail_dim, num_neg, learning_rate):
        self.head_factors = head_factors
        self.rel_factors = rel_factors
        #self.tail_factors = tail_factors
        self.input_shape = head_factors.shape[1]
        self.output_shape = tail_dim
        self.initial_learning_rate = learning_rate
        self.num_neg = num_neg
        #self.lambda_bilinear = 1e-4
        self.weight_size = 1
        self.eta_bilinear = 0.001

    def _create_placeholders(self):
        with tf.name_scope('input_data'):
            self.head_input = tf.placeholder(tf.int32, shape=None, name='head_input')
            self.rel_input = tf.placeholder(tf.int32, shape=None, name='rel_input')
            #self.tail_input = tf.placeholder(tf.int32, shape=None, name='tail_input')
            self.tail_out = tf.placeholder(tf.float32, shape=[None, self.num_neg+1, self.output_shape], name='tail_out')
    
    def _create_variables(self):
        with tf.name_scope('embedding'):
            self.head_embeddings = tf.get_variable(dtype=tf.float32, initializer=self.head_factors, name='head_embeddings', trainable=False)
            self.rel_embeddings = tf.get_variable(dtype=tf.float32, initializer=self.rel_factors, name='rel_embeddings', trainable=False)
            #self.tail_embeddings = tf.get_variable(dtype=tf.float32, initializer=self.tail_factors, name='tail_embeddings', trainable=False)
            self.W = tf.Variable(tf.truncated_normal(shape=[self.input_shape, self.output_shape], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.input_shape + self.output_shape))),name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            self.b = tf.Variable(tf.truncated_normal(shape=[1, self.output_shape], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.input_shape + self.output_shape))),\
                name='Bias_for_MLP', dtype=tf.float32, trainable=True)
            #self.user_bias = tf.get_variable('user_bias', shape=(self.n_users, self.n_factors + 1), dtype=tf.float32, trainable=True) 

    def _create_graph(self):
        self.head_emb = tf.nn.embedding_lookup(self.head_embeddings, self.head_input)
        self.rel_emb = tf.nn.embedding_lookup(self.rel_embeddings, self.rel_input)
        self.tail_emb = self.tail_out #tf.nn.embedding_lookup(self.tail_embeddings, self.tail_input) # (b, d')
        
        self.h_r_ = self.head_emb + self.rel_emb # (b, d)
        self.t_ = tf.nn.tanh(tf.matmul(self.h_r_, self.W) + self.b) # (b, d')
        t_ = tf.expand_dims(self.t_, 1) # (b, 1, d')
        self.similarity = tf.reduce_sum(t_ * self.tail_emb, -1) # (b, s+1)
        self.output = self.similarity[:, 0:1] - self.similarity[:, 1:] # (b, s)

    def _create_loss(self):
        with tf.name_scope('loss'):
            #output = tf.reduce_sum(tf.log(tf.sigmoid(self.output)), axis=-1)
            #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            output = tf.reduce_sum(tf.log(tf.sigmoid(self.output)), axis=-1)
            self.loss_l2 = tf.reduce_sum(tf.square(self.W))
            self.loss = -tf.reduce_mean(output) + self.eta_bilinear * self.loss_l2 #+ self.lambda_bilinear * sum(reg_losses)

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

def dict2matrix(vec): #dict
    idx2id = vec.keys()
    id2idx = {idx2id[x]:x for x in range(len(idx2id))}
    matrix = np.array(vec.values(), dtype=np.float32)
    return id2idx, matrix

def neg_sample(ens, pos, num_neg):
    cans = copy.copy(ens)
    if pos in cans:
        cans.remove(pos)
    '''else:
        print pos'''
    return np.random.choice(cans, num_neg)
    
def generate_train_input(head2idx, rel2idx, rel_en_emb, kg, num_neg):
    head_input, rel_input, tail_input = [], [], []
    for (h, t, r) in kg.values:
        en_linevec = rel_en_emb[r]
        ridx = rel2idx[r]
        hidx = head2idx[h]
        cans = list(en_linevec.keys()) # candidate ens in rel r
        negs = list(neg_sample(cans, t, num_neg))
        if t in en_linevec:
            head_input.append(hidx)
            rel_input.append(ridx) 
            tvecs = [en_linevec[t]] + [en_linevec[x] for x in negs] # (s, d')
            tail_input.append(tvecs) # (b, s, d')
    head_input = np.array(head_input, dtype=np.float32)
    rel_input = np.array(rel_input, dtype=np.float32)
    tail_input = np.array(tail_input, dtype=np.float32)
    return head_input, rel_input, tail_input

def generate_test_input(head2idx, rel2idx, rel_en_emb, kg, num_neg):
    head_input, rel_input = [], []
    head_rels = {} # record head connect with rels, format: {head:[rel]}
    for (h, _, r) in kg.values:
        head_rels.setdefault(h, [])
        head_rels[h].append(r)
    heads, rels = list(kg['head'].unique()), list(kg['rel'].unique())
    for h in heads:
        con_rels = head_rels[h]
        hid = head2idx[h]
        for r in rels:
            if r not in con_rels: # complete
                head_input.append(hid)
                rel_input.append(rel2idx[r])
    head_input = np.array(head_input, dtype=np.float32)
    rel_input = np.array(rel_input, dtype=np.float32)
    return head_input, rel_input


def shuffle_data(inputs, shuffle_size):
    order = np.arange(shuffle_size)
    np.random.shuffle(order)
    return [i[order] for i in inputs]  

def training(n_epochs=10, batch_size=256, num_neg=4):
    data = Dataset('../data/ml-1m/')
    item2en, en2item = data.item2en, data.en2item
    rel_vec, en_vec = data.load_transX()
    kg = data.kg[data.kg['head'].isin(item2en)]
    rel_en_emb, tail_dim = data.load_u2a_LINE()
    # filter items from ens as head
    heads = list(kg['head'].unique())
    heads_vec = {x: en_vec[x] for x in heads}
    rels = list(kg['rel'].unique())
    rels_vec = {x: rel_vec[x] for x in rels}

    # dict2matrix: from dict to get idx and 2d-array
    head2idx, head_factors = dict2matrix(heads_vec)
    rel2idx, rel_factors = dict2matrix(rels_vec)
    
    head_input, rel_input, tail_input = generate_train_input(head2idx, rel2idx, rel_en_emb, kg, num_neg)
    head_input, rel_input, tail_input = shuffle_data([head_input, rel_input, tail_input], head_input.shape[0])

    valid_num = int(0.1 * head_input.shape[0])
    valid_set = [head_input[-valid_num:], rel_input[-valid_num:], tail_input[-valid_num:]]
    head_input, rel_input, tail_input = head_input[:-valid_num], rel_input[:-valid_num], tail_input[:-valid_num]
    #valid_set = [head_input, rel_input, tail_input]
    print head_input.shape, rel_input.shape, tail_input.shape
    model = Ourmodel(head_factors, rel_factors, tail_dim, num_neg, 0.01)
    model.build_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver = tf.train.Saver()
        print "initialized"
        print "training"
        eval(sess, model, valid_set)
        for epoch in range(n_epochs):
            losses, lrs, loss_l2s = [], [], []
            start = 0
            while start < head_input.shape[0]:
                feed_dict = dict()
                feed_dict[model.head_input] = head_input[start:(start+batch_size)]
                feed_dict[model.rel_input] = rel_input[start:(start+batch_size)]
                feed_dict[model.tail_out] = tail_input[start:(start+batch_size)]
                feed_dict[model.global_step] = epoch
                _, loss, lr, loss_l2 = sess.run([model.optimizer, model.loss, model.learning_rate, model.loss_l2], feed_dict)
                start += batch_size
            losses.append(loss)
            lrs.append(lr)
            loss_l2s.append(loss_l2)
            print "epoch", epoch, ", loss", np.array(losses).mean(), np.array(lrs).mean(), np.array(loss_l2s).mean()
            eval(sess, model, valid_set)
    
        # to generate the testset for en loss completion
        print "generating loss completion"
        head_input, rel_input = generate_test_input(head2idx, rel2idx, rel_en_emb, kg, num_neg)
        print "the shape of completion data is", head_input.shape
        test(sess, data, model, [head_input, rel_input], head2idx, rel2idx)

def idx2id_transfer(id2idx):
    return pd.Series(data=id2idx.keys(), index=id2idx.values())

def test(sess, data, model, test_set, head2idx, rel2idx, batch_size=256):
    idx2head = idx2id_transfer(head2idx)
    idx2rel = idx2id_transfer(rel2idx)
    head_input, rel_input = test_set
    start = 0
    fp = open(data.path + "completion_byTransE.txt", 'w')
    while start < head_input.shape[0]:
        feed_dict = dict()
        hin, rin = head_input[start:(start+batch_size)], rel_input[start:(start+batch_size)]
        feed_dict[model.head_input] = hin
        feed_dict[model.rel_input] = rin
        tails_pred_vec = sess.run(model.t_, feed_dict=feed_dict) #(b, d')
        start += batch_size
        for i in range(hin.shape[0]):
            h, r = idx2head[hin[i]], idx2rel[rin[i]]
            t_vec = tails_pred_vec[i]
            t_vec = [str(t_vec[x]) for x in range(len(t_vec))]
            # output format: head   rel     completion_vec
            fp.write(h + '\t' + r + '\t' + '\t'.join(t_vec) + '\n')
    fp.close()



def eval(sess, model, valid_set, batch_size=256):
    head_input, rel_input, tail_input = valid_set
    start = 0
    acc = []
    while start < head_input.shape[0]:
        feed_dict = dict()
        feed_dict[model.head_input] = head_input[start:(start+batch_size)]
        feed_dict[model.rel_input] = rel_input[start:(start+batch_size)]
        feed_dict[model.tail_out] = tail_input[start:(start+batch_size)]
        predictions, loss = sess.run([model.similarity, model.loss], feed_dict=feed_dict)
        start += batch_size
        #print predictions.shape
        for p in range(predictions.shape[0]):
            pred = predictions[p]
            if pred[0] == max(pred):
                acc.append(1)
            else:
                acc.append(0)
    print "accuracy is", np.array(acc).mean()
    
if __name__ == "__main__":
    training()    