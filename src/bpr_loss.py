import numpy as np
import logging, math, os

os.environ["CUDA_VISIBLE_DEVICES"] = '3' 

import pandas as pd

import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Lambda, TimeDistributed, Dropout
from keras.layers import Dot, Concatenate, Add, Multiply, Subtract
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
import keras.backend as K
K.set_image_dim_ordering('tf')

from utils import Dataset

def parse_args():
    print "some input parameter"

class Ourmodel(object):
    def __init__(self, n_items, n_users, embedding, learning_rate):
        self.num_neg = 4
        self.n_items = n_items
        self.n_users = n_users
        self.embedding = embedding
        self.learning_rate = learning_rate
        self.lambda_bilinear = 1e-3

    def _create_placeholders(self):
        with tf.name_scope('input_data'):
            self.user_input = tf.placeholder(tf.int32, shape=[None, ])
            self.item_l = tf.placeholder(tf.int32, shape=[None, ])
            self.item_r = tf.placeholder(tf.int32, shape=[None, ])
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
        self.user_emb = tf.nn.embedding_lookup(self.user_embedding, self.user_input) # (b, d)
        self.item_l_emb = tf.nn.embedding_lookup(self.item_embedding, self.item_l) # (b, d)
        self.item_r_emb = tf.nn.embedding_lookup(self.item_embedding, self.item_r) # (b, d)
        # layers to merge two encoded input
        #self.similarity = self._sim([self.user_emb, self.item_emb])# + bias_i # shape: b*s
        #s1 = tf.reduce_sum(tf.multiply(self.user_emb, self.item_l_emb), axis=-1, keep_dims=False) # (b, )
        #s2 = tf.reduce_sum(tf.multiply(self.user_emb, self.item_r_emb), axis=-1, keep_dims=False) # (b, )
        self.similarity = tf.reduce_sum(tf.multiply(self.user_emb, self.item_l_emb), axis=-1, keep_dims=False)
        self.output = tf.reduce_sum(tf.multiply(self.user_emb, self.item_l_emb - self.item_r_emb), axis=-1, keep_dims=False)
        #self.output = tf.reduce_sum(tf.sigmoid(output[:,1:]), axis=-1, keep_dims=False)

    def _create_loss(self):
        with tf.name_scope('loss'):
            output = tf.log_sigmoid(self.output) # (b, )
            #self.l2_loss = tf.reduce_mean(tf.reduce_sum(self.user_emb * self.user_emb, axis=[1,2]))
            #self.l2_loss = tf.nn.l2_loss(self.user_embedding) + tf.nn.l2_loss(self.item_embedding)
            self.l2_loss = tf.nn.l2_loss(self.user_emb) + tf.nn.l2_loss(self.item_l_emb) + tf.nn.l2_loss(self.item_r_emb)
            #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            self.loss = -tf.reduce_mean(output) + self.lambda_bilinear * self.l2_loss

    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta2=0.98).minimize(self.loss)
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_graph()
        self._create_loss()
        self._create_optimizer()
        logging.info("already build the computing graph...")

def training(n_epochs=40, batch_size=256, resample=True):
    
    data = Dataset('../data/ml-1m/')
    #idx_images, idx2item = data.item2image()
    items = data.train['ItemId'].unique()
    user_negs = data.neg_sample(items)
    users = user_negs.keys()
    user2idx = pd.Series(data=np.arange(len(users)), index=users)
    item2idx = pd.Series(data=np.arange(len(items)), index=items)
    train_u, train_l, train_r = generate_input(data.train, item2idx, user_negs, user2idx)
    n_items, n_users, embedding = len(items), len(users), 32
    model = Ourmodel(n_items, n_users, embedding, 0.01)
    model.build_graph()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print "initialized"
        
        losses = []
        l2_losses = []
        for epoch in range(n_epochs):
            if resample == True:
                #user_negs = data.neg_sample_pop(idx2item, pop, pop_items)
                user_negs = data.neg_sample(items)
                train_u, train_l, train_r = generate_input(data.train, item2idx, user_negs, user2idx)
            start = 0
            while start < train_l.shape[0]:
                feed_dict = dict()
                feed_dict[model.user_input] = train_u[start:(start+batch_size)]
                feed_dict[model.item_l] = train_l[start:(start+batch_size)]
                feed_dict[model.item_r] = train_r[start:(start+batch_size)]
                '''print epoch, start, start+batch_size
                print train_l[start:(start+batch_size)]
                print train_r[start:(start+batch_size)]'''
                _, loss, l2_loss = sess.run([model.optimizer, model.loss, model.l2_loss], feed_dict)
                '''print loss'''
                start += batch_size
                losses.append(loss)
                l2_losses.append(l2_loss)
            print "epoch", epoch, ", loss", np.array(losses).mean(), np.array(l2_losses).mean()
            #eval(sess, model, data, item2idx, user2idx, idx2item, users, False)
            if (epoch+1) % 1 == 0:
                eval(sess, model, data, item2idx, user2idx, items, users)
        eval(sess, model, data, item2idx, user2idx, items, users)

def generate_input(train_set, item2idx, user_negs, user2idx):
    # generate the input data
    train_u, train_l, train_r = [], [], []
    #train_y = [] # all set 0
    #print item2idx[:10]
    for [u, i, _] in train_set.values:
        cansidx = [item2idx[i]] + [ item2idx[x] for x in user_negs[u] ] # [pos, neg1, neg2, ...]
        uidx, iidx = user2idx[u], item2idx[i]
        for cidx in cansidx:
            train_u.append(uidx)
            train_l.append(iidx)
            train_r.append(cidx)
    train_u = np.array(train_u, dtype=np.int32)
    train_l = np.array(train_l, dtype=np.int32)
    train_r = np.array(train_r, dtype=np.int32)
    #train_y = np.array(train_y, dtype=np.float32)
    print train_u.shape, train_l.shape, train_r.shape #, train_y.shape
    order = np.arange(len(train_l))
    np.random.shuffle(order)
    train_u, train_l, train_r = train_u[order], train_l[order], train_r[order]
    return train_u, train_l, train_r #, train_y

def eval(sess, model, data, item2idx, user2idx, items, users):
    hits, ndcgs, mrrs, losses = [],[],[],[]
    #users, items = user2idx.index, item2idx.index
    n_copy = model.num_neg + 1
    user_negs = data.load_negative()
    for u in user_negs:
        gtItem, cans = user_negs[u]
        if gtItem not in items:
            continue
        test_u, test_l, test_r = [], [], []
        cansidx = [item2idx[x] for x in cans]
        uidx = user2idx[u]
        for iidx in cansidx:
            test_u.append(uidx)
            test_l.append(iidx)
            test_r.append(iidx)

        test_u = np.array(test_u, dtype=np.int32)
        test_l = np.array(test_l, dtype=np.int32)
        test_r = np.array(test_r, dtype=np.int32)
        '''print test_l.shape, test_r.shape'''
        feed_dict = dict()
        feed_dict[model.user_input] = test_u
        feed_dict[model.item_l] = test_l
        feed_dict[model.item_r] = test_r
        predictions, _ = sess.run([model.similarity, model.loss], feed_dict=feed_dict)
        assert predictions.shape[0] == len(test_l)
        (hr, ndcg, mrr) = _eval_one_rating(predictions)
        
        hits.append(hr)
        ndcgs.append(ndcg) 
        mrrs.append(mrr) 

    hr = _mean_dict(hits)
    ndcg = _mean_dict(ndcgs)
    mrr = np.array(mrrs).mean()
    #test_loss = np.array(losses).mean()
    
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

def _eval_one_rating(predictions):
    
    neg_predict, pos_predict = predictions[:-1], predictions[-1]
    position = (neg_predict >= pos_predict).sum()

    K = [1, 3, 5, 10, 15, 20]
    hr, ndcg = {}, {}
    for k in K:
        hr[k] = _getHR(position, k)
        ndcg[k] = _getNDCG(position, k)
    mrr = 1.0 / (position+1)
    '''print predictions[:-1]
    print predictions[-1]'''
    return (hr, ndcg, mrr)

def _getHR(location, K):
    if location < K:
        return 1.0
    return 0.0
def _getNDCG(location, K):
    if location < K:
        return math.log(2) / math.log(location+2)
    return 0


if __name__ == "__main__":
    training()    
