import numpy as np
import logging, math, os
import pickle

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

from keras_self_attention import SeqSelfAttention
from utils import Dataset

def parse_args():
    print "some input parameter"

class Ourmodel(object):
    def __init__(self, n_items, n_users, n_factors, item_factors, embedding, learning_rate):
        self.num_neg = 4
        self.n_items = n_items
        self.n_users = n_users
        self.n_factors = n_factors
        self.embedding = embedding
        self.item_factors = item_factors
        self.input_shape = (n_factors, self.embedding)
        self.initial_learning_rate = learning_rate
        self.lambda_bilinear = 1e-4
        #self.eta_bilinear = 1e-5

    def _create_placeholders(self):
        with tf.name_scope('input_data'):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1])
            self.item_input = tf.placeholder(tf.int32, shape=[None, self.num_neg+1])
            #self.y_pred = tf.placeholder(tf.float32, shape=[None, 1])
    
    def _create_variables(self):
        with tf.name_scope('embedding'):
            self.user_map_embedding = tf.Variable(tf.truncated_normal(shape=[self.n_users, self.n_factors * self.embedding], mean=0.0, stddev=0.01),name='user_map', dtype=tf.float32)
            self.user_embedding = tf.Variable(tf.truncated_normal(shape=[self.n_users, self.embedding], mean=0.0, stddev=0.01),name='user_emb', dtype=tf.float32)
            self.item_map_embedding = tf.get_variable( dtype=tf.float32, initializer=tf.reshape(self.item_factors, [-1, self.n_factors * self.embedding]), name='item_map', trainable=False)
            self.item_embedding = tf.Variable(tf.truncated_normal(shape=[self.n_items, self.embedding], mean=0.0, stddev=0.01),name='item_emb', dtype=tf.float32)
            #self.bias = tf.Variable(tf.zeros(self.n_items, 1),name='bias')
            #self.user_bias = tf.get_variable('user_bias', shape=(self.n_users, self.n_factors+1), dtype=tf.float32, trainable=True)
            #self.W = tf.Variable(tf.truncated_normal(shape=[self.n_factors*self.embedding, self.n_factors*self.embedding], mean=0.0, stddev=tf.sqrt(tf.div(2.0, 2 * self.n_factors*self.embedding))),name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            #self.b = tf.Variable(tf.truncated_normal(shape=[1, self.n_factors*self.embedding], mean=0.0, stddev=tf.sqrt(tf.div(2.0, 2 * self.n_factors*self.embedding))),name='Bias_for_MLP', dtype=tf.float32, trainable=True)

    def _create_graph_selfAtt(self):
        # get user_map and item_mapsq
        self.user_map = tf.nn.embedding_lookup(self.user_map_embedding, self.user_input) # (b, 1, k, d)
        self.item_map = tf.nn.embedding_lookup(self.item_map_embedding, self.item_input) # (b, s, k, d)
        self.user_emb = tf.nn.embedding_lookup(self.user_embedding, self.user_input)
        self.item_emb = tf.nn.embedding_lookup(self.item_embedding, self.item_input)
        #bias_i = tf.nn.embedding_lookup(self.bias, self.item_input)
        network = SeqSelfAttention(input_shape=self.input_shape, attention_activation='relu')
        encoded_l = network(self.user_map) # shape: b*k*d
        encoded_r = TimeDistributed(network)(self.item_map) # shape: b*s*k*d
        # layers to merge two encoded input
        self.UI_simMap = self._matsim([encoded_l, encoded_r]) # shape: (b, s, k)
        user_w = tf.nn.embedding_lookup(self.user_bias, self.user_input) # (b, k)
        user_w = tf.expand_dims(tf.nn.softmax(user_w), axis=1)
        self.similarity = tf.reduce_sum(tf.multiply(self.UI_simMap, user_w), axis=-1, keep_dims=False) # shape (b, s)
        self.output = self.similarity[:, 0:1] - self.similarity # (b, s)
        
    def _create_graph(self):
        # get user_map and item_mapsq
        self.user_map = tf.nn.embedding_lookup(self.user_map_embedding, self.user_input) # (b, 1, k, d)
        self.item_map = tf.nn.embedding_lookup(self.item_map_embedding, self.item_input) # (b, s, k, d)
        self.user_emb = tf.nn.embedding_lookup(self.user_embedding, self.user_input) # (b, 1, d)
        self.item_emb = tf.nn.embedding_lookup(self.item_embedding, self.user_input) # (b, s, d)

        '''self.UI_simMap = self._matsim([self.user_map, self.item_map]) # shape: (b, s, k)
        user_w = tf.nn.embedding_lookup(self.user_bias, self.user_input) # (b, k)
        user_w = tf.nn.softmax(user_w)
        user_w = tf.expand_dims(user_w, axis=1)
        self.similarity = tf.reduce_sum(self.UI_simMap * user_w, axis=-1, keep_dims=False) # shape (b, s)'''
        self.similarity = tf.reduce_sum(self.user_map * self.item_map, axis=-1)
        self.similarity += tf.reduce_sum(self.user_emb * self.item_emb, axis=-1)
        self.output = self.similarity[:, 0:1] - self.similarity[:, 1:] # (b, s)
        #self.output = tf.reduce_sum(tf.sigmoid(output[:,1:]), axis=-1, keep_dims=False)

    def _matsim(self, input):
        i1, i2 = input[0], input[1] # shape: (b, k, d), (b, s, k, d)
        i1 = tf.expand_dims(i1, axis=1) # shape: (b, 1, k, d)
        #i1 = tf.tile(i1, [1, self.num_neg+1, 1, 1]) # shape: (b, s, k, d)
        #ri12 = tf.multiply(i1, i2) # shape: (b, s, k, d)
        ri12 = tf.reduce_sum(i1 * i2, axis=-1, keep_dims=False) # shape (b, s, k)
        return ri12

    def _create_loss(self):
        with tf.name_scope('loss'):
            output = tf.reduce_sum(tf.log(tf.sigmoid(self.output)), axis=-1)
            #self.l2_loss = tf.nn.l2_loss(self.user_map_embedding) + tf.nn.l2_loss(self.i2)
            self.l2_loss = tf.nn.l2_loss(self.user_map) + tf.nn.l2_loss(self.item_emb) + tf.nn.l2_loss(self.user_emb)
            #self.l2_loss += tf.nn.l2_loss(self.user_emb) + tf.nn.l2_loss(self.item_emb)
            #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            
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

def training(n_epochs=40, batch_size=256, num_neg=4, resample=True, verbose=1):
    
    data = Dataset('../data/ml-1m/')
    idx_images, idx2item = data.item2image()
    idx_images = idx_images[:, -1:, :]
    #pop, pop_items = data.get_pop()
    #user_negs = data.neg_sample_pop(idx2item, pop, pop_items)
    #user_negs = data.neg_sample(idx2item)
    #users = user_negs.keys()
    #print "users are", users[:10]
    #print idx2item[:10]
    users = data.train['UserId'].unique()
    user2idx = pd.Series(data=np.arange(len(users)), index=users)
    item2idx = pd.Series(data=np.arange(len(idx2item)), index=idx2item)
    train_l, train_r = generate_input(data.train, item2idx, user2idx, num_neg)
    n_items, n_factors, embedding = idx_images.shape
    print n_items, len(idx2item)
    print "the shape of images is", idx_images.shape
    model = Ourmodel(n_items, len(users), n_factors, idx_images, embedding, 0.01)
    model.build_graph()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print "initialized"

        #initialize for training batches
        #feed_dict[model.user_input] = train_l
        #feed_dict[model.item_input] = train_r
        
        #eval(sess, model, data, item2idx, user2idx, idx2item, users, False)
        #save(model.user_bias.eval(), 'user_bias.epoch0')
        #save_UI_simMap(sess, model, train_l, train_r, 'train_DT.pkl')

        losses, lrs, l2_losss = [], [], []
        for epoch in range(n_epochs):
            if resample == True:
                #user_negs = data.neg_sample_pop(idx2item, pop, pop_items)
                #user_negs = data.neg_sample(idx2item)
                train_l, train_r = generate_input(data.train, item2idx, user2idx, num_neg)
            start = 0
            while start < train_l.shape[0]:
                feed_dict = dict()
                feed_dict[model.user_input] = train_l[start:(start+batch_size)]
                feed_dict[model.item_input] = train_r[start:(start+batch_size)]
                feed_dict[model.global_step] = epoch
                '''print epoch, start, start+batch_size
                print train_l[start:(start+batch_size)]
                print train_r[start:(start+batch_size)]'''
                _, loss, lr, l2_loss = sess.run([model.optimizer, model.loss, model.learning_rate, model.l2_loss], feed_dict)
                '''print loss'''
                start += batch_size
                losses.append(loss)
                lrs.append(lr)
                l2_losss.append(l2_loss)
            print "epoch", epoch, ", loss", np.array(losses).mean(), np.array(lrs).mean(), np.array(l2_losss).mean()
            #eval(sess, model, data, item2idx, user2idx, idx2item, users, False)
            if (epoch+1) % verbose == 0:
                eval(sess, model, data, item2idx, user2idx, idx2item, users)
                #train_l, train_r = generate_input(data.train, item2idx, user2idx, num_neg)
                #save(model.user_bias.eval(), 'user_bias.epoch' + str(epoch+1))
        #save_UI_simMap(sess, model, train_l, train_r, 'train_DT.pkl')
        #eval(sess, model, data, item2idx, user2idx, idx2item, users, True)
        eval(sess, model, data, item2idx, user2idx, idx2item, users)

def save(vectors, filename):
    fp = open(filename, 'w')
    for i in range(vectors.shape[0]):
        v = [str(round(x, 5)) for x in vectors[i]]
        fp.write("\t".join(v) + '\n')
    fp.close()

def generate_input(train_set, item2idx, user2idx, num_neg):
    # generate the input data
    train_l, train_r = [], []
    pos = train_set['ItemId'].groupby(train_set['UserId'])
    for u, group in pos:
        pis = list(group.values) # the positive items
        negs = list(set(item2idx.index.values) - set(pis))
        for i in pis:
            if i not in item2idx:
                continue
            cans = [i] + list(np.random.choice(negs, num_neg))
            cans2idx = [item2idx[x] for x in cans]
            train_l.append([user2idx[u]])
            train_r.append(cans2idx)
    train_l = np.array(train_l, dtype=np.int32)
    train_r = np.array(train_r, dtype=np.int32)
    #train_y = np.array(train_y, dtype=np.float32)
    print train_l.shape, train_r.shape #, train_y.shape
    order = np.arange(len(train_l))
    np.random.shuffle(order)
    train_l, train_r = train_l[order], train_r[order]
    return train_l, train_r #, train_y

def save_UI_simMap(sess, model, train_l, train_r, filename, batch_size=256):
    UI_simMap = []
    UI_label = []
    start = 0
    while start < train_l.shape[0]:
        feed_dict = dict()
        feed_dict[model.user_input] = train_l[start:(start+batch_size)]
        feed_dict[model.item_input] = train_r[start:(start+batch_size)]
        simMap = sess.run(model.UI_simMap, feed_dict=feed_dict) # (b, s, k)
        shape = simMap.shape  # (b, s, k)
        UI_simMap.append(simMap.reshape((-1, shape[-1]))) #(b*s, k)
        labels = np.zeros(shape[1], dtype=np.float32)
        labels[0] = 1
        labels = np.repeat([labels], shape[0], axis=0) # (b*s,)
        UI_label.append(labels.reshape(-1, 1))
        start += batch_size
        '''if start % (4 * batch_size) == 0:
            print start'''
    #feed_dict = dict()
    #feed_dict[model.user_input] = train_l
    #feed_dict[model.item_input] = train_r
    #UI_simMap = sess.run(model.UI_simMap, feed_dict=feed_dict) # (b, s, k)
    UI_label = np.concatenate(UI_label, axis=0)
    UI_simMap = np.concatenate(UI_simMap, axis=0)
    print UI_simMap.shape, UI_label.shape
    fp = open(filename, 'wb')
    pickle.dump(np.concatenate([UI_label, UI_simMap], axis=-1), fp, -1) #(b*s, k+1)
    '''for b in range(UI_simMap.shape[0]):
        sim_map = UI_simMap[b]
        pos = sim_map[0]
        neg = sim_map[1:]
        fp.write('1\t' + '\t'.join([str(x) for x in pos]) + '\n')
        for s in range(neg.shape[0]):
            fp.write('0\t' + '\t'.join([str(x) for x in neg[s]]) + '\n')'''
    fp.close()

def eval(sess, model, data, item2idx, user2idx, items, users, save=False):
    hits, ndcgs, mrrs, losses = [],[],[],[]
    #users, items = user2idx.index, item2idx.index
    n_copy = model.num_neg + 1
    user_negs = data.load_negative()
    UI_simMap = {}
    for u in user_negs:
        gtItem, cans = user_negs[u]
        if gtItem not in items:
            continue
        test_l, test_r = [], []
        cansidx = [item2idx[x] for x in cans]
        uidx = user2idx[u]
        for iidx in cansidx:
            test_l.append([uidx])
            test_r.append([iidx] * n_copy)
        test_l = np.array(test_l, dtype=np.int32)
        test_r = np.array(test_r, dtype=np.int32)
        feed_dict = dict()
        feed_dict[model.user_input] = test_l
        feed_dict[model.item_input] = test_r
        #predictions, simMap = sess.run([model.similarity, model.UI_simMap], feed_dict=feed_dict)
        predictions = sess.run(model.similarity, feed_dict=feed_dict)
        #UI_simMap.append(simMap[:, 0, :])
        (hr, ndcg, mrr) = _eval_one_rating(predictions[:,0])
        hits.append(hr)
        ndcgs.append(ndcg) 
        mrrs.append(mrr) 
    '''if save == True:
        UI_simMap[u] = simMap[:, 0, :]
        fp = open('test_DT.pkl', 'wb')
        pickle.dump(UI_simMap, fp, -1)
        fp.close()'''
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
