import numpy as np
import logging, math, os, time
import pickle
import argparse
import multiprocessing


import pandas as pd

import tensorflow as tf

'''from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Lambda, TimeDistributed, Dropout
from keras.layers import Dot, Concatenate, Add, Multiply, Subtract
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
import keras.backend as K
K.set_image_dim_ordering('tf')

from keras_self_attention import SeqSelfAttention'''
from utils import Dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', type=bool, default=False, help='pretrain or not')
    parser.add_argument('--path', type=str, default='../data/music/', help='dataset path')
    parser.add_argument('--gpu', type=str, default='0', help='the gpu id')
    print "some input parameter"
    args = parser.parse_args()
    return args

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
        self.lambda_bilinear = 1e-5
        self.weight_size = 1
        self.eta_bilinear = 0.01

    def _create_placeholders(self):
        with tf.name_scope('input_data'):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1])
            self.item_input = tf.placeholder(tf.int32, shape=[None, self.num_neg+1])
            #self.y_pred = tf.placeholder(tf.float32, shape=[None, 1])
    
    def _create_variables(self):
        with tf.name_scope('embedding'):
            '''self.user_map_embedding = tf.Variable(tf.truncated_normal(shape=[self.n_users, self.n_factors * self.embedding], mean=0.0, stddev=0.01),name='user_map', dtype=tf.float32)
            self.user_embedding = tf.Variable(tf.truncated_normal(shape=[self.n_users, self.embedding], mean=0.0, stddev=0.01),name='user_emb', dtype=tf.float32)
            self.item_map_embedding = tf.get_variable(dtype=tf.float32, initializer=tf.reshape(self.item_factors, [-1, self.n_factors * self.embedding]), name='item_map', trainable=False)
            self.item_embedding = tf.Variable(tf.truncated_normal(shape=[self.n_items, self.embedding], mean=0.0, stddev=0.01),name='item_emb', dtype=tf.float32)'''
            self.user_map_embedding = tf.Variable(tf.truncated_normal(shape=[self.n_users, self.n_factors, self.embedding], mean=0.0, stddev=0.01),name='user_map', dtype=tf.float32)
            self.user_embedding = tf.Variable(tf.truncated_normal(shape=[self.n_users, self.embedding], mean=0.0, stddev=0.01),name='user_emb', dtype=tf.float32)
            self.item_map_embedding = tf.get_variable(dtype=tf.float32, initializer=self.item_factors, name='item_map', trainable=False)
            self.item_embedding = tf.Variable(tf.truncated_normal(shape=[self.n_items, self.embedding], mean=0.0, stddev=0.01),name='item_emb', dtype=tf.float32)
            #self.W = tf.Variable(tf.truncated_normal(shape=[self.n_factors+1, self.weight_size], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.weight_size + self.n_factors+1))),name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            #self.b = tf.Variable(tf.truncated_normal(shape=[1, self.weight_size], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.weight_size + self.n_factors+1))),name='Bias_for_MLP', dtype=tf.float32, trainable=True)
            #self.user_bias = tf.get_variable('user_bias', shape=(self.n_users, self.n_factors + 1), dtype=tf.float32, trainable=True)
            

    def _create_graph(self):
        # get user_map and item_mapsq
        self.user_map = tf.nn.embedding_lookup(self.user_map_embedding, self.user_input) # (b, 1, k, d)
        self.item_map = tf.nn.embedding_lookup(self.item_map_embedding, self.item_input) # (b, s, k, d)
        self.user_emb = tf.nn.embedding_lookup(self.user_embedding, self.user_input) # (b, 1, d)
        self.item_emb = tf.nn.embedding_lookup(self.item_embedding, self.item_input) # (b, s, d)

        #network = SeqSelfAttention(input_shape=(self.n_factors, self.embedding), attention_activation='relu')
        #encoded_l = TimeDistributed(network)(self.user_map) # shape: b**k*d
        #encoded_r = TimeDistributed(network)(self.item_map) # shape: b*s*k*d
        encoded_l = self.user_map
        encoded_r = self.item_map
        self.UI_simMap = tf.concat([tf.reduce_sum(encoded_l * encoded_r, axis=-1), \
                            tf.reduce_sum(self.user_emb * self.item_emb, axis=-1, keep_dims=True)], axis=-1) # (b, s, k) (b, s, 1) -> (b, s, k+1)
        self.similarity = tf.reduce_sum(self.UI_simMap, axis=-1) # (b, s)
        
        '''UI_simMap = tf.reshape(self.UI_simMap, [-1, self.n_factors+1])
        dense_out = tf.matmul(UI_simMap, self.W) + self.b #tf.matmul(UI_simMap, self.W) + self.b #tf.nn.sigmoid(tf.matmul(UI_simMap, self.W) + self.b)
        dense_out = tf.reshape(dense_out, [-1, self.num_neg+1, self.weight_size])
        self.similarity = tf.reduce_sum(dense_out, axis=-1) # (b, s)'''
        

        #self.similarity = tf.reduce_sum(tf.reduce_sum(self.user_map * self.item_map, axis=-1), axis=-1)
        #self.similarity += tf.reduce_sum(self.user_emb * self.item_emb, axis=-1)
        #self.UI_simMap = tf.concat([tf.reduce_sum(self.user_map * self.item_map, axis=-1), \
        #                    tf.reduce_sum(self.user_emb * self.item_emb, axis=-1, keep_dims=True)], axis=-1) # (b, s, k) (b, s, 1) -> (b, s, k+1)
        
        '''self.KB_sim = tf.reduce_sum(self.user_map * self.item_map, axis=-1) # (b, s, k)
        self.MF_sim = tf.reduce_sum(self.user_emb * self.item_emb, axis=-1, keep_dims=True) # (b, s, 1)
        self.user_w = tf.expand_dims(tf.nn.softmax(tf.squeeze(tf.nn.embedding_lookup(self.user_bias, self.user_input))), axis=1) # (b, 1, k+1)
        self.similarity = tf.reduce_sum(tf.concat([self.KB_sim, self.MF_sim], axis=-1) * self.user_w, axis=-1) # (b, s)'''
        self.output = self.similarity[:, 0:1] - self.similarity[:, 1:] # (b, s)

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

def training(n_epochs=40, batch_size=256, num_neg=4, resample=True, verbose=10):
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    data = Dataset(args.path)
    idx_images, idx2item = data.item4cke()
    #idx_images, idx2item = data.item2image()
    #idx_images = idx_images [:, -1:, :]
    #print idx_images[0]
    users = data.train['UserId'].unique()
    user2idx = pd.Series(data=np.arange(len(users)), index=users)
    item2idx = pd.Series(data=np.arange(len(idx2item)), index=idx2item)
    start_time = time.time()
    train_l, train_r = generate_input_fast(data.train, item2idx, user2idx, num_neg)
    print "generating data needed for 1 epoch", time.time() - start_time, 's'
    n_items, n_factors, embedding = idx_images.shape
    
    print n_items, len(idx2item)
    print "the shape of images is", idx_images.shape
    model = Ourmodel(n_items, len(users), n_factors, idx_images, embedding, 0.01)
    model.build_graph()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print "initialized"

        #initialize for training batches
        #feed_dict[model.user_input] = train_l
        #feed_dict[model.item_input] = train_r
        
        #eval(sess, model, data, item2idx, user2idx, idx2item, users, False)
        #save(model.user_bias.eval(), 'user_bias.epoch0')
        #save_UI_simMap(sess, model, train_l, train_r, 'train_DT.pkl')
        if args.pretrain == True:
            print "load the pretrain model"
            saver.restore(sess, "./model/cke.ckpt")
            #eval(sess, model, data, item2idx, user2idx, idx2item, users)
            #save_UI_simMap(sess, model, train_l, train_r, 'train_DT.pkl')
            eval(sess, model, data, item2idx, user2idx, idx2item, users, True)
            return 

        for epoch in range(n_epochs):
            if resample == True:
                #user_negs = data.neg_sample_pop(idx2item, pop, pop_items)
                #user_negs = data.neg_sample(idx2item)
                train_l, train_r = generate_input_fast(data.train, item2idx, user2idx, num_neg)
            losses, lrs, l2_losss = [], [], []
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
                #_, loss, l2_loss = sess.run([model.optimizer, model.loss, model.l2_loss], feed_dict)               
                start += batch_size
                losses.append(loss)
                lrs.append(lr)
                l2_losss.append(l2_loss)
            print "epoch", epoch, ", loss", np.array(losses).mean(), np.array(lrs).mean(), np.array(l2_losss).mean()
            #print lrs
            #eval(sess, model, data, item2idx, user2idx, idx2item, users, False)
            if (epoch+1) % verbose == 0:
                eval(sess, model, data, item2idx, user2idx, idx2item, users)
                #train_l, train_r = generate_input(data.train, item2idx, user2idx, num_neg)
                #save(model.user_bias.eval(), 'user_bias.epoch' + str(epoch+1))
        #save_UI_simMap(sess, model, train_l, train_r, 'train_DT.pkl')
        #eval(sess, model, data, item2idx, user2idx, idx2item, users, True)
        #eval(sess, model, data, item2idx, user2idx, idx2item, users)
        #saver.save(sess, "./model/cke.ckpt")

def save(vectors, filename):
    fp = open(filename, 'w')
    for i in range(vectors.shape[0]):
        v = [str(round(x, 5)) for x in vectors[i]]
        fp.write("\t".join(v) + '\n')
    fp.close()

def generate_input_fast(train_set, item2idx, user2idx, num_neg, num_thread=10):
    u_is = {}
    pos = train_set['ItemId'].groupby(train_set['UserId'])
    for u, group in pos:
        u_is[u] = list(group.values)
    thread_size = train_set.shape[0] / num_thread
    q = multiprocessing.Queue()
    # create multiprocessing
    thread_ps = []
    for thread in range(num_thread-1):
        train_set_thread = train_set[thread*thread_size: (thread+1)*thread_size]
        p = multiprocessing.Process(target=generate_input, args=(train_set_thread, item2idx, user2idx, num_neg, q, u_is, thread))
        p.start()
        thread_ps.append(p)
    train_set_thread = train_set[(num_thread-1)*thread_size:]
    p = multiprocessing.Process(target=generate_input, args=(train_set_thread, item2idx, user2idx, num_neg, q, u_is, num_thread-1))
    p.start()
    thread_ps.append(p)
    # merge all the data generate from multiprocessing
    train_l, train_r = [], []
    t_data = {}
    for thread in range(num_thread):
        [thread_id, l, r] = q.get()
        t_data[thread_id] = [l, r]
        train_l.append(l)
        train_r.append(r)
        #print "get queue data", l.shape, r.shape
    '''for thread_id in t_data:
        [l, r] = t_data[thread_id]
        train_l.append(l)
        train_r.append(r)'''
    for p in thread_ps:
        p.join()
    train_l = np.concatenate(train_l, axis=0)
    train_r = np.concatenate(train_r, axis=0)
    order = np.arange(len(train_l))
    np.random.shuffle(order)
    train_l, train_r = train_l[order], train_r[order]
    print "after merging", train_l.shape, train_r.shape
    return train_l, train_r

def generate_input(train_set, item2idx, user2idx, num_neg, q=None, u_is=None, thread_id=0):
    # generate the input data
    train_l, train_r = [], []
    pos = train_set['ItemId'].groupby(train_set['UserId'])
    items = item2idx.index.values
    n_items = len(items)
    for u, group in pos:
        pis = list(group.values) # the positive items
        all_pis = pis
        if u_is is not None:
            all_pis = u_is[u]
        for i in pis:
            if i not in item2idx:
                continue
            negs = []
            while len(negs) < num_neg:
                ti = items[np.random.randint(n_items)]
                if ti not in negs and ti not in all_pis:
                    negs.append(ti)
            cans = [i] + negs
            #cans = [i] + list(np.random.choice(negs, num_neg))
            cans2idx = [item2idx[x] for x in cans]
            train_l.append([user2idx[u]])
            train_r.append(cans2idx)
    train_l = np.array(train_l, dtype=np.int32)
    train_r = np.array(train_r, dtype=np.int32)
    #train_y = np.array(train_y, dtype=np.float32)
    if q is not None:
        q.put([thread_id, train_l, train_r])
        #print "thread over"
        exit(0)
    order = np.arange(len(train_l))
    np.random.shuffle(order)
    train_l, train_r = train_l[order], train_r[order]
    print train_l.shape, train_r.shape #, train_y.shape
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
    UI_label = np.concatenate(UI_label, axis=0)
    UI_simMap = np.concatenate(UI_simMap, axis=0)
    print UI_simMap.shape, UI_label.shape
    fp = open(filename, 'wb')
    pickle.dump(np.concatenate([UI_label, UI_simMap], axis=-1), fp, -1) #(b*s, k+1)
    fp.close()

def eval(sess, model, data, item2idx, user2idx, items, users, save=False):
    hits, ndcgs, mrrs, losses = [],[],[],[]
    #users, items = user2idx.index, item2idx.index
    n_copy = model.num_neg + 1
    user_negs = data.load_negative()
    UI_simMap = {}
    # check the test item with special attrs
    special_attrs = ['film.film.actor', 'film.film.directed_by']
    special_items = data.get_item_with_attrs(special_attrs) # kgid
    number = 0
    for u in user_negs:
        gtItem, cans = user_negs[u]
        if gtItem[0] not in items:# or data.item2en[gtItem[0]] in special_items:
            continue
        number += 1
        test_l, test_r = [], []
        cansidx = [item2idx[x] for x in cans if x in item2idx]
        cansidx.append(item2idx[gtItem[0]])
        uidx = user2idx[u]
        for iidx in cansidx:
            test_l.append([uidx])
            test_r.append([iidx] * n_copy)
        test_l = np.array(test_l, dtype=np.int32)
        test_r = np.array(test_r, dtype=np.int32)
        feed_dict = dict()
        feed_dict[model.user_input] = test_l
        feed_dict[model.item_input] = test_r
        predictions, simMap = sess.run([model.similarity, model.UI_simMap], feed_dict=feed_dict)
        #predictions = sess.run(model.similarity, feed_dict=feed_dict)
        UI_simMap[u] = simMap[:, 0, :]
        (hr, ndcg, mrr) = _eval_one_rating(predictions[:,0]) # get predictions[:,0] because test one user by one user
        hits.append(hr)
        ndcgs.append(ndcg) 
        mrrs.append(mrr) 
    if save == True:
        fp = open('test_DT.pkl', 'wb')
        pickle.dump(UI_simMap, fp, -1)
        fp.close()
    hr = _mean_dict(hits)
    ndcg = _mean_dict(ndcgs)
    mrr = np.array(mrrs).mean()
    #test_loss = np.array(losses).mean()
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
