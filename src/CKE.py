'''
Changed on May 7th by Jin Huang.
Note!!!
The part of the code is copied from 
Tensorflow Implementation of the Baseline Model, CKE, in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import tensorflow as tf
import os, time
import argparse
import numpy as np

from gcn_generator import CKE_generator
from utils import Dataset
from evaluate import eval_one_rating

os.environ["CUDA_VISIBLE_DEVICES"] = '1' 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', type=bool, default=False, help='pretrain or not')
    print "some input parameter"
    args = parser.parse_args()
    return args

class CKE(object):
    def __init__(self, data_config, pretrain_data=None):
        self._parse_args(data_config, pretrain_data)
        self._build_inputs()
        self.weights = self._build_weights()
        self._build_model()
        self._build_loss()
        self._statistics_params()

    def _parse_args(self, data_config, pretrain_data):
        self.model_type = 'cke'
        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_entities = data_config['n_entities']
        self.n_relations = data_config['n_relations']

        self.lr = data_config['lr']
        # settings for CF part.
        self.emb_dim = data_config['embed_size'] 
        #self.batch_size = data_config['batch_size'] 

        # settings for KG part.
        self.kge_dim = data_config['kge_size'] # args.kge_size

        self.regs = data_config['regs'] # eval(args.regs)

        self.verbose = 1

    def _build_inputs(self):
        # for user-item interaction modelling
        self.u = tf.placeholder(tf.int32, shape=[None,], name='u')
        self.pos_i = tf.placeholder(tf.int32, shape=[None,], name='pos_i')
        self.neg_i = tf.placeholder(tf.int32, shape=[None,], name='neg_i')

        # for knowledge graph modeling (TransD)
        self.h = tf.placeholder(tf.int32, shape=[None], name='h')
        self.r = tf.placeholder(tf.int32, shape=[None], name='r')
        self.pos_t = tf.placeholder(tf.int32, shape=[None], name='pos_t')
        self.neg_t = tf.placeholder(tf.int32, shape=[None], name='neg_t')


    def _build_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['user_embed'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embed')
            all_weights['item_embed'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embed')
            print('using xavier initialization')
        else:
            all_weights['user_embed'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                    name='user_embed', dtype=tf.float32)
            all_weights['item_embed'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                    name='item_embed', dtype=tf.float32)
            print('using pretrained initialization')

        all_weights['kg_entity_embed'] = tf.Variable(initializer([self.n_entities, 1, self.emb_dim]),
                                                     name='kg_entity_embed')
        all_weights['kg_relation_embed'] = tf.Variable(initializer([self.n_relations, self.kge_dim]),
                                                       name='kg_relation_embed')

        all_weights['trans_W'] = tf.Variable(initializer([self.n_relations, self.emb_dim, self.kge_dim]))

        return all_weights

    def _build_model(self):
        self.u_e, self.pos_i_e, self.neg_i_e = self._get_cf_inference(self.u, self.pos_i, self.neg_i)

        self.h_e, self.r_e, self.pos_t_e, self.neg_t_e= self._get_kg_inference(self.h, self.r, self.pos_t, self.neg_t)

        # All predictions for all users.
        self.batch_predictions = tf.matmul(self.u_e, self.pos_i_e, transpose_a=False, transpose_b=True)


    def _build_loss(self):
        self.kg_loss, self.kg_reg_loss = self._get_kg_loss()
        self.cf_loss, self.cf_reg_loss = self._get_cf_loss()

        self.base_loss = self.cf_loss
        self.kge_loss = self.kg_loss
        self.reg_loss = self.regs[0] * self.cf_reg_loss + self.regs[1] * self.kg_reg_loss
        self.loss = self.base_loss + self.kge_loss + self.reg_loss

        # Optimization process.
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _get_kg_inference(self, h, r, pos_t, neg_t):
        # head & tail entity embeddings: batch_size *1 * emb_dim
        h_e = tf.nn.embedding_lookup(self.weights['kg_entity_embed'], h)
        pos_t_e = tf.nn.embedding_lookup(self.weights['kg_entity_embed'], pos_t)
        neg_t_e = tf.nn.embedding_lookup(self.weights['kg_entity_embed'], neg_t)

        # relation embeddings: batch_size * kge_dim
        r_e = tf.nn.embedding_lookup(self.weights['kg_relation_embed'], r)

        # relation transform weights: batch_size * kge_dim * emb_dim
        trans_M = tf.nn.embedding_lookup(self.weights['trans_W'], r)

        # batch_size * 1 * kge_dim -> batch_size * kge_dim
        h_e = tf.reshape(tf.matmul(h_e, trans_M), [-1, self.kge_dim])
        pos_t_e = tf.reshape(tf.matmul(pos_t_e, trans_M), [-1, self.kge_dim])
        neg_t_e = tf.reshape(tf.matmul(neg_t_e, trans_M), [-1, self.kge_dim])

        # l2-normalize
        h_e = tf.nn.l2_normalize(h_e, axis=1)
        r_e = tf.nn.l2_normalize(r_e, axis=1)
        pos_t_e = tf.nn.l2_normalize(pos_t_e, axis=1)
        neg_t_e = tf.nn.l2_normalize(neg_t_e, axis=1)

        return h_e, r_e, pos_t_e, neg_t_e

    def _get_cf_inference(self, u, pos_i, neg_i):
        u_e = tf.nn.embedding_lookup(self.weights['user_embed'], u)
        pos_i_e = tf.nn.embedding_lookup(self.weights['item_embed'], pos_i)
        neg_i_e = tf.nn.embedding_lookup(self.weights['item_embed'], neg_i)

        pos_i_kg_e = tf.reshape(tf.nn.embedding_lookup(self.weights['kg_entity_embed'], pos_i), [-1, self.emb_dim])
        neg_i_kg_e = tf.reshape(tf.nn.embedding_lookup(self.weights['kg_entity_embed'], neg_i), [-1, self.emb_dim])

        return u_e, pos_i_e + pos_i_kg_e, neg_i_e + neg_i_kg_e

    def _get_kg_loss(self):
        def _get_kg_score(h_e, r_e, t_e):
            kg_score = tf.reduce_sum(tf.square((h_e + r_e - t_e)), 1, keepdims=True)
            return kg_score

        pos_kg_score = _get_kg_score(self.h_e, self.r_e, self.pos_t_e)
        neg_kg_score = _get_kg_score(self.h_e, self.r_e, self.neg_t_e)

        maxi = tf.log(tf.nn.sigmoid(neg_kg_score - pos_kg_score))
        kg_loss = tf.negative(tf.reduce_mean(maxi))
        kg_reg_loss = tf.nn.l2_loss(self.h_e) + tf.nn.l2_loss(self.r_e) + \
                      tf.nn.l2_loss(self.pos_t_e) + tf.nn.l2_loss(self.neg_t_e)

        return kg_loss, kg_reg_loss

    def _get_cf_loss(self):
        def _get_cf_score(u_e, i_e):
            cf_score = tf.reduce_sum(tf.multiply(u_e, i_e), axis=1)
            return cf_score

        pos_cf_score = _get_cf_score(self.u_e, self.pos_i_e)
        neg_cf_score = _get_cf_score(self.u_e, self.neg_i_e)

        maxi = tf.log(1e-10 + tf.nn.sigmoid(pos_cf_score - neg_cf_score))
        cf_loss = tf.negative(tf.reduce_mean(maxi))
        cf_reg_loss = tf.nn.l2_loss(self.u_e) + tf.nn.l2_loss(self.pos_i_e) + tf.nn.l2_loss(self.neg_i_e)

        return cf_loss, cf_reg_loss

    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

    def train(self, sess, feed_dict):
        return sess.run([self.opt, self.loss, self.base_loss, self.kge_loss, self.reg_loss], feed_dict)

    def evaluate(self, sess, feed_dict):
        batch_predictions = sess.run(self.batch_predictions, feed_dict)
        return batch_predictions

def training(path, n_epochs=40, batch_size=1024, num_neg=4, learning_rate=0.01, embedding=32, resample=True):
    args = parse_args()
    # ------------- load data ------------- #
    data = Dataset(path)
    data_generator = CKE_generator(data)
    #n_users, n_rels, n_entities = data_generator.n_users, data_generator.n_rels, data_generator.n_entities
    start_time = time.time()
    train_users, train_pos_items, train_neg_items = data_generator.generate_rs_data_fast()
    kg_heads, kg_pos_tails, kg_neg_tails, kg_rels = data_generator.generate_kg_data_fast()
    print "generate data, time:", time.time() - start_time, 's'

    # ------------ build the model ------------- #
    data_config = dict()
    data_config['n_users'] = data_generator.n_users
    data_config['n_items'] = data_generator.n_items
    data_config['n_entities'] = data_generator.n_entities
    data_config['n_relations'] = data_generator.n_rels
    data_config['lr'] = 0.01
    data_config['embed_size'] = 32
    #data_config['batch_size'] = batch_size
    data_config['kge_size'] = 32
    data_config['regs'] = [1e-5,1e-5] #,1e-2]

    model = CKE(data_config=data_config)

    # ------------ train the model -------------- #
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        print "initialized"
        if args.pretrain == True:
            print "load the pretrain model"
            saver.restore(sess, "./model/ml-1m_cke_NoKGLoss.ckpt") # usermap
            #saver.restore(sess, "./model/usermap.ckpt") 
            evaluate(sess, model, data, data_generator)
        else:
            for epoch in range(n_epochs):
                loss, base_loss, kge_loss, reg_loss = [], [], [], []
                if resample == True and epoch > 0:
                    train_users, train_pos_items, train_neg_items = data_generator.generate_rs_data_fast()
                    kg_heads, kg_pos_tails, kg_neg_tails, kg_rels = data_generator.generate_kg_data_fast()
                start = 0
                while start < kg_heads.shape[0] or start < train_users.shape[0]:
                    feed_dict = dict()
                    l_edge, r_edge = start, start + batch_size
                    # ------------ deal RS data ------------ #
                    if r_edge >= train_users.shape[0]: # the rs data run out
                        if l_edge < train_users.shape[0]:
                            random_choices = np.random.choice(train_users.shape[0], batch_size-(train_users.shape[0]-l_edge))
                            feed_dict[model.u] = np.concatenate((train_users[l_edge:], train_users[random_choices]), axis=0)
                            feed_dict[model.pos_i] = np.concatenate((train_pos_items[l_edge:], train_pos_items[random_choices]), axis=0)
                            feed_dict[model.neg_i] = np.concatenate((train_neg_items[l_edge:], train_neg_items[random_choices]), axis=0)
                        else:
                            random_choices = np.random.choice(train_users.shape[0], batch_size)
                            feed_dict[model.u] = train_users[random_choices]
                            feed_dict[model.pos_i] = train_pos_items[random_choices]
                            feed_dict[model.neg_i] = train_neg_items[random_choices]
                    else:
                        feed_dict[model.u] = train_users[l_edge:r_edge]
                        feed_dict[model.pos_i] = train_pos_items[l_edge:r_edge]
                        feed_dict[model.neg_i] = train_neg_items[l_edge:r_edge]
                    # ----------- deal KG data ------------- #
                    if r_edge >= kg_heads.shape[0]:
                        if l_edge < kg_heads.shape[0]:
                            random_choices = np.random.choice(kg_heads.shape[0], batch_size-(kg_heads.shape[0]-l_edge))
                            feed_dict[model.h] = np.concatenate((kg_heads[l_edge:], kg_heads[random_choices]), axis=0)
                            feed_dict[model.r] = np.concatenate((kg_rels[l_edge:], kg_rels[random_choices]), axis=0)
                            feed_dict[model.pos_t] = np.concatenate((kg_pos_tails[l_edge:], kg_pos_tails[random_choices]), axis=0)
                            feed_dict[model.neg_t] = np.concatenate((kg_neg_tails[l_edge:], kg_neg_tails[random_choices]), axis=0)
                        else:
                            random_choices = np.random.choice(train_users, batch_size)
                            feed_dict[model.h] = kg_heads[random_choices]
                            feed_dict[model.r] = kg_rels[random_choices]
                            feed_dict[model.pos_t] = kg_pos_tails[random_choices]
                            feed_dict[model.neg_t] = kg_neg_tails[random_choices]
                    else:
                        feed_dict[model.h] = kg_heads[l_edge:r_edge]
                        feed_dict[model.r] = kg_rels[l_edge:r_edge]
                        feed_dict[model.pos_t] = kg_pos_tails[l_edge:r_edge]
                        feed_dict[model.neg_t] = kg_neg_tails[l_edge:r_edge]
                    # ----------- train the model ------------- #
                    _, batch_loss, batch_base_loss, batch_kge_loss, batch_reg_loss = model.train(sess, feed_dict=feed_dict)
                    start = r_edge
                    loss.append(batch_loss)
                    base_loss.append(batch_base_loss)
                    kge_loss.append(batch_kge_loss)
                    reg_loss.append(batch_reg_loss)
                    #exit(0)
                print "epoch", epoch, ", loss", np.array(loss).mean(), np.array(kge_loss).mean(), np.array(reg_loss).mean()
                if (epoch + 1) % 10 == 0:
                    evaluate(sess, model, data, data_generator)
                #eval(sess, model, data, item2idx, user2idx, idx2item, users, False)
            saver.save(sess, "./model/ml-1m_cke_NoKGLoss.ckpt")
        

def evaluate(sess, model, data, data_generator):
    # ------------ test the model -------------- #
    hits, ndcgs, mrrs, losses = [],[],[],[]
    test_user_negs = data.load_negative() # {u:[gtItem, negs]}
    for u in test_user_negs:
        [gtItem, negs] = test_user_negs[u]
        gtItem = gtItem[0]
        if gtItem not in data_generator.items:
            continue
        uidx = data_generator.user2idx[u]
        gtidx = data_generator.item2idx[gtItem]
        negidxs = [data_generator.item2idx[x] for x in negs if x in data_generator.items]
        canidxs = negidxs + [gtidx]
        feed_dict = dict()
        feed_dict[model.u] = np.array([uidx]*len(canidxs), dtype=np.int32)
        feed_dict[model.pos_i] = np.array(canidxs, dtype=np.int32)
        predictions = model.evaluate(sess, feed_dict)
        (hr, ndcg, mrr) = eval_one_rating(np.diag(predictions)) # get predictions[:,0] because test one user by one user
        #print hr, ndcg, mrr
        #print canidxs
        #print predictions
        #print feed_dict[model.u].shape, feed_dict[model.pos_i].shape, predictions.shape
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


if __name__ == '__main__':
    training('../data/ml-1m/')