import os
import tensorflow as tf
import numpy as np
from model import RippleNet
import time
import math

def get_u_is(train_data):
    uidxs = set(train_data[:, 0])
    iidxs = set(train_data[:, 1])
    u_is = {}
    for i in range(train_data.shape[0]):
        t = train_data[i]
        uidx, iidx = t[0], t[1]
        u_is.setdefault(uidx, [])
        u_is[uidx].append(iidx)
    return uidxs, iidxs, u_is

def generate_neg_sampling(u_is, iidxs, train_data, num_neg=4):
    canidxs = list(iidxs)
    n_cans = len(canidxs)
    train_data_new = []
    for i in range(train_data.shape[0]):
        t = train_data[i]
        uidx, iidx = t[0], t[1]
        posidxs = u_is[uidx]
        negidxs = []
        while len(negidxs) < num_neg:
            t = canidxs[np.random.randint(n_cans)]
            if t not in negidxs and i not in posidxs:
                negidxs.append(t)
                train_data_new.append([uidx, iidx, t])
    return np.array(train_data, dtype=np.int64)

def train(args, data_info, show_loss):
    train_data_just_ui = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]
    uidx_negs = data_info[6]

    # -------------- record u_is and neg_sampling for training ------------- #
    uidxs, iidxs, u_is = get_u_is(train_data_just_ui)
    # generate_neg_sampling(u_is, iidxs, train_data)

    model = RippleNet(args, n_entity, n_relation)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        #eval_acc = evaluation(sess, args, model, eval_data, ripple_set, args.batch_size)
        #test(sess, args, model, test_data, uidx_negs, ripple_set, args.batch_size)
        
        for step in range(args.n_epoch):
            start_time = time.time()
            train_data = generate_neg_sampling(u_is, iidxs, train_data_just_ui)
            print("negative sampling done. %f s" % (time.time()-start_time))
            np.random.shuffle(train_data)
            eval_data = train_data[-4096:, :]
            start = 0
            while start < train_data.shape[0]:
                feed_dict = get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size)
                feed_dict[model.global_step] = step
                _, loss = model.train(sess, feed_dict)
                start += args.batch_size
                #if start % 102400 == 0:
                    #if show_loss:
                    #print(start)
                #print('%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss))

            # evaluation
            #train_acc = evaluation(sess, args, model, train_data, ripple_set, args.batch_size)
            eval_acc = evaluation(sess, args, model, eval_data, ripple_set, args.batch_size)
            test(sess, args, model, test_data, uidx_negs, ripple_set, args.batch_size)
            print('epoch %d    train acc: %.4f    eval acc: %.4f ' 
                  % (step, eval_acc, eval_acc))


def get_feed_dict(args, model, data, ripple_set, start, end):
    feed_dict = dict()
    feed_dict[model.pos_items] = data[start:end, 1]
    feed_dict[model.neg_items] = data[start:end, 2]
    for i in range(args.n_hop):
        feed_dict[model.memories_h[i]] = [ripple_set[user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_r[i]] = [ripple_set[user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_t[i]] = [ripple_set[user][i][2] for user in data[start:end, 0]]
    return feed_dict


def evaluation(sess, args, model, data, ripple_set, batch_size):
    start = 0
    acc_list, losses = [], []
    while start < data.shape[0]:
        acc, loss = model.eval(sess, get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        acc_list.append(acc)
        losses.append(loss)
        start += batch_size
    print("the loss is %f" % float(np.array(loss).mean()))
    return float(np.concatenate(acc_list, axis=0).mean())


def get_feed_dict_test(args, model, data, ripple_set, start, end):
    feed_dict = dict()
    feed_dict[model.pos_items] = data[start:end, 1]
    for i in range(args.n_hop):
        feed_dict[model.memories_h[i]] = [ripple_set[user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_r[i]] = [ripple_set[user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_t[i]] = [ripple_set[user][i][2] for user in data[start:end, 0]]
    return feed_dict

def test(sess, args, model, data, uidx_negs, ripple_set, batch_size):
    start = 0
    preds = []
    while start < data.shape[0]:
        pred = model.test(sess, get_feed_dict_test(args, model, data, ripple_set, start, start + batch_size))
        preds.append(pred)
        start += batch_size
    preds = np.concatenate(preds, axis=0)
    scores = {}
    hits, ndcgs, mrrs = [],[],[]
    for i in range(len(data)):
        uidx, iidx = data[i]
        scores.setdefault(uidx, {})
        scores[uidx][iidx] = preds[i]
    for uidx in uidx_negs:
        iidx, negs = uidx_negs[uidx]
        part_scores = scores[uidx]
        predictions = [part_scores[x] for x in negs+[iidx]]
        (hr, ndcg, mrr) = eval_one_rating(predictions)
        hits.append(hr)
        ndcgs.append(ndcg) 
        mrrs.append(mrr) 
        
    hr = _mean_dict(hits)
    ndcg = _mean_dict(ndcgs)
    mrr = np.array(mrrs).mean()
    print("the number of ui pairs in testset is %d" % len(uidx_negs))
    s = ['HR@'+str(x) for x in hr] + ['NGCD@'+str(x) for x in ndcg] + ['mrr'] #+ ['loss']
    print('\t'.join(s))
    s = [str(round(hr[x], 5)) for x in hr] + [str(round(ndcg[x], 5)) for x in ndcg] + [str(round(mrr, 5))] #+ [str(round(test_loss, 5))]
    print("\t".join(s))

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

def eval_one_rating(predictions):
    neg_predict, pos_predict = predictions[:-1], predictions[-1]
    position = (neg_predict > pos_predict).sum()
    K = [1, 3, 5, 10, 15, 20]
    hr, ndcg = {}, {}
    for k in K:
        hr[k] = _getHR(position, k)
        ndcg[k] = _getNDCG(position, k)
    mrr = 1.0 / (position+1)
    #print (str(position)+str(hr))
    return (hr, ndcg, mrr)

def _getHR(location, K):
    if location < K:
        return 1.0
    return 0.0
def _getNDCG(location, K):
    if location < K:
        return math.log(2) / math.log(location+2)
    return 0
