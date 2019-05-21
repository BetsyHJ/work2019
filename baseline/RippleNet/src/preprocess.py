import argparse
import numpy as np

def read_item_index_to_entity_id_file():
    #file = '../data/' + DATASET + '/item_index2entity_id_rehashed.txt'
    filename = PATH + 'KG/seed1.txt'
    print('reading item index to entity id file: ' + filename + ' ...')
    i = 0
    for line in open(filename, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1

def load_negative():
    filename = PATH + 'RS/test.negative'
    user_negs = {}
    for line in open(filename):
        s = line.strip().split('\t')
        negs = s[1:]
        #u, gtItem = s[0].strip()[1:-1].split(',')
        ss = s[0].strip()[1:-1].split(',')
        u, gtItem = ss[0],ss[1]
        user_negs[u] = [gtItem, negs]
    return user_negs

def convert_rating():
    #file = '../data/' + DATASET + '/' + RATING_FILE_NAME[DATASET]
    filename = PATH + 'RS/train.rating'

    print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()
    for line in open(filename, encoding='utf-8').readlines()[1:]:
        array = line.strip().split('\t')
        
        item_index_old = array[1]
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = array[0]
        if user_index_old not in user_pos_ratings:
            user_pos_ratings[user_index_old] = set()
        user_pos_ratings[user_index_old].add(item_index)
    print('converting rating file ...')

    writer = open('../data/' + args.DATASET + '/ratings_final.txt', 'w', encoding='utf-8')
    user_cnt = 0
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        #unwatched_set = item_set - pos_item_set
        for item in pos_item_set:
            writer.write('%d\t%d\n' % (user_index, item))
            #for neg in np.random.choice(list(unwatched_set), size=4, replace=False):
            #    writer.write('%d\t%d\t%d\n' % (user_index, item, neg))
        #    writer.write('%d\t%d\t1\n' % (user_index, item))
        #for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
        #    writer.write('%d\t%d\t0\n' % (user_index, item))
    writer.close()
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))

def convert_test_data():
    #file = '../data/' + DATASET + '/' + RATING_FILE_NAME[DATASET]
    #print(user_index_old2new)
    user_negs = load_negative()
    fp1 = open('../data/' + args.DATASET + '/ratings_test_neg_final.txt', 'w', encoding='utf-8')
    fp2 = open('../data/' + args.DATASET + '/ratings_test_final.txt', 'w', encoding='utf-8')
    for u in user_negs:
        [gtItem, negs] = user_negs[u]
        if gtItem not in item_index_old2new:
            continue
        iidx = item_index_old2new[gtItem]
        uidx = user_index_old2new[u]
        negsidx = [str(item_index_old2new[x]) for x in negs if x in item_index_old2new]
        fp1.write('('+str(uidx) + ',' + str(iidx) + ')' + '\t' + '\t'.join(negsidx) + '\n')
        fp2.write('%d\t%d\n' % (uidx, iidx))
        for i in negsidx:
            fp2.write('%d\t%s\n' % (uidx, i))

def convert_kg():
    print('converting kg file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    writer = open('../data/' + args.DATASET + '/kg_final.txt', 'w', encoding='utf-8')

    filename = PATH + '/KG/KG.txt'
    for line in open(filename, 'r'):
        array = line.strip().split('\t') # format: head tail rel
        head_old = array[0]
        relation_old = array[2]
        tail_old = array[1]

        if head_old not in entity_id2index:
            entity_id2index[head_old] = entity_cnt
            entity_cnt += 1
        head = entity_id2index[head_old]

        if tail_old not in entity_id2index:
            entity_id2index[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2index[tail_old]

        if relation_old not in relation_id2index:
            relation_id2index[relation_old] = relation_cnt
            relation_cnt += 1
        relation = relation_id2index[relation_old]

        writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--DATASET', type=str, default='ml-1m', help='which dataset in this path to preprocess')
    args = parser.parse_args()
    PATH = "../../../data/" + args.DATASET + '/'

    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()

    user_index_old2new = dict()

    read_item_index_to_entity_id_file()
    convert_rating()
    convert_test_data()
    convert_kg()

    print('done')
