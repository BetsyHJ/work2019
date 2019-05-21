import sys, time
import pandas as pd
import numpy as np
import pickle
import copy
from collections import Counter
# to load the RS and KG data
# np.random.seed(517)

class Dataset(object):
    path = None 
    train, test = None, None # rating set
    KG_path = 'KG' #'HIN' #'KG'
    RS_path = 'RS' #'RS_short' # 'RS'
    kg = None
    item2en, en2item = None, None
    n_negs = 4

    def load_RS(self):
        filename = self.path + self.RS_path + '/train.rating'
        ratings = pd.read_csv(filename, sep='\t', header=None, dtype='str')
        ratings.columns = ['UserId', 'ItemId', 'rating']
        self.train = ratings

        filename = self.path + self.RS_path + '/test.rating'
        ratings = pd.read_csv(filename, sep='\t', header=None, dtype='str')
        ratings.columns = ['UserId', 'ItemId', 'rating']
        self.test = ratings

    def load_negative(self):
        filename = self.path + self.RS_path + '/test.negative'
        user_negs = {}
        for line in open(filename):
            s = line.strip().split('\t')
            negs = s[1:]
            #u, gtItem = s[0].strip()[1:-1].split(',')
            ss = s[0].strip()[1:-1].split(',')
            u, gtItem = ss[0],ss[1:]
            user_negs[u] = [gtItem, negs]
        return user_negs

    def load_KG(self):
        filename = self.path + self.KG_path + '/KG.txt' #'/KG_comp_transe_top3.txt' #'/KG.txt'
        triples = pd.read_csv(filename, header=None, sep = '\t', dtype='str')
        triples.columns = ['head', 'tail', 'rel']
        # filter un-related entities
        items = list(self.train['ItemId'].unique())
        useful_hs = [self.item2en[x] for x in items]
        triples = triples[triples['head'].isin(useful_hs)]
        self.kg = triples

    def load_item2en(self):
        filename = self.path + self.KG_path + '/seed1.txt'
        item2en = pd.read_csv(filename, header=None, index_col=0, sep = '\t', dtype='str')
        item2en.index = item2en.index.astype('str')
        en2item = pd.Series(data=item2en.index, index=item2en.values[:, 0], dtype='str')
        en2item.index = en2item.index.astype('str')
        #print item2en[:10], en2item[:10]
        #self.item2en, self.en2item = item2en, en2item
        item2en_new = pd.Series(index=item2en.index, data=item2en.values[:, 0], dtype='str')
        self.item2en, self.en2item = item2en_new, en2item

    def __init__(self, path):
        self.path = path
        self.load_RS()
        self.load_item2en()
        self.load_KG()
        self.filter_KG()
        
    def get_pop(self):
        pop = self.train.groupby('ItemId').size()
        pop_items = pop.index.values
        print pop_items[:10], len(pop)
        pop = pop.values
        pop = 1.0 * pop.cumsum() / pop.sum()
        pop[-1] = 1.0
        print pop
        return pop, pop_items


    def filter_KG(self):
        print "before filter", len(self.kg)
        self.kg = self.kg[self.kg['head'].isin(self.en2item.index)]
        # filter by relations
        #selected_rels = ['film.film.genre', 'film.film.reviewer', 'film.film.actor', 'media_common.netflix_title.netflix_genres']
        #selected_rels = ['film.film.actor', 'film.film.directed_by']
        #print "just consider selected_rels", selected_rels
        #self.kg = self.kg[self.kg['rel'].isin(selected_rels)]
        print "after filter", len(self.kg)

    def neg_sample(self, idx2item):
        pos = self.train['ItemId'].groupby(self.train['UserId']) #self.itemknn['item2'].groupby(self.itemknn['item1'])
        items = idx2item
        n_items = len(items)
        #print idx2item[:10]
        user_negs = {} # format: {user: [items]}
        for u, group in pos:
            negs = []
            pis = group.values # the positive items
            #print u, pis[:10]
            for _ in range(self.n_negs):
                t = items[np.random.randint(0, n_items)]
                while t in negs or t in pis:
                    t = items[np.random.randint(0, n_items)]
                negs.append(t)
            user_negs[u] = negs
        return user_negs

    def neg_sample_pop(self, idx2item, pop, pop_items):
        pos = self.train['ItemId'].groupby(self.train['UserId']) #self.itemknn['item2'].groupby(self.itemknn['item1'])
        items = idx2item
        n_items = len(items)

        #print idx2item[:10]
        user_negs = {} # format: {user: [items]}
        for u, group in pos:
            negs = []
            pis = group.values # the positive items
            #print u, pis[:10]
            for _ in range(self.n_negs):
                t = pop_items[np.searchsorted(pop, np.random.rand())]
                while t not in items or t in negs or t in pis:
                    t = pop_items[np.searchsorted(pop, np.random.rand())]
                negs.append(t)
            user_negs[u] = negs
        return user_negs

    def __read_emb(self, filename):
        f = open(filename)
        emb = {}
        for line in f.readlines():
            s = line.strip().split()
            vec = [float(x) for x in s[1:]]
            emb[s[0]] = np.array(vec, dtype=np.float32)
        f.close()
        return emb

    def load_transX(self):
        rel_file = self.path + self.KG_path + '/TransEOut/relation.emb'
        en_file = self.path + self.KG_path + '/TransEOut/entity.emb'
        rel_vec = self.__read_emb(rel_file)
        en_vec = self.__read_emb(en_file)
        return rel_vec, en_vec
    
    def load_u2a_LINE(self):
        path = self.path + self.KG_path + '/u2a_LINE/'
        rels = list(self.kg['rel'].unique())
        rel_en_emb = {} # {rel: {en:emb} }
        n_ens, dim = 0, 0
        for r in rels:
            filename = path + str(r) + '_u2a.LINE.emb'
            en_emb = {}
            for line in open(filename, 'r'):
                s = line.strip().split()
                if len(s) > 2:
                    en, emb = s[0], np.array([float(x) for x in s[1:]], dtype=np.float32)
                    en_emb[en] = emb
                else:
                    n_ens, dim = int(s[0]), int(s[1])
            if n_ens != len(en_emb) or dim != en_emb[en_emb.keys()[0]].shape[0]:
                print "load u2a LINEOutput error"
                exit(0)
            rel_en_emb[r] = en_emb
        return rel_en_emb, dim
    
    def load_u2e_LINE(self):
        path = self.path + self.KG_path + '/u2e_LINE/'
        rels = list(self.kg['rel'].unique())
        rel_en_emb = {} # {rel: {en:emb} }
        n_ens, dim = 0, 0
        for r in rels:
            filename = path + str(r) + '_u2e.LINE.emb'
            en_emb = {}
            for line in open(filename, 'r'):
                s = line.strip().split()
                if len(s) > 2:
                    en, emb = s[0], np.array([float(x) for x in s[1:]], dtype=np.float32)
                    en_emb[en] = emb
                else:
                    n_ens, dim = int(s[0]), int(s[1])
            if n_ens != len(en_emb) or dim != en_emb[en_emb.keys()[0]].shape[0]:
                print "load u2a LINEOutput error"
                exit(0)
            rel_en_emb[r] = en_emb
        return rel_en_emb, dim

    def load_GCN(self):
        en_file = self.path + '/GCNOut/node_gcn.emb'
        en_vec = self.__read_emb(en_file)
        return en_vec

    def load_bpr(self):
        bpr_file = self.path + 'RS/BPROut/item.emb'
        item_vec = self.__read_emb(bpr_file)
        return item_vec
    
    def load_line(self):
        bpr_file = self.path + 'LINEOut/item.emb'
        item_vec = self.__read_emb(bpr_file)
        return item_vec

    def load_itemknn(self):
        knn_file = self.path + 'item_knn_20.txt'
        triples = pd.read_csv(knn_file, header=None, sep = '\t', dtype='str')
        triples.columns = ['item1', 'item2', 'jaccard']
        return triples

    def item4cke(self):
        item_image = {} # format: {item: matrix}
        items = list(self.en2item.values) 
        rel_vec, en_vec = self.load_transX()
        # get item -> image/matrix for next step
        m_items = set(self.train['ItemId'].unique()) | set(self.test['ItemId'].unique()) 
        for item2en in self.en2item.index.values:
            if self.en2item[item2en] not in m_items:
                continue
            image = [en_vec[item2en]]
            item_image[self.en2item[item2en]] = image
        
        # change format from dict to 3-d array
        idx2item = item_image.keys()
        idx_images = np.array(item_image.values(), dtype=np.float32)
        print "the number of items is", len(idx2item)
        return idx_images, idx2item

    def item2image(self):
        item_image = {} # format: {item: matrix}
        items = list(self.en2item.values) 
        #rel_vec, en_vec = self.load_transX()
        en_vec = self.load_GCN()
        dim = en_vec[en_vec.keys()[0]].shape[0]

        rels = list(self.kg['rel'].unique())
        rel2id = pd.Series(data=np.arange(len(rels)), index=rels)
        id2rel = pd.Series(data=rels, index=np.arange(len(rels)))
        rel2id.to_csv('rel2id.txt', sep='\t')
        #exit(0)
        # if add item bpr embedding as a dim
        #if True:
        #    item_vec = self.load_bpr()
            #item_vec = self.load_line()

        # read item_rel_ens for calculating item_image
        item_rel_ens = {} # format: {item:{rel:[ens]}}
        en2item = list(self.en2item.index)
        #print items[:10]
        for [h, t, r] in self.kg.values:
            if h in en2item and self.en2item[h] in items and r in rel2id:
                item_rel_ens.setdefault(h, {})
                item_rel_ens[h].setdefault(rel2id[r], [])
                item_rel_ens[h][rel2id[r]].append(t)
        #print len(item_rel_ens)
        # get item -> image/matrix for next step
        m_items = set(self.train['ItemId'].unique()) | set(self.test['ItemId'].unique()) 
        for item2en in item_rel_ens:
            if self.en2item[item2en] not in m_items:
                continue
            image = []
            rel_ens = item_rel_ens[item2en]
            #item2en_vec = en_vec[item2en]
            for rid in range(len(rels)):
                if rid not in rel_ens: # if not related entities, try the old way (h + r = t)
                    #ens_vec = item2en_vec + rel_vec[id2rel[rid]] # for transX pretrained 
                    ens_vec = np.zeros(dim, dtype=np.float32)
                else: # if has related entities, use add/ et. to deal with
                    ens_vec = np.zeros(dim, dtype=np.float32)
                    ens = rel_ens[rid]
                    n_ens = 0
                    for en in ens:
                        #if en in en_vec:
                        ens_vec += en_vec[en]
                        n_ens += 1
                    if n_ens > 0:
                        ens_vec /= n_ens
                image.append(ens_vec)
            ## add item_transe_emb
            #image.append(en_vec[item2en])
            ## add item_bpr_emb
            '''it = self.en2item[item2en]
            if it in item_vec:
                image.append(item_vec[it])
            else:
                #print it
                image.append(np.zeros(item2en_vec.shape[0], dtype=np.float32))'''
            item_image[self.en2item[item2en]] = image
        
        # change format from dict to 3-d array
        idx2item = item_image.keys()
        idx_images = np.array(item_image.values(), dtype=np.float32)
        print "the number of items is", len(idx2item)
        #print "the shape of images is", idx_images.shape
        #return idx_images[:,-1:,:], idx2item
        return idx_images, idx2item
    
    def get_rel_mostEn(self):
        r_popen = {} #{re:ens}
        rels = self.kg['rel'].unique()
        for r in rels:
            kg = self.kg[self.kg['rel'] == r]
            pop = kg.groupby('tail').size()
            ens = pop.index.values
            pop = list(pop.values)
            print max(pop), ens[pop.index(max(pop))]
            r_popen[r] = ens[pop.index(max(pop))]
        return r_popen
    
    def item_attr_completion(self, item_rel_ens, itemknn, rels):
        item_rel_ens_new = copy.deepcopy(item_rel_ens)
        # generate itemknn
        item_items = {} # {item:[neighbors]}
        itemknn = itemknn.sort_values(by=['item1', 'jaccard'], ascending = [True, False]) # rising, downing
        pos = itemknn['item2'].groupby(itemknn['item1'])
        items, ens = self.item2en.index, self.item2en.values
        for i, group in pos:
            pis = list(group.values) # the positive items
            if i in items:
                item_items[self.item2en[i]] = [self.item2en[x] for x in pis if x in items]
        #print item_items.keys()[:10]
        r_com = {} # {r:count}
        for item in item_rel_ens:
            rel_ens = item_rel_ens[item]
            if item not in item_items:
                print item
                continue
            item_neighbors = item_items[item]
            for r in range(len(rels)):
                related_attrs = []
                if r not in rel_ens: # use the attr of similar items
                    for x in item_neighbors:
                        if r in item_rel_ens[x]: # x in item_rel_ens and
                            related_attrs += item_rel_ens[x][r]
                if len(related_attrs) > 0:
                    attr_count = Counter(related_attrs)
                    related_attrs = [a for (a, _) in attr_count.most_common(3)]
                    #print related_attrs
                    item_rel_ens_new[item][r] = related_attrs
        return item_rel_ens_new

    def load_completion_byTransE(self):
        filename = self.path + "completion_byTransE.txt" # format: en rel vec
        en_rel_vec = {} # {en:{rel:vec}}
        for line in open(filename, 'r'):
            s = line.strip().split()
            en, rel = s[0], s[1]
            en_rel_vec.setdefault(en, {})
            en_rel_vec[en][rel] = np.array(s[2:], dtype=np.float32)
        return en_rel_vec

    def item2image_u2a(self, Train=False):
        item_image = {} # format: {item: matrix}
        items = list(self.en2item.values) 
        #rel_en_emb, dim = self.load_u2a_LINE()
        rel_en_emb, dim = self.load_u2e_LINE()
        rels = list(self.kg['rel'].unique())
        rel2id = pd.Series(data=np.arange(len(rels)), index=rels)
        id2rel = pd.Series(data=rels, index=np.arange(len(rels)))
        rel2id.to_csv('rel2id.txt', sep='\t')
        # if add item bpr embedding as a dim
        #if True:
        #    item_vec = self.load_bpr()
            #item_vec = self.load_line()

        # read item_rel_ens for calculating item_image
        item_rel_ens = {} # format: {item:{rel:[ens]}}
        en2item = list(self.en2item.index)
        for [h, t, r] in self.kg.values:
            if h in en2item and self.en2item[h] in items and r in rel2id:
                item_rel_ens.setdefault(h, {})
                item_rel_ens[h].setdefault(rel2id[r], [])
                item_rel_ens[h][rel2id[r]].append(t)
        ## complete the lossed attr
        #print "do completion"
        #itemknn = self.load_itemknn()
        #item_rel_ens = self.item_attr_completion(item_rel_ens, itemknn, rels)
        #en_rel_vec = self.load_completion_byTransE()

        # get item -> image/matrix for next step
        m_items = set(self.train['ItemId'].unique()) | set(self.test['ItemId'].unique())
        r_popen = self.get_rel_mostEn()
        for item2en in item_rel_ens:
            if self.en2item[item2en] not in m_items:
                continue
            image = []
            rel_ens = item_rel_ens[item2en]
            for rid in range(len(rels)):
                #if rid not in rel_ens: # if not related entities, use np.zeros(dim)
                #    ens_vec = np.zeros(dim, dtype=np.float32)
                if rid not in rel_ens: # if not related ens, use the pop ens in this relation
                    '''if Train == True:
                        popen = r_popen[rels[rid]]
                        ens_emb = rel_en_emb[rels[rid]] # {en:emb}
                        ens_vec = ens_emb[popen]
                    else: # when for test, we need to keep the loss of the attribute
                        ens_vec = np.zeros(dim, dtype=np.float32)'''
                    '''if item2en in en_rel_vec:
                        ens_vec = en_rel_vec[item2en][rels[rid]]
                    else:
                        ens_vec = np.zeros(dim, dtype=np.float32)'''
                    ens_vec = np.zeros(dim, dtype=np.float32)
                else: # if has related entities, use add/ et. to deal with
                    ens, ens_emb = rel_ens[rid], rel_en_emb[rels[rid]] # later: {en:emb}
                    n_ens = len(ens)
                    ens_vec = np.zeros(dim, dtype=np.float32)
                    for en in ens:
                        if en in ens_emb:
                            ens_vec += ens_emb[en] / n_ens
                image.append(ens_vec)
            '''## add item_bpr_emb
            it = self.en2item[item2en]
            if it in item_vec:
                image.append(item_vec[it])
            else:
                image.append(np.zeros(dim, dtype=np.float32))'''
            item_image[self.en2item[item2en]] = image
        
        # change format from dict to 3-d array
        idx2item = item_image.keys()
        idx_images = np.array(item_image.values(), dtype=np.float32)
        print "the number of items is", len(idx2item)
        #print "the shape of images is", idx_images.shape
        #return idx_images[:,-1:,:], idx2item
        return idx_images, idx2item

    def item_user2image_u2e(self):
        item_image, user_image = {}, {} # format: {item/user: matrix}
        items = list(self.en2item.values) 
        rel_en_emb, dim = self.load_u2e_LINE()
        rels = list(self.kg['rel'].unique())
        rel2id = pd.Series(data=np.arange(len(rels)), index=rels)
        id2rel = pd.Series(data=rels, index=np.arange(len(rels)))
        #rel2id.to_csv('rel2id.txt', sep='\t')

        # read item_rel_ens for calculating item_image
        item_rel_ens = {} # format: {item:{rel:[ens]}}
        en2item = list(self.en2item.index)
        for [h, t, r] in self.kg.values:
            if h in en2item and self.en2item[h] in items and r in rel2id:
                item_rel_ens.setdefault(h, {})
                item_rel_ens[h].setdefault(rel2id[r], [])
                item_rel_ens[h][rel2id[r]].append(t)

        # get item -> image/matrix for next step
        m_items = set(self.train['ItemId'].unique()) | set(self.test['ItemId'].unique())
        r_popen = self.get_rel_mostEn()
        for item2en in item_rel_ens:
            if self.en2item[item2en] not in m_items:
                continue
            i_image = []
            rel_ens = item_rel_ens[item2en]
            for rid in range(len(rels)):
                if rid not in rel_ens: 
                    ens_vec = np.zeros(dim, dtype=np.float32)
                else: # if has related entities, use add/ et. to deal with
                    ens, ens_emb = rel_ens[rid], rel_en_emb[rels[rid]] # later: {en:emb}
                    n_ens = len(ens)
                    ens_vec = np.zeros(dim, dtype=np.float32)
                    for en in ens:
                        if en in ens_emb:
                            ens_vec += ens_emb[en] / n_ens
                i_image.append(ens_vec)
            item_image[self.en2item[item2en]] = i_image
        
        # get user -> image/matrix for next step
        users = list(self.train['UserId'].unique())
        for u in users:
            u_image = []
            for rid in range(len(rels)):
                ens_emb = rel_en_emb[rels[rid]]
                ens_vec = np.zeros(dim, dtype=np.float32)
                if u in ens_emb:
                    ens_vec = ens_emb[u]
                u_image.append(ens_vec)
            user_image[u] = u_image
            
        # change format from dict to 3-d array
        idx2item = item_image.keys()
        iidx_images = np.array(item_image.values(), dtype=np.float32)
        idx2user = user_image.keys()
        uidx_images = np.array(user_image.values(), dtype=np.float32)
        print "the number of items and users are", len(idx2item), len(idx2user)
        return iidx_images, idx2item, uidx_images, idx2user 

    def _triples2dict(self, triples):
        _dictList = {} #dict()
        for [u, i, _] in triples.values:
            _dictList.setdefault(u, [])
            _dictList[u].append(i)
        return _dictList

    def get_train_dictList(self):
        return self._triples2dict(self.train)

    def get_test_dictList(self):
        return self._triples2dict(self.test)

    def get_item_with_attrs(self, rels):
        #kg = self.kg[self.kg['rel'].isin(rels)]
        #items = list(kg['head'].unique())
        kg = self.kg[self.kg['rel'] == rels[0]]
        items = set(kg['head'].unique())
        for r in rels[1:]:
            kg = self.kg[self.kg['rel'] == r]
            items = set(items) & set(kg['head'].unique())
            items = list(items)
        return items

if __name__ == '__main__':
    print "load and calculate the RS and KG data"
    path = '../data/music/'
    data = Dataset(path)
    fid2mid = pd.read_csv(path + 'RS/mid2idx/item2id.txt', sep='\t', header=None, index_col=1, dtype='str')
    print fid2mid[:10]
    fid2mid.index = fid2mid.index.astype('int')
    fid2mid = pd.Series(data=fid2mid.values[:, 0], index=fid2mid.index)
    print fid2mid[:10]
    mids = fid2mid.values
    item_map = []
    idx_images, idx2item = data.item2image()
    item2id = pd.Series(data=np.arange(len(idx2item)), index=idx2item)
    loss = np.zeros(idx_images[0].shape, dtype=np.float32)
    for i in range(len(mids)):
        #print fid2mid[i]
        mid = fid2mid[i]
        if mid in idx2item:
            item_map.append(idx_images[item2id[mid]])
        else:
            item_map.append(loss)
    item_map = np.array(item_map, dtype=np.float32)
    fp = open(path + 'RS/mid2idx/item_map.pkl', 'wb')
    pickle.dump(item_map, fp, -1)
    fp.close()
