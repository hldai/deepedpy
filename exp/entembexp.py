import numpy as np
import random
import torch
import torch.nn.functional as F
import json
import logging
import time
from models import deschypembed
from utils import datautils
import ctypes

randgenlib = None
try:
    randgenlib = np.ctypeslib.load_library('randgen', 'clib')
    array_1d_double = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='CONTIGUOUS')
    array_1d_int = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')
    randgenlib.gen_rand_neg.argtypes = [array_1d_double, array_1d_double, ctypes.c_int32, ctypes.c_double,
                                        ctypes.c_int32, array_1d_double, array_1d_int]
    print('using c lib.')
except OSError:
    print('c lib not loaded.')


class RandNegWordGen:
    def __init__(self, vocab, freqs, unig_power=0.6):
        self.vocab = vocab
        self.n_words = len(self.vocab)
        self.freqs = [max(f, 100) for f in freqs]
        self.unig_power = unig_power
        self.w_f_start, self.w_f_end, self.total_freq = self.__build_rand_gen_arr()

    def __build_rand_gen_arr(self):
        w_f_start = np.zeros(len(self.vocab), np.float64)
        w_f_end = np.zeros(len(self.vocab), np.float64)
        total_freq = 0
        for i, (w, freq) in enumerate(zip(self.vocab, self.freqs)):
            w_f_start[i] = total_freq
            total_freq += freq ** self.unig_power
            w_f_end[i] = total_freq
        return w_f_start, w_f_end, total_freq

    def get_rand_word_id(self):
        v = random.random() * self.total_freq
        idx_left, idx_right = 0, self.n_words - 1
        while idx_left <= idx_right:
            idx_mid = (idx_left + idx_right) // 2
            if self.w_f_start[idx_mid] <= v <= self.w_f_end[idx_mid]:
                return idx_mid
            elif self.w_f_start[idx_mid] > v:
                idx_right = idx_mid - 1
            elif self.w_f_end[idx_mid] < v:
                idx_left = idx_mid + 1
            else:
                assert False
        assert False
        # return -1

    def gen_rand_word_ids(self, n):
        rand_word_ids = np.zeros(n, np.int32)
        if randgenlib is None:
            for i in range(n):
                rand_word_ids[i] = self.get_rand_word_id()
            return rand_word_ids
        uniform_rand_arr = np.random.uniform(size=n)
        randgenlib.gen_rand_neg(self.w_f_start, self.w_f_end, self.n_words, self.total_freq, n, uniform_rand_arr,
                                rand_word_ids)
        return rand_word_ids


class DescDataProducer:
    def __init__(self, rand_word_gen: RandNegWordGen, word_id_dict, wid_to_entity_id_dict, articles_file,
                 entity_title_word_ids, n_words_per_entity, batch_size, n_iter):
        self.wid_to_entity_id_dict = wid_to_entity_id_dict
        self.entity_title_word_ids = entity_title_word_ids
        self.word_id_dict = word_id_dict
        self.articles_file = articles_file
        self.f = open(articles_file, encoding='utf-8')
        self.n_words_per_entity = n_words_per_entity
        self.batch_size = batch_size
        self.iter = 0
        self.n_iter = n_iter
        self.rand_word_gen = rand_word_gen

    def __get_valid_words_from_sents(self, sents):
        words = list()
        for sent in sents:
            sent_words = sent.split(' ')
            for w in sent_words:
                word_id = self.word_id_dict.get(w, None)
                if word_id is not None:
                    words.append(word_id)
        return words

    # TODO use title words
    def get_pos_words_batch(self):
        if self.iter == self.n_iter:
            return list(), list()

        entity_ids, all_pos_word_ids = list(), list()
        for i in range(self.batch_size):
            try:
                line = next(self.f)
            except StopIteration:
                self.f.close()
                self.iter += 1
                logging.info('desc iter={}'.format(self.iter))
                if self.iter == self.n_iter:
                    break
                self.f = open(self.articles_file, encoding='utf-8')
                line = next(self.f)
            obj = json.loads(line)
            wid = obj['wid']
            entity_id = self.wid_to_entity_id_dict.get(wid, None)
            if entity_id is None:
                i -= 1
                continue

            entity_ids.append(entity_id)
            # word_ids = self.__get_valid_words_from_sents(obj['sents'])
            # word_ids = obj['word_ids']
            word_ids = self.entity_title_word_ids[entity_id] + obj['word_ids']
            if len(word_ids) == 0:
                word_ids = self.entity_title_word_ids[entity_id]
            if len(word_ids) == 0:
                word_ids = self.rand_word_gen.gen_rand_word_ids(10)

            rand_pos_words = list()
            for _ in range(self.n_words_per_entity):
                rand_idx = random.randint(0, len(word_ids) - 1)
                rand_pos_words.append(word_ids[rand_idx])
            all_pos_word_ids.append(rand_pos_words)
        return entity_ids, all_pos_word_ids


class AnchorDataProducer:
    def __init__(self, rand_word_gen: RandNegWordGen, anchor_cxt_file, wid_to_entity_id_dict,
                 entity_title_word_ids, hyp_ctxt_len, n_words_per_entity, batch_size, n_iter):
        self.entity_title_word_ids = entity_title_word_ids
        self.rand_word_gen = rand_word_gen
        self.wid_to_entity_id_dict = wid_to_entity_id_dict
        self.hyp_ctxt_len = hyp_ctxt_len
        self.anchor_cxt_file = anchor_cxt_file
        self.f = open(self.anchor_cxt_file, encoding='utf-8')
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.iter = 0
        self.n_words_per_entity = n_words_per_entity

    def get_pos_words_batch(self):
        if self.iter == self.n_iter:
            return list(), list()

        entity_ids, all_pos_word_ids = list(), list()
        for i in range(self.batch_size):
            try:
                line = next(self.f)
            except StopIteration:
                self.f.close()
                self.iter += 1
                logging.info('anchor iter={}'.format(self.iter))
                if self.iter == self.n_iter:
                    break
                self.f = open(self.anchor_cxt_file, encoding='utf-8')
                line = next(self.f)
            obj = json.loads(line)
            wid = obj['target_wid']
            entity_id = self.wid_to_entity_id_dict.get(wid, None)
            if entity_id is None:
                i -= 1
                continue

            entity_ids.append(entity_id)
            # cxt_words = obj['cxt'].split(' ')
            cxt_word_ids = obj['word_ids']
            word_ids = self.__get_word_ids_from_cxt(cxt_word_ids, obj['beg_idx'], obj['end_idx'])
            if len(word_ids) == 0:
                word_ids = self.entity_title_word_ids[entity_id]
            if len(word_ids) == 0:
                word_ids = self.rand_word_gen.gen_rand_word_ids(10)

            rand_pos_words = list()
            for _ in range(self.n_words_per_entity):
                rand_idx = random.randint(0, len(word_ids) - 1)
                rand_pos_words.append(word_ids[rand_idx])
            all_pos_word_ids.append(rand_pos_words)
        return entity_ids, all_pos_word_ids

    def __get_word_ids_from_cxt(self, cxt_word_ids, anchor_beg_idx, anchor_end_idx):
        word_ids = list()
        beg_idx = max(0, anchor_beg_idx - self.hyp_ctxt_len)
        end_idx = min(len(cxt_word_ids), anchor_end_idx + self.hyp_ctxt_len)
        for i in range(beg_idx, end_idx):
            if anchor_beg_idx <= i < anchor_end_idx:
                continue
            word_id = cxt_word_ids[i]
            # word_id = self.word_id_dict.get(w, None)
            if word_id > -1:
                word_ids.append(word_id)
        return word_ids


def fill_neg_words(neg_word_gen: RandNegWordGen, pos_word_ids_batch, n_neg_words):
    t = 0
    word_ids_batch, pos_word_idxs = list(), list()
    for pos_word_ids in pos_word_ids_batch:
        # cur_entity_word_ids, cur_entity_pos_word_idxs = list(), list()
        # for pos_word_id in pos_word_ids:
        #     tbeg = time.time()
        #     neg_word_ids = list()
        #     for i in range(n_neg_words - 1):
        #         neg_word_ids.append(neg_word_gen.get_rand_word_id())
        #     t += time.time() - tbeg
        #     pos_idx = random.randint(0, n_neg_words - 1)
        #     neg_word_ids.insert(pos_idx, pos_word_id)
        #
        #     cur_entity_pos_word_idxs.append(pos_idx)
        #     cur_entity_word_ids += neg_word_ids
        tbeg = time.time()
        cur_entity_word_ids = neg_word_gen.gen_rand_word_ids(len(pos_word_ids) * n_neg_words)
        cur_entity_pos_word_idxs = list()
        for i, pos_word_id in enumerate(pos_word_ids):
            pos_idx = random.randint(0, n_neg_words - 1)
            cur_entity_word_ids[i * n_neg_words + pos_idx] = pos_word_id
            cur_entity_pos_word_idxs.append(pos_idx)
        t += time.time() - tbeg

        word_ids_batch.append(cur_entity_word_ids)
        pos_word_idxs.append(cur_entity_pos_word_idxs)
    return word_ids_batch, pos_word_idxs, t


def get_entity_title_word_ids(entity_titles, word_to_id_dict):
    import re

    entity_title_word_ids = list()
    pattern = re.compile(r'[.\w]+')
    for title in entity_titles:
        words = pattern.findall(title)
        word_ids = list()
        for w in words:
            word_id = word_to_id_dict.get(w, None)
            if word_id is not None:
                word_ids.append(word_id)
        entity_title_word_ids.append(word_ids)
    return entity_title_word_ids


def get_init_entity_vecs(entity_title_word_ids, word_vecs):
    entity_vecs = list()
    dim = word_vecs.shape[1]
    for word_ids in entity_title_word_ids:
        entity_vec = np.zeros(dim, np.float32) if len(word_ids) > 0 else np.random.normal(scale=0.1, size=dim)
        for word_id in word_ids:
            entity_vec += word_vecs[word_id]
        if len(word_ids) > 0:
            entity_vec /= len(word_ids)
        entity_vecs.append(entity_vec)
    return np.asarray(entity_vecs, np.float32)


def traindeschyp(device, word_freq_vec_pkl, title_wid_file, articles_file, anchor_cxt_file, n_words_per_ent,
                 batch_size, lr, output_file, hyp_ctxt_len=10, n_desc_iter=200, n_neg_words=5, unig_power=0.6):
    word_vocab, freqs, word_vecs = datautils.load_pickle_data(word_freq_vec_pkl)
    word_to_id_dict = {w: i for i, w in enumerate(word_vocab)}
    n_words = word_vecs.shape[0]
    title_wid_df = datautils.load_csv(title_wid_file, False)
    wid_vocab = [wid for _, wid in title_wid_df.itertuples(False, None)]
    entity_titles = [title for title, _ in title_wid_df.itertuples(False, None)]
    entity_title_word_ids = get_entity_title_word_ids(entity_titles, word_to_id_dict)
    wid_idx_dict = {wid: i for i, wid in enumerate(wid_vocab)}
    n_entities = len(wid_vocab)

    logging.info('{} entities, {} words, word_ves: {}'.format(n_entities, n_words, word_vecs.shape))
    neg_word_gen = RandNegWordGen(word_vocab, freqs, unig_power)

    init_entity_vecs = get_init_entity_vecs(entity_title_word_ids, word_vecs)
    model = deschypembed.DescHypEmbed(word_vecs, n_entities, init_entity_vecs)
    if device.type == 'cuda':
        model = model.cuda(device.index)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

    desc_data_producer = DescDataProducer(
        neg_word_gen, word_to_id_dict, wid_idx_dict, articles_file, entity_title_word_ids,
        n_words_per_ent, batch_size, n_desc_iter)
    anchor_data_producer = AnchorDataProducer(
        neg_word_gen, anchor_cxt_file, wid_idx_dict, entity_title_word_ids, hyp_ctxt_len, n_words_per_ent,
        batch_size, 15)
    use_desc_data = True
    num_batches_per_epoch = 8000
    step = 0
    losses = list()
    while True:
        model.train()
        if use_desc_data:
            entity_ids, pos_word_ids_batch = desc_data_producer.get_pos_words_batch()
            if len(entity_ids) == 0:
                use_desc_data = False
                logging.info('start using anchor data ...')
                continue
        else:
            entity_ids, pos_word_ids_batch = anchor_data_producer.get_pos_words_batch()
            # print(pos_word_ids_batch)
            if len(entity_ids) == 0:
                break

        word_ids_batch, pos_word_idxs_batch, ttn = fill_neg_words(neg_word_gen, pos_word_ids_batch, n_neg_words)

        cur_batch_size = len(entity_ids)
        word_ids = torch.tensor(word_ids_batch, dtype=torch.long, device=device)
        entity_ids_tt = torch.tensor(entity_ids, dtype=torch.long, device=device)
        target_idxs = torch.tensor(pos_word_idxs_batch, dtype=torch.long, device=device)
        scores = model(cur_batch_size, word_ids, entity_ids_tt)
        scores = scores.view(-1, n_neg_words)
        target_scores = scores[list(range(cur_batch_size * n_words_per_ent)), target_idxs.view(-1)]
        # target_scores = get_target_scores_with_try()
        loss = torch.mean(F.relu(0.1 - target_scores.view(-1, 1) + scores))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, float('inf'))
        optimizer.step()
        losses.append(loss.data.cpu().numpy())
        step += 1
        if step % num_batches_per_epoch == 0:
            logging.info('i={}, loss={}'.format(step, sum(losses)))
            # logging.info('i={}, loss={}, t0={:.4f}, t1={:.4f}, t2={:.4f}, t3={:.4f}, tn={:.4f}'.format(
            #     step, sum(losses), t0, t1, t2, t3, tn))
            losses = list()

            if output_file:
                entity_vecs = F.normalize(model.entity_embed_layer.weight)
                datautils.save_pickle_data((wid_vocab, entity_vecs.data.cpu().numpy()), output_file)
                logging.info('saved model to {}'.format(output_file))


def get_target_scores_with_try(scores, cur_batch_size, n_words_per_ent, target_idxs, word_ids_batch, n_words,
                               entity_ids, n_entities):
    try:
        target_scores = scores[list(range(cur_batch_size * n_words_per_ent)), target_idxs.view(-1)]
    except RuntimeError:
        for word_ids_tmp in word_ids_batch:
            for word_id in word_ids_tmp:
                if word_id < 0 or word_id >= n_words:
                    print(word_id, n_words)
                    print("WORD ID INCORRECT!")
        for entity_id in entity_ids:
            if entity_id < 0 or entity_id >= n_entities:
                print(entity_id, n_entities)
                print("ENTITY ID INCORRECT!")
        exit()
    return target_scores
