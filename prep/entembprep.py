import numpy as np
import gzip
import json
import pandas as pd
from utils import datautils


def gen_mrel_title_wid_mapping(mrel_cands_files, redirects_file, title_wid_file, output_file):
    import re

    with open(redirects_file, encoding='utf-8') as f:
        df = pd.read_csv(f, na_filter=False)
    redirects_dict = {title_from: title_to for title_from, title_to in df.itertuples(False, None)}
    title_wid_dict = datautils.load_title_wid_file(title_wid_file)
    valid_wids = set(title_wid_dict.values())

    mrel_title_wid_dict = dict()
    for mrel_cands_file in mrel_cands_files:
        print(mrel_cands_file)
        f = open(mrel_cands_file, encoding='utf-8')
        for line_idx, line in enumerate(f):
            parts = line.strip().split('\t')
            idx = 6
            if parts[idx] == 'EMPTYCAND':
                continue
            while idx < len(parts) and parts[idx] != 'GT:' and idx - 6 < 30:
                m = re.match('(.*?),(.*?),(.*)', parts[idx])
                wid, title = int(m.group(1)), m.group(3)
                redirected_title = redirects_dict.get(title, None)
                if redirected_title is not None:
                    rwid = title_wid_dict.get(redirected_title, None)
                    if rwid is not None:
                        wid = rwid

                if wid not in valid_wids:
                    wid = title_wid_dict.get(title, None)
                if wid is not None:
                    mrel_title_wid_dict[title] = wid
                idx += 1
        f.close()
    datautils.save_csv(list(mrel_title_wid_dict.items()), ['title', 'wid'], output_file)


def gen_mrel_title_wid_mapping_for_all(mrel_cands_dir, redirects_file, title_wid_file, output_file):
    import os
    cand_files = list()
    for filename in os.listdir(mrel_cands_dir):
        filepath = os.path.join(mrel_cands_dir, filename)
        if os.path.isfile(filepath):
            cand_files.append(filepath)
    gen_mrel_title_wid_mapping(cand_files, redirects_file, title_wid_file, output_file)


def gen_word_freq_vec_pkl(word_freq_file, w2v_file, dim, filter_stop_words, output_file):
    stop_words = None
    if filter_stop_words:
        from utils.stopwords import deeped_stop_words
        stop_words = deeped_stop_words

    with open(word_freq_file, encoding='utf-8') as f:
        df = pd.read_csv(f)
    word_freq_dict = {w: cnt for w, cnt in df.itertuples(False, None)}
    vocab, vecs = list(), list()
    f = open(w2v_file, encoding='utf-8')
    for line in f:
        parts = line.strip().split(' ')
        if len(parts) != dim + 1:
            print(line)
            continue
        # assert(len(parts) == dim + 1)
        if parts[0] not in word_freq_dict or len(parts[0]) < 2:
            continue
        if stop_words is not None and parts[0].lower() in stop_words:
            continue
        vocab.append(parts[0])
        vec = np.asarray([float(v) for v in parts[1:]], np.float32)
        vecs.append(vec)
    f.close()

    freqs = np.asarray([word_freq_dict[w] for w in vocab], np.int32)
    vecs = np.asarray(vecs, np.float32)
    datautils.save_pickle_data((vocab, freqs, vecs), output_file)


def gen_mrel_title_to_wid_file(cands_files, output_file):
    title_wid_dict = dict()
    for cands_file in cands_files:
        cands_objs = datautils.read_json_objs(cands_file)
        for cands_obj in cands_objs:
            for wid, title in zip(cands_obj['wids'], cands_obj['titles']):
                title_wid_dict[title] = wid
    datautils.save_csv(list(title_wid_dict.items()), ['title', 'wid'], output_file)


def filter_wiki_articles(wids, word_to_id_dict, articles_file, output_file):
    f = gzip.open(articles_file, 'rt', encoding='utf-8')
    fout = open(output_file, 'w', encoding='utf-8', newline='\n')
    for i, line in enumerate(f):
        if i % 100000 == 0:
            print(i)
        x = json.loads(line)
        if x['wid'] not in wids:
            continue

        words = list()
        for sent in x['sents']:
            sent_words = sent.split(' ')
            for w in sent_words:
                word_id = word_to_id_dict.get(w, None)
                if word_id is not None:
                    words.append(word_id)
        fout.write('{}\n'.format(json.dumps({'wid': x['wid'], 'word_ids': words})))
    f.close()
    fout.close()


def filter_anchor_cxt(wids, word_to_id_dict, anchor_cxt_file, output_file):
    f = gzip.open(anchor_cxt_file, 'rt', encoding='utf-8')
    fout = open(output_file, 'w', encoding='utf-8', newline='\n')
    for i, line in enumerate(f):
        if i % 1000000 == 0:
            print(i)
        # if i > 100:
        #     break
        x = json.loads(line)
        target_wid = x['target_wid']
        if target_wid not in wids:
            continue

        word_ids = list()
        for w in x['cxt'].split(' '):
            word_id = word_to_id_dict.get(w, -1)
            # if word_id == -1:
            #     print(w)
            word_ids.append(word_id)

        fout.write('{}\n'.format(json.dumps(
            {'target_wid': target_wid, 'word_ids': word_ids, 'beg_idx': x['beg_idx'], 'end_idx': x['end_idx']})))
        # if i > 100:
        #     break
    f.close()
    fout.close()


def filter_wiki_for_training(required_title_wid_file, word_freq_vec_pkl, articles_file, anchor_cxt_file,
                             output_articles_file, output_anchor_cxt_file):
    df = datautils.load_csv(required_title_wid_file)
    wids = {wid for _, wid in df.itertuples(False, None)}

    word_vocab, freqs, word_vecs = datautils.load_pickle_data(word_freq_vec_pkl)
    word_to_id_dict = {w: i for i, w in enumerate(word_vocab)}

    filter_wiki_articles(wids, word_to_id_dict, articles_file, output_articles_file)
    filter_anchor_cxt(wids, word_to_id_dict, anchor_cxt_file, output_anchor_cxt_file)


def gen_full_embed_title_wid_file(mrel_title_wid_file, core_title_wids_file, all_title_wid_file, output_file):
    wid_title_dict = datautils.load_title_wid_file(all_title_wid_file, to_wid_title_dict=True)
    df = datautils.load_csv(mrel_title_wid_file)
    embed_wid_title_dict = {wid: wid_title_dict[wid] for _, wid in df.itertuples(False, None)}
    df = datautils.load_csv(core_title_wids_file)
    for title, wid in df.itertuples(False, None):
        embed_wid_title_dict[wid] = title
    embed_title_wid_tups = [(title, wid) for wid, title in embed_wid_title_dict.items()]
    datautils.save_csv(embed_title_wid_tups, ['title', 'wid'], output_file)


def __read_mrel_ent_vocab_file(filename):
    wiki_url_prefix = 'en.wikipedia.org/wiki/'
    entity_names = list()
    f = open(filename, encoding='utf-8')
    for line in f:
        parts = line.strip().split('\t')
        entity_name = parts[0][len(wiki_url_prefix):].replace('_', ' ').replace('%22', '"')
        entity_names.append(entity_name)
    f.close()
    return entity_names


def replace_with_provided_ent_embed(provided_ent_vocab_file, provided_ent_embed_file, mrel_title_to_wid_file,
                                    self_trained_ent_embed_pkl, output_file):
    df = datautils.load_csv(mrel_title_to_wid_file)
    mrel_title_wid_dict = {title: wid for title, wid in df.itertuples(False, None)}
    mrel_entity_names = __read_mrel_ent_vocab_file(provided_ent_vocab_file)
    mrel_entity_vecs = np.load(provided_ent_embed_file)
    assert mrel_entity_vecs.shape[0] == len(mrel_entity_names)

    mrel_wids = [mrel_title_wid_dict.get(name, -1) for name in mrel_entity_names]
    wid_vocab, entity_vecs = datautils.load_pickle_data(self_trained_ent_embed_pkl)
    wid_eid_dict = {wid: i for i, wid in enumerate(wid_vocab)}
    extra_entity_vecs = list()
    for wid, mrel_entity_vec in zip(mrel_wids, mrel_entity_vecs):
        if wid > -1:
            eid = wid_eid_dict.get(wid, None)
            if eid is not None:
                entity_vecs[eid] = mrel_entity_vec
            else:
                extra_entity_vecs.append(mrel_entity_vec)
                wid_vocab.append(wid)
    new_entity_vecs = np.zeros((entity_vecs.shape[0] + len(extra_entity_vecs), entity_vecs.shape[1]), np.float32)
    for i, vec in enumerate(entity_vecs):
        new_entity_vecs[i] = vec
    for i, vec in enumerate(extra_entity_vecs):
        new_entity_vecs[i + entity_vecs.shape[0]] = vec
    datautils.save_pickle_data((wid_vocab, new_entity_vecs), output_file)
