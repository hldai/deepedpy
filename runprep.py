import os
import config
from prep import entembprep
from utils import datautils


def gen_candidates(wcwy_pem_file, title_wid_file):
    print(f'loading {title_wid_file} ...')
    wid_title_dict = {wid: title for title, wid in datautils.load_csv(title_wid_file).itertuples(False, None)}
    print(f'loading {wcwy_pem_file} ...')
    mstr_pem_dict = datautils.load_pickle_data(wcwy_pem_file)

    wid_probs = mstr_pem_dict['Japan']

    # wid_prob_tups = [(wid, prob) for wid, prob in wid_probs.items()]
    # wid_prob_tups.sort(key=lambda x: -x[1])
    print(len(wid_probs))
    for wid, prob in wid_probs[:10]:
        print(wid, wid_title_dict.get(wid), prob)


wcwy_pem_file = os.path.join(config.DATA_DIR, 'w15cwy-mstr-wid-pems.pkl')
title_wid_file = os.path.join(config.DATA_DIR, 'enwiki-20151002-title-wid.txt')
# gen_candidates(wcwy_pem_file, title_wid_file)

needed_title_wid_file = os.path.join(config.DATA_DIR, 'mrel-title-to-wid-aidatac.txt')
aidatac_wiki_toktext_file = os.path.join(config.DATA_DIR, 'aidatac_wiki_toktext.txt')
aidatac_anchor_cxt_file = os.path.join(config.DATA_DIR, 'aidatac_anchor_cxt.txt')
entembprep.filter_wiki_for_training(
    needed_title_wid_file, config.WORD_FREQ_VEC_PKL, config.WIKI_TOKTEXT_FILE, config.WIKI_ANCHOR_CXT_FILE,
    aidatac_wiki_toktext_file, aidatac_anchor_cxt_file)
