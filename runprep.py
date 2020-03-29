import os
import config
from prep import entembprep

title_wid_file = os.path.join(config.DATA_DIR, 'mrel-title-to-wid-aidatac.txt')
aidatac_wiki_toktext_file = os.path.join(config.DATA_DIR, 'aidatac_wiki_toktext.txt')
aidatac_anchor_cxt_file = os.path.join(config.DATA_DIR, 'aidatac_anchor_cxt.txt')
entembprep.filter_wiki_for_training(
    title_wid_file, config.WORD_FREQ_VEC_PKL, config.WIKI_TOKTEXT_FILE, config.WIKI_ANCHOR_CXT_FILE,
    aidatac_wiki_toktext_file, aidatac_anchor_cxt_file)
