import socket
from os.path import join

RES_DIR = '/home/data/hldai/res'
OUTPUT_DIR = '/home/data/hldai/el/deepeddata'
DATA_DIR = '/home/data/hldai/el/deepeddata'
LOG_DIR = join(OUTPUT_DIR, 'log')

MACHINE_NAME = socket.gethostname()

WIKI_TOKTEXT_FILE = join(RES_DIR, 'wiki/enwiki-20151002-pages-articles-toktext.txt.gz')
WIKI_ANCHOR_CXT_FILE = join(RES_DIR, 'wiki/enwiki-20151002-anchor-cxt.json.gz')
WORD_FREQ_VEC_PKL = join(DATA_DIR, 'enwiki-20151002-word-freq-w2v.pkl')
