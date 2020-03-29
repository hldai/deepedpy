import os
import torch
import datetime
import random
import numpy as np
from utils import datautils, utils
from utils.loggingutils import init_universal_logging
from exp import entembexp
import config


def wid_vocab_from_title_wid_file(title_wid_file):
    title_wid_df = datautils.load_csv(title_wid_file)
    return [wid for _, wid in title_wid_df.itertuples(False, None)]


def wid_vocab_from_wids_file(wids_file):
    with open(wids_file, encoding='utf-8') as f:
        return [int(line.strip()) for line in f]


def __train():
    n_words_per_ent = 20
    batch_size = 500
    n_desc_iter = 400
    lr = 0.3
    torch.manual_seed(3572)
    np.random.seed(779)
    random.seed(1772)

    title_wid_file = os.path.join(config.DATA_DIR, 'mrel-title-to-wid-aidatac.txt')
    articles_file = os.path.join(config.DATA_DIR, 'aidatac_wiki_toktext.txt')
    anchor_cxt_file = os.path.join(config.DATA_DIR, 'aidatac_anchor_cxt.txt')
    output_entity_vecs_file = os.path.join(config.DATA_DIR, 'entity-vecs-aidatac-tmp.pkl')

    entembexp.traindeschyp(device, config.WORD_FREQ_VEC_PKL, title_wid_file, articles_file, anchor_cxt_file,
                           n_words_per_ent, batch_size=batch_size, lr=lr, output_file=output_entity_vecs_file,
                           n_desc_iter=n_desc_iter)


if __name__ == '__main__':
    # log_file = None
    str_today = datetime.date.today().strftime('%y-%m-%d')
    log_file = '{}/{}-{}-{}.log'.format(config.LOG_DIR, os.path.splitext(os.path.basename(__file__))[0],
                                        str_today, config.MACHINE_NAME)
    init_universal_logging(log_file, mode='a', to_stdout=True)

    args = utils.parse_idx_device_args()

    cuda_device_str = 'cuda' if len(args.d) == 0 else 'cuda:{}'.format(args.d[0])
    device = torch.device(cuda_device_str) if torch.cuda.device_count() > 0 else torch.device('cpu')
    device_ids = args.d
    __train()
