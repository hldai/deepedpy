import pandas as pd
import pickle
import json


def load_title_wid_file(filename, to_wid_title_dict=False):
    with open(filename, encoding='utf-8') as f:
        df = pd.read_csv(f, na_filter=False)
    if to_wid_title_dict:
        return {wid: title for title, wid in df.itertuples(False, None)}
    return {title: wid for title, wid in df.itertuples(False, None)}


def load_csv(file, na_filter=True):
    with open(file, encoding='utf-8') as f:
        return pd.read_csv(f, na_filter=na_filter)


def save_csv(data, columns, output_file):
    with open(output_file, 'w', encoding='utf-8', newline='\n') as fout:
        pd.DataFrame(data, columns=columns).to_csv(fout, index=False, line_terminator='\n')


def load_pickle_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_pickle_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def save_json_objs(objs, output_file):
    with open(output_file, 'w', encoding='utf-8', newline='\n') as fout:
        for v in objs:
            fout.write('{}\n'.format(json.dumps(v, ensure_ascii=False)))


def read_json_objs(filename):
    objs = list()
    with open(filename, encoding='utf-8') as f:
        for line in f:
            objs.append(json.loads(line))
    return objs
