def parse_idx_device_args():
    import argparse

    parser = argparse.ArgumentParser(description='dhl')
    parser.add_argument('idx', type=int, default=0, nargs='?')
    parser.add_argument('-d', type=int, default=[], nargs='+')
    return parser.parse_args()


def json_objs_to_kvlistdict(objs, key_str):
    d = dict()
    for x in objs:
        cur_key = x[key_str]
        cur_key_objs = d.get(cur_key, list())
        if not cur_key_objs:
            d[cur_key] = cur_key_objs
        cur_key_objs.append(x)
    return d


def json_objs_to_kvdict(objs, key_str):
    return {x[key_str]: x for x in objs}
