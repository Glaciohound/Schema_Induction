import argparse

from load_muc4 import load_muc4
from muc4_tools import print_contents


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/muc34/TASK/CORPORA", type=str)
    args = parser.parse_args()
    return args


def main(args):
    dev_corpora, dev_events, tst_corpora, tst_events = \
        load_muc4(args.dir)
    all_types = ["ATTACK", "BOMBING", "KIDNAPPING", "ARSON", "ROBBERY"]
    dev_corpora_dict = {_doc["title"]: _doc["content"] for _doc in dev_corpora}
    dev_grouped = {
        _type: list(filter(
            lambda x: x["INCIDENT: TYPE"][0] == _type,
            dev_events))
        for _type in all_types}
    for _type, _samples in dev_grouped.items():
        print(_type)
        print("=" * 30)
        for _sample in _samples[:5]:
            print(_sample)
            print_contents(dev_corpora_dict[_sample["MESSAGE: ID"]],
                           cased=True)
            print("-" * 30)
            print()
        print()


if __name__ == "__main__":
    args = get_args()
    main(args)
