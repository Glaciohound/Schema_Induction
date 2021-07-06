import argparse

from load_muc4 import load_muc4


def get_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


def main(args):
    dev_corpora, dev_events, tst_corpora, tst_events, proper_nouns = \
        load_muc4()


if __name__ == "__main__":
    args = get_args()
    main(args)
