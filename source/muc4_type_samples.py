import argparse
import json

from components.load_muc4 import load_muc4
from components.muc4_tools import \
    extract_relevant_sentences, corpora_to_dict, all_types


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/muc34/TASK/CORPORA", type=str)
    parser.add_argument("--output-file", type=str,
                        default="data/muc34/outputs/muc4-type-samples.txt")
    args = parser.parse_args()
    return args


def main(args):
    dev_corpora, dev_events, tst_corpora, tst_events, proper_nouns = \
        load_muc4()
    dev_corpora_dict = corpora_to_dict(dev_corpora)
    dev_grouped = {
        _type: list(filter(
            lambda x: x["INCIDENT: TYPE"][0] == _type,
            dev_events))
        for _type in all_types}
    with open(args.output_file, 'w') as f:
        for _type, _samples in dev_grouped.items():
            f.write(_type)
            f.write("\n")
            f.write("=" * 30)
            f.write("\n")
            for _sample in _samples[:30]:
                f.write(json.dumps(_sample, indent=4))
                f.write("\n")
                f.write(json.dumps(
                    extract_relevant_sentences(
                        _sample, dev_corpora_dict[_sample["MESSAGE: ID"]]),
                    indent=4
                )[0])
                f.write("\n")
                f.write("-" * 30)
                f.write("\n")
            f.write("\n")


if __name__ == "__main__":
    args = get_args()
    main(args)
