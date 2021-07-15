import argparse
from collections import Counter, defaultdict
# import numpy as np

from load_muc4 import load_muc4


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/muc34", type=str)
    parser.add_argument(
        "--cache-file", default="data/muc34/outputs/muc4_loaded_cache.pkl")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-file", type=str,
                        default="data/muc34/outputs/muc4_statistics.txt")
    args = parser.parse_args()
    return args


def dual_output(line, f):
    print(line)
    f.write(str(line) + "\n")


def main(args):
    dev_corpora, dev_events, tst_corpora, tst_events, proper_nouns = \
        load_muc4(args.dir, args.cache_file, args.overwrite)

    with open(args.output_file, "w") as f:
        for corpora, events, name in [
                (dev_corpora, dev_events, "DEV"),
                (tst_corpora, tst_events, "TST")]:
            dual_output(f"======== In {name} set ==========", f)
            dual_output(f"Number of docs: {len(corpora)}, ", f)
            dual_output(f"Number of events: {len(events)}", f)

            event_count = defaultdict(lambda: 0)
            for doc in corpora:
                event_count[doc["title"]] = 0
            for _event in events:
                event_count[_event["MESSAGE: ID"]] += 1
            event_count = Counter(list(event_count.values()))
            dual_output("Counting events per document:", f)
            dual_output(dict(sorted(event_count.items())), f)
            type_count = Counter(list(map(lambda x: x["INCIDENT: TYPE"][0],
                                          events)))
            dual_output("Counting incident types:" + str(type_count), f)
            dual_output("", f)


if __name__ == "__main__":
    args = get_args()
    main(args)
