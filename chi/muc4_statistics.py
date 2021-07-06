import argparse
from collections import Counter, defaultdict
import numpy as np

from load_muc4 import load_muc4


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/muc34", type=str)
    parser.add_argument(
        "--cache-file", default="data/muc34/TASK/muc4_loaded_cache.pkl")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return args


def main(args):
    dev_corpora, dev_events, tst_corpora, tst_events, proper_nouns = \
        load_muc4(args.dir, args.cache_file, args.overwrite)

    print("========Examples=========")
    print(np.random.choice(dev_corpora, 2).tolist())
    print()

    for corpora, events, name in [
            (dev_corpora, dev_events, "DEV"),
            (tst_corpora, tst_events, "TST")]:
        print(f"======== In {name} set ==========")
        print(f"Number of docs: {len(corpora)}, ")
        print(f"Number of events: {len(events)}")

        event_count = defaultdict(lambda: 0)
        for doc in corpora:
            event_count[doc["title"]] = 0
        for _event in events:
            event_count[_event["MESSAGE: ID"]] += 1
        event_count = Counter(list(event_count.values()))
        print("Counting events per document:",
              dict(sorted(event_count.items())))
        type_count = Counter(list(map(lambda x: x["INCIDENT: TYPE"][0],
                                      events)))
        print("Counting incident types:", type_count)
        print()


if __name__ == "__main__":
    args = get_args()
    main(args)
