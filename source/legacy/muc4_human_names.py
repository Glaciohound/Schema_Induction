import argparse

from components.muc4_tools import get_event_keywords
from components.load_muc4 import load_muc4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-human-names", type=str,
                        default="data/muc34/outputs/all-human-names.txt")
    args = parser.parse_args()
    return args


def main(args):

    _, dev_events, _, tst_events, _ = load_muc4()
    keywords = list(map(get_event_keywords, dev_events)) +\
        list(map(get_event_keywords, tst_events))
    human_names = [
        _name
        for _event_keywords in keywords
        for _name, _category in _event_keywords.items()
        if _category.endswith("NAME")
        # for _subname in _name.split(" ")
    ]
    human_names = list(set(human_names))

    with open(args.all_human_names, 'w') as f:
        f.writelines(
            list(map(lambda x: x + "\n", human_names))
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
