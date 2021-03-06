import json
from collections import Counter, defaultdict

from components.load_muc4 import load_muc4
from components.muc4_tools import get_all_sentences
from components.get_args import get_args
from components.logging import logger


def dual_output(line, f):
    logger.info(line, level_offset=1)
    f.write(str(line) + "\n")


def main(args):
    dev_corpora, dev_events, tst_corpora, tst_events, proper_nouns = \
        load_muc4(args.data_dir, args.loading_cache_file,
                  args.overwrite_loading)
    dev_sentences = get_all_sentences(dev_corpora)
    tst_sentences = get_all_sentences(tst_corpora)
    all_sentences = dict(
        list(dev_sentences.items()) + list(tst_sentences.items()))
    with open(args.all_sentences_file, 'w') as f:
        json.dump(dict(zip(map(str, all_sentences.keys()),
                           all_sentences.values())), f, indent=4)

    with open(args.statistics_file, "w") as f:
        for corpora, events, sentences, name in [
                (dev_corpora, dev_events, dev_sentences, "DEV"),
                (tst_corpora, tst_events, tst_sentences, "TST")]:
            dual_output(f"======== In {name} set ==========", f)
            dual_output(f"Number of docs: {len(corpora)}, ", f)
            dual_output(f"Number of events: {len(events)}", f)
            dual_output(f"Number of sentences: {len(sentences)}", f)

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
