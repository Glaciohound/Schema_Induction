import os
import copy
import itertools
from glob import glob
from collections import defaultdict


def read_muc_file(filename: str, filetype: str):
    with open(filename, 'r') as f:
        all_lines = f.readlines()
    output = []

    if filetype == 'corpora':
        def post_process_article(article):
            if article["content"][-1].strip() == "":
                article["content"].pop()

        default_article = {
            "title": "",
            "content": [""],
        }
        article = copy.deepcopy(default_article)

        for _line in all_lines:
            _line = _line.strip()
            if _line == "" and article["title"] == "":
                continue
            elif _line.startswith("DEV-MUC") or _line.startswith("TST1-MUC"):
                if article["title"] != "":
                    post_process_article(article)
                    output.append(article)
                    article = copy.deepcopy(default_article)
                article['title'] = _line.strip().split(' ')[0]
            elif _line == "":
                if article["content"][-1] != "":
                    article["content"].append("")
            else:
                article["content"][-1] += _line
        post_process_article(article)
        output.append(article)

    elif filetype == "events":
        def post_process_event(event):
            event["MESSAGE: ID"] = event["MESSAGE: ID"][0].split(' ')[0]
            assert len(event) == 25, event
        event = defaultdict(list)
        last_key = ""
        content_offset = 36

        for _line in all_lines:
            _line = _line.rstrip()
            if _line == "":
                if len(event.keys()) != 0:
                    post_process_event(event)
                    output.append(event)
                event = defaultdict(list)
                last_key = ""
            elif _line.startswith(";"):
                continue
            elif _line[0] in map(lambda x: str(x), range(10)):
                if "MESSAGE: ID" in _line:
                    content_offset = 15 + len(_line[15:]) - \
                        len(_line[15:].lstrip())
                last_key = _line[4:content_offset].rstrip()
                event[last_key].append(_line[content_offset:].lstrip())
            else:
                assert last_key != "", filename + _line
                event[last_key].append(_line[content_offset:].lstrip())
        if len(event.keys()) != 0:
            post_process_event(event)
            output.append(event)

    return output


def load_muc4(muc4_dir):
    filelist = glob(os.path.join(muc4_dir, "*"))
    dev_corpora_files = list(filter(lambda x: "dev-muc3" in x, filelist))
    dev_events_files = list(filter(lambda x: "key-dev" in x, filelist))
    tst_corpora_files = list(filter(
        lambda x: "tst" in x and "muc" in x,
        filelist
    ))
    tst_events_files = list(filter(lambda x: "key-tst" in x, filelist))
    dev_corpora_files.sort()
    dev_events_files.sort()
    tst_corpora_files.sort()
    tst_events_files.sort()
    dev_corpora = list(itertools.chain(
        *[read_muc_file(_filename, "corpora")
          for _filename in dev_corpora_files]))
    dev_events = list(itertools.chain(
        *[read_muc_file(_filename, "events")
          for _filename in dev_events_files]))
    tst_corpora = list(itertools.chain(
        *[read_muc_file(_filename, "corpora")
          for _filename in tst_corpora_files]))
    tst_events = list(itertools.chain(
        *[read_muc_file(_filename, "events")
          for _filename in tst_events_files]))
    return dev_corpora, dev_events, tst_corpora, tst_events
