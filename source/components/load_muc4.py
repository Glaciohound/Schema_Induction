import os
import re
import copy
# import pickle
import json
import string
import itertools
from glob import glob
from collections import defaultdict
from tqdm import tqdm


def readlines(filename, strip=True):
    with open(filename, 'r') as input_file:
        all_lines = list(map(lambda x: x.strip() if strip else x,
                             input_file.readlines()))
    return all_lines


def read_muc_article(all_lines, proper_nouns):
    output = []
    punc = f"[ {string.punctuation}]"
    proper_nouns_list = list(proper_nouns.keys())
    proper_nouns_list.sort(reverse=True)
    proper_nouns_pattern = (
        punc + "(" +
        "|".join(map(lambda x: x.lower(), proper_nouns_list)) +
        ")" + punc
    )

    def post_process_article(article):
        if article["content"][-1].strip() == "":
            article["content"].pop()
        article["content-cased-split"] = []
        for _paragraph in article["content"]:
            _paragraph = _paragraph.lower()
            for _ in range(2):
                _paragraph = re.sub(
                    r"{}".format(proper_nouns_pattern),
                    lambda x: x.group(0)[0] +
                    proper_nouns.get(
                        x.group(0)[1:-1].upper(),
                        {"cased": x.group(0)[1:-1]}
                    )["cased"] +
                    x.group(0)[-1],
                    _paragraph,
                )
            _paragraph = re.sub(
                r"([^A-Z][\.\]]\"* +\"*|\[|^ *)[a-z]",
                lambda x: x.group(0)[:-1] + x.group(0)[-1].upper(),
                _paragraph
            )
            _paragraph_split = []
            last_split_pos = 0
            for _match in re.finditer(
                r"([^A-Z\.]\.[\" ]*([A-Z\[\(]|$)|$)",
                _paragraph
            ):
                start = _match.start()
                if _paragraph[start-1:start+1] in ("Mr", "Dr") or \
                        _paragraph_split[start-2:start+1] in ("Mrs",):
                    continue
                split_pos = _match.end() - 1

                def not_contains_alpha(x):
                    return x.lower() == x.upper()
                if not_contains_alpha(_paragraph[last_split_pos: split_pos]):
                    continue
                if split_pos != len(_paragraph) - 1:
                    while _paragraph[split_pos] not in ". " and \
                            split_pos > last_split_pos:
                        split_pos -= 1
                assert split_pos != last_split_pos, _paragraph
                split_pos += 1
                _paragraph_split.append(_paragraph[last_split_pos:split_pos])
                last_split_pos = split_pos
            if _paragraph_split != []:
                article["content-cased-split"].append(_paragraph_split)

    default_article = {
        "title": "",
        "content": [""],
    }
    article = copy.deepcopy(default_article)

    for _line in tqdm(all_lines):
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
            article["content"][-1] += " " + _line
    post_process_article(article)
    output.append(article)
    return output


def read_muc_event(filename: str):
    all_lines = readlines(filename, strip=False)
    output = []

    def post_process_event(event):
        event["MESSAGE: ID"] = event["MESSAGE: ID"][0].split(' ')[0]
        assert len(event) == 25, list(enumerate(event))
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


def proper_noun_to_cased(name):
    abbr = (
        "USSR",
        "FRG",
        "GDR",
        "ROC",
        "U.S.", "U. S.", "USA",
        "UCA", "UES"
    )
    unimportant_case_words = (
        "of", "and",
        "de", "la", "el", "del", "a"
    )
    if name in abbr:
        return name
    else:
        for _unimportant in unimportant_case_words:
            name = re.sub(
                r" " + _unimportant.upper() + r"( |$)",
                lambda x: x.group(0).lower(),
                name
            )
        name = re.sub(
            r"[A-Z]{2}",
            lambda x: x.group(0)[0] + x.group(0)[1].lower(),
            name
        )
        name = re.sub(
            r"[a-z][A-Z]",
            lambda x: x.group(0).lower(),
            name
        )
    return name


def recognize_proper_noun(_line, default_type):
    is_synonym = " = " in _line or " -> " in _line
    is_new = False
    if _line.endswith(" <-NEW"):
        is_new = True
        _line = _line.replace(" <-NEW", "")
    if is_synonym:
        if " = " in _line:
            name, original = _line.split(" = ")
        else:
            name, original = _line.split(" -> ")
        return name, {
            "type": default_type or None,
            "cased": proper_noun_to_cased(name),
            "original": original,
            "is-new": is_new,
        }

    else:
        _line = _line.split(": ")[-1]
        origin_match = re.search(r" \*[a-z ]*\*", _line)
        if origin_match is not None:
            _line = _line[:origin_match.start()]
            _origin = origin_match.group(0)[2:-1]
        else:
            _origin = None
        type_match = re.search(r" (\(|\[).*(\)|\])", _line)
        if type_match is not None:
            _type = type_match.group(0)[2:-1]
            _line = _line[:type_match.start()]
        else:
            _type = default_type
        name = _line.strip()
        return name, {
            "type": _type.lower(),
            "cased": proper_noun_to_cased(name),
            "is-new": is_new,
            "origin": _origin
        }


def recognize_proper_nouns(lines, default_type=None):
    output = dict()
    for _line in lines:
        if "+ name" in _line or _line == "" or _line.startswith("#"):
            continue
        name, content = recognize_proper_noun(_line, default_type)
        output[name] = content
    return output


def load_muc4(
    muc4_dir="data/muc34",
    cache_file="data/muc34/outputs/muc4_loaded_cache.json",
    overwrite=False
):
    if cache_file is not None and os.path.exists(cache_file) and not overwrite:
        with open(cache_file, 'r') as f:
            all_data = json.load(f)
        return all_data

    loc_lines, loc_syn_lines, bld_lines, int_lines, str_lines, \
        nation_lines, other_lines \
        = tuple(map(lambda x: readlines(
            os.path.join(muc4_dir, "TASK/TASKDOC", x)), (
                "set-list-location.v4", "places-synonyms.v3",
                "places-buildings-etc.v3", "places-international.v3",
                "places-streets-etc.v2",
                "set-list-foreign-nation.v5",
                "other-proper-nouns.chi",
            )))
    proper_nouns = recognize_proper_nouns(
        nation_lines[22: 98] + nation_lines[102: 121], "nation")
    proper_nouns.update(recognize_proper_nouns(
        loc_lines[55: 1656] + loc_syn_lines[25: 82], "location"
    ))
    proper_nouns.update(recognize_proper_nouns(
        bld_lines[18: 82], "building"
    ))
    proper_nouns.update(recognize_proper_nouns(
        int_lines[14: 31], "international"
    ))
    proper_nouns.update(recognize_proper_nouns(
        str_lines[9: 83], "street"
    ))
    proper_nouns.update(recognize_proper_nouns(
        other_lines, "other"
    ))

    filelist = glob(os.path.join(muc4_dir, "TASK/CORPORA/*"))
    dev_corpora_files = list(filter(lambda x: "dev-muc3" in x, filelist))
    dev_events_files = list(filter(lambda x: "key-dev" in x, filelist))
    tst_corpora_files = list(filter(
        lambda x: "tst" in x and "muc" in x, filelist
    ))
    tst_events_files = list(filter(lambda x: "key-tst" in x, filelist))
    dev_corpora_files.sort()
    dev_corpora_lines = list(itertools.chain(
        *list(map(readlines, dev_corpora_files))
    ))
    dev_events_files.sort()
    tst_corpora_files.sort()
    tst_corpora_lines = list(itertools.chain(
        *list(map(readlines, tst_corpora_files))
    ))
    tst_events_files.sort()
    dev_corpora = read_muc_article(dev_corpora_lines, proper_nouns)
    dev_events = list(itertools.chain(
        *[read_muc_event(_filename)
          for _filename in dev_events_files]))
    for i, _event in enumerate(dev_events):
        _event["event-ID"] = f"DEV-EVENTS-{i}"
    tst_corpora = read_muc_article(tst_corpora_lines, proper_nouns)
    tst_events = list(itertools.chain(
        *[read_muc_event(_filename)
          for _filename in tst_events_files]))
    for i, _event in enumerate(tst_events):
        _event["event-ID"] = f"TST-EVENTS-{i}"
    all_data = dev_corpora, dev_events, tst_corpora, tst_events, proper_nouns

    if overwrite or \
            (cache_file is not None and not os.path.exists(cache_file)):
        with open(cache_file, 'w') as f:
            json.dump(all_data, f, indent=2)
    return all_data
