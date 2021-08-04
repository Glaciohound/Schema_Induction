import re
import json
import itertools
from collections import defaultdict
import math
import numpy as np

from components.constants import \
    all_types, months, interested_categories
from components.logic_tools import sort_rank


def get_all_sentences(corpora):
    if isinstance(corpora, dict):
        corpora = corpora.values()
    output = [
        _sentence
        for _article in corpora
        for _paragraph_split in _article["content-cased-split"]
        for _sentence in _paragraph_split
    ]
    return output


def get_all_paragraphs(corpora):
    if isinstance(corpora, dict):
        corpora = corpora.values()
    output = [
        "".join(_paragraph_split)
        for _article in corpora
        for _paragraph_split in _article["content-cased-split"]
    ]
    return output


def convert_date(date):
    output = [date.lstrip("0")]
    if len(date) > 6:
        _day, _month, _ = date.split(" ")
        output.append(
            _day.lstrip('0') + " " +
            months[_month.upper()[:3]].upper()
        )
    else:
        _day, _month = date.split(" ")
        if _day.isalpha():
            _day, _month = _month, _day
        output.append(_day.lstrip('0') + " " + months[_month].upper())
    return output


def get_event_keywords(event):
    typed_keywords = {}
    punctuation = " !\"#$%&\'*+,-./:;<=>?@\\^_`{|}~"
    null_arguments = ["", "-", "*"]
    for _type, _list in event.items():
        if _type not in interested_categories:
            continue
        for _piece in _list:
            _arguments = []
            for _split in _piece.split("/"):
                if not (_type.endswith("DATE") or _type.endswith("LOCATION"))\
                        and _split not in null_arguments:
                    _arguments.append(
                        _split.strip(punctuation)
                    )
                elif _piece not in ["", "-", "*"]:
                    if _type.endswith("LOCATION"):
                        _arguments.extend(list(
                            re.sub(r"\([ A-Z]*\)", "",
                                   _level.split('-')[0]).strip(" ")
                            for _level in _split.split(": ")
                        ))
                    elif _type.endswith("DATE"):
                        full_date = _split.strip()
                        _arguments.append(full_date)
                        for _single_date in full_date.strip("()").split('-'):
                            _single_date = _single_date.strip("- ")
                            if _single_date != "":
                                _arguments.extend(convert_date(_single_date))
                else:
                    continue
            if len(_arguments) == 1:
                typed_keywords[_arguments[0]] = _type
            elif len(_arguments) > 1:
                typed_keywords[tuple(_arguments)] = _type

    return typed_keywords


def extract_relevant_sentences(event, article,
                               content_arg="content-cased-split"):
    all_sentences = list(
        itertools.chain(*[
            _paragraph_split
            for _paragraph_split in article[content_arg]
        ]))
    keywords = get_event_keywords(event)
    counter = defaultdict(lambda: {"count": 0, "arguments": {}})
    missing_counter = [True] * len(keywords)
    for i, sentence in enumerate(all_sentences):
        sentence_upper = sentence.upper()
        for j, (keyword, _category) in enumerate(keywords.items()):
            if isinstance(keyword, tuple):
                sub_pattern = "(" + "|".join(map(re.escape, keyword)) + ")"
            else:
                sub_pattern = re.escape(keyword)
            pattern = "(^|[^A-Z])" + sub_pattern + "($|[^A-Z])"
            _match = re.search(pattern, sentence_upper)
            if _match is not None:
                missing_counter[j] = False
                counter[i]["count"] += len(re.findall(pattern, sentence_upper))
                submatch = re.search(sub_pattern, _match.group(0).upper())
                start_pos = _match.start() + submatch.start()
                counter[i]["arguments"][
                    sentence[start_pos: start_pos + len(submatch.group(0))]
                ] = _category
                counter[i]["index"] = i
                counter[i]["sentence"] = all_sentences[i]
                counter[i]["article"] = event["MESSAGE: ID"]
    counter = sorted(list(counter.values()), key=lambda x: x["count"],
                     reverse=True)
    missing_counter = dict(zip(keywords, missing_counter))
    return counter, missing_counter


def extract_relevant_sentences_from_events(events, corpora, content_arg):
    if isinstance(corpora, list):
        corpora = corpora_to_dict(corpora)
    all_results = [
        extract_relevant_sentences(_event, corpora[_event["MESSAGE: ID"]],
                                   content_arg)
        for _event in events
    ]
    output = [_sentence for _event in all_results for _sentence in _event[0]]
    # missing = np.concatenate(
    #     [list(_event[1].values()) for _event in all_results]
    # )
    # missing_counter = {
    #     "total": missing.shape[0],
    #     "missing": missing.sum()
    # }
    # print(missing_counter)
    return output


def corpora_to_dict(corpora):
    return {
        _article["title"]: _article
        for _article in corpora
    }


def merge_sentences_to_paragraph(corpora):
    iterable = corpora if isinstance(corpora, list) else corpora.values()
    for article in iterable:
        article["content-split"] = [
            "".join(_paragraph)
            for _paragraph in article["content-split-cased"]
        ]


def rebalance_by_weight(rank, selected_names, all_names_index):
    output = defaultdict(lambda: 0)
    for _word, _weight in rank.items():
        type_word = all_names_index.get(_word, None)
        balance_weight = (
            math.pow(selected_names[type_word]['weight'], 1/2)
            if type_word in selected_names else 1
        )
        output[type_word] += _weight / balance_weight
    output = sort_rank(output)
    if len(output) == 0:
        output = {None: 0}
    elif len(output) > 1:
        output.pop(None)
    # elif len(output) > 2:
    #     weights = list(output.values())
    #     if weights[1] > weights[2] * 2:
    #         output.pop(None)
    return output


def group_events_by_type(all_events):
    type_category = "INCIDENT: TYPE"
    groupby_type = {
        _type: list(filter(
            lambda x: x[type_category][0] == _type,
            all_events
        ))
        for _type in all_types
    }
    return groupby_type


def get_all_type_sentences(all_events, corpora, content_arg):
    if isinstance(corpora, list):
        corpora = corpora_to_dict(corpora)
    groupby_type = group_events_by_type(all_events)
    type_sentences = {
        _type: [
            extract_relevant_sentences(
                _event, corpora[_event["MESSAGE: ID"]], content_arg
            )[0]
            for _event in _group
        ]
        for _type, _group in groupby_type.items()
    }
    return type_sentences


def calculate_precision_recall(pred_list, gt_list):
    precision_recall = {}
    total_hit = 0
    for _type in all_types:
        retrieved = pred_list[_type]
        gt = gt_list[_type]
        hit = np.intersect1d(retrieved, gt)
        total_hit += hit.shape[0]
        precision_recall[_type] = {
            "precision": hit.shape[0] / retrieved.shape[0],
            "recall": hit.shape[0] / gt.shape[0],
            "F1": hit.shape[0] / (gt.shape[0] + retrieved.shape[0]) * 2
        }
    total_pred = sum(map(len, pred_list.values()))
    total_gt = sum(map(len, gt_list.values()))
    precision_recall["total"] = {
        "precision": total_hit / total_pred,
        "recall": total_hit / total_gt,
        "F1": total_hit / (total_pred + total_gt) * 2,
    }
    return precision_recall


def load_selected_names(selected_names_file, all_types):
    manual_synonym = {
        _keywords[0]: _keywords[1:]
        for _keywords in all_types.values()
        if len(_keywords) > 1
    }
    with open(selected_names_file, 'r') as f:
        selected_names = json.load(f)
    for seed, synset in manual_synonym.items():
        for _synonym in synset:
            if _synonym in selected_names:
                _synonym_content = selected_names.pop(_synonym)
                selected_names[seed]["lemma_names"].extend(
                    _synonym_content["lemma_names"]
                )
                selected_names[seed]["weight"] += _synonym_content["weight"]
    return selected_names
