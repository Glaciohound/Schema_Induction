import re
import json
from collections import defaultdict
import math

from components.constants import \
    all_event_types, months, interested_categories
from components.logic_tools import sort_rank, reverse_dict


def get_all_sentences(corpora, write_in=False):
    if isinstance(corpora, list):
        corpora = corpora_to_dict(corpora)
    output = {
        (_title, (_i, _j)): _sentence
        for _title, _article in corpora.items()
        for _i, _paragraph_split in enumerate(_article["content-cased-split"])
        for _j, _sentence in enumerate(_paragraph_split)
    }
    if write_in:
        write_all_elements_to_corpora(
            output, corpora
        )
    return output


def get_all_paragraphs(corpora, write_in=False, split=False):
    if isinstance(corpora, list):
        corpora = corpora_to_dict(corpora)
    output = {
        (_title, _i): _paragraph_split if split else "".join(_paragraph_split)
        for _title, _article in corpora.items()
        for _i, _paragraph_split in enumerate(_article["content-cased-split"])
    }
    if write_in:
        write_all_elements_to_corpora(
            output, corpora
        )
    return output


def write_all_elements_to_corpora(content, corpora):
    for title, element in content.items():
        title = title[0]
        article = corpora[title]
        if "elements" not in article:
            article["elements"] = []
        article["elements"].append(element)


def get_all_type_contents(
        all_events, corpora, content_type,
        to_element=False, num_contents_each_event=1
):
    if isinstance(corpora, list):
        corpora = corpora_to_dict(corpora)
    groupby_type = group_events_by_type(all_events)
    type_sentences = {
        _type: [
            extract_relevant_sentences(
                _event, corpora[_event["MESSAGE: ID"]], content_type
            )[0]
            for _event in _group
        ]
        for _type, _group in groupby_type.items()
    }
    if to_element:
        type_sentences = {
            _type: [
                _sentence
                for _sentences in _group
                for _sentence in _sentences[:num_contents_each_event]
            ]
            for _type, _group in type_sentences.items()
        }
    return type_sentences


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


def extract_relevant_sentences(event, article, content_type="sentence"):
    if content_type == "sentence":
        all_sentences = [
            _sentence
            for _paragraph_split in article["content-cased-split"]
            for _sentence in _paragraph_split
        ]
    elif content_type == "paragraph" or content_type == "paragraph-split":
        all_sentences = [
            "".join(_paragraph_split)
            for _paragraph_split in article["content-cased-split"]
        ]
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


def extract_relevant_sentences_from_events(events, corpora, content_type):
    if isinstance(corpora, list):
        corpora = corpora_to_dict(corpora)
    all_results = [
        extract_relevant_sentences(
            _event, corpora[_event["MESSAGE: ID"]], content_type)
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
        for _type in all_event_types
    }
    return groupby_type


def load_selected_names(selected_names_file, manual_synonym):
    manual_synonym = {
        _keywords[0]: _keywords[1:]
        for _keywords in manual_synonym.values()
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
    all_selected_names_index = reverse_dict(
        selected_names, inner_key="lemma_names")
    return selected_names, all_selected_names_index


def get_element_from_index(article, index):
    if isinstance(index, tuple):
        content = article["content-cased-split"][int(index[0])][int(index[1])]
    else:
        content = "".join(article["content-cased-split"][int(index)])
    return content
