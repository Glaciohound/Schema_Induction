import re
import itertools
from collections import defaultdict
import math

characters_to_strip = "Ġ@"


def get_all_sentences(corpora):
    output = list(itertools.chain(
        *list(itertools.chain(
            *list(
                itertools.chain(_paragraph_split)
                for _paragraph_split in _article["content-cased-split"]
            )
        ) for _article in corpora)
    ))
    return output


def get_event_keywords(event):
    return set(filter(
        lambda x: x not in ["", "-", "*"],
        itertools.chain(*list(
            list(map(lambda x: x[1:-1], re.findall(r"\"[^\"]*\"", _piece)))
            for _list in event.values() if isinstance(_list, list)
            for _piece in _list
        ))))


def extract_relevant_sentences(event, article):
    all_sentences = list(
        itertools.chain(*[
            _paragraph_split
            for _paragraph_split in article["content-cased-split"]
        ]))
    keywords = get_event_keywords(event)
    counter = defaultdict(lambda: 0)
    for i, sentence in enumerate(all_sentences):
        sentence_upper = sentence.upper()
        for keyword in keywords:
            # keyword = re.sub(r" *\(.*\) *", "", keyword)
            # keyword = re.sub(r"$[- ]*", "", keyword)
            if keyword in sentence_upper:
                counter[i] += len(re.findall(keyword, sentence_upper))
    counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    output = list(map(lambda x: (all_sentences[x[0]], x[1]), counter))
    if counter != []:
        return output
    else:
        return None


def corpora_to_dict(corpora):
    return {
        _article["title"]: _article
        for _article in corpora
    }


def sort_rank(rank):
    return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True))


def merge_ranked_list(lists):
    full_ranking = defaultdict(lambda: 0)
    for single_list in lists:
        for _i, _word in enumerate(single_list):
            full_ranking[_word.strip(characters_to_strip)] += 1 / (_i+1)
    full_ranked_list = sort_rank(full_ranking)
    return full_ranked_list


def rebalance_by_weight(rank, selected_names, all_names_index):
    rank = {
        _word: _weight
        # / math.sqrt(
        # selected_names[all_names_index[_word]]["weight"]
        # )
        for _word, _weight in rank.items()
        if _word in all_names_index
    }
    rank = sort_rank(rank)
    if len(rank) == 0:
        rank = {None: 0}
    return rank
