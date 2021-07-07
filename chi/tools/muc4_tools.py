import re
import itertools
from collections import defaultdict


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
    return set(itertools.chain(*list(
        list(map(lambda x: x[1:-1], re.findall(r"\"[^\"]*\"", _piece)))
        for _list in event.values() if isinstance(_list, list)
        for _piece in _list
    )))


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
            if keyword in sentence_upper:
                counter[i] += 1
    counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    if counter != []:
        return all_sentences[counter[0][0]]
    else:
        return None


def corpora_to_dict(corpora):
    return {
        _article["title"]: _article
        for _article in corpora
    }
