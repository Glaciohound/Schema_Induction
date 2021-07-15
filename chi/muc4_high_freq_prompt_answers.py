import os
import argparse
import numpy as np
import itertools
import json
from collections import defaultdict, Counter
from tqdm import tqdm

import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

from load_muc4 import load_muc4
from tools.muc4_tools import \
    get_all_sentences, extract_relevant_sentences, \
    corpora_to_dict, merge_ranked_list, characters_to_strip, \
    rebalance_by_weight


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="roberta-large")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--list-len", type=int, default=10)
    parser.add_argument("--raw-output",
                        default="data/muc34/outputs/all-prompt-results.json")
    parser.add_argument("--selected-names-file",
                        default="data/muc34/outputs/selected-names.json")
    parser.add_argument("--by-type-cache-file", type=str,
                        default="data/muc34/outputs/by-type-cache.json")
    parser.add_argument("--by-type-output", type=str,
                        default="data/muc34/outputs/by-type-output.json")
    parser.add_argument("--retrieval-output", type=str,
                        default="data/muc34/outputs/retrieval-output.json")
    parser.add_argument("--task-name", type=str, default="all",
                        choices=["all", "by-type", "retrieve-by-type"])
    parser.add_argument("--num-select-names", type=int, default=30)
    parser.add_argument("--top-num-synset-LM", type=int, default=10)
    args = parser.parse_args()
    return args


prompt_sentences = (
    "The reporter witnessed the _ .",
    "This is a typical _ incident.",
)
all_types = {
    "ATTACK": "attack",
    "BOMBING": "bombing",
    "ARSON": "fire",
    "KIDNAPPING": "kidnapping",
    "ROBBERY": "robbery"
}


def prompt_all_sentences(corpora, args):
    if os.path.exists(args.raw_output):
        with open(args.raw_output, 'r') as f:
            prompted_lists = json.load(f)
    else:
        from LM_prompt import get_LM, LM_prompt
        tokenizer, maskedLM = get_LM(args.model_name)

        all_sentences = get_all_sentences(corpora)
        if 0 < args.num_samples <= len(all_sentences):
            all_sentences = np.random.choice(
                all_sentences, args.num_samples, False)
        all_prompts = [
            _sentence+" "+prompt_sentence
            for _sentence in all_sentences
            for prompt_sentence in prompt_sentences
        ]
        all_prompt_answers = LM_prompt(
            all_prompts,
            tokenizer, maskedLM,
            args.model_name, args.top_k
        )
        prompted_lists = list(map(lambda x: x[0], all_prompt_answers))
        with open(args.raw_output, 'w') as f:
            json.dump(prompted_lists, f)

    # full_ranking = merge_ranked_list(map(lambda x: [x[0]], ranked_lists))
    full_ranking = merge_ranked_list(prompted_lists)
    selected_names = dict()
    all_selected_names = set()
    try:
        nltk.data.find("corpora/wordnet.zip")
    except LookupError:
        nltk.download("wordnet")
    lemmatizer = WordNetLemmatizer()
    porter = PorterStemmer()
    lancaster = LancasterStemmer()

    def intersect(set1, set2):
        return len(set(set1).intersection(set(set2))) != 0

    def all_attr_of_synsets(_query, attr_name, filter_pos=None):
        output = list(map(
            lambda _syn: getattr(_syn, attr_name, "")(),
            filter(
                lambda _syn: True if filter_pos is None else
                _syn.pos() == filter_pos,
                wordnet.synsets(_query)
            )
        ))
        return output

    def stemming(word):
        stemmed = [word]
        for stemmer in (porter, lancaster):
            _stemmed = stemmer.stem(word)
            if _stemmed.lower() != word.lower() and \
                    _stemmed.lower()+"e" != word.lower():
                for _suffix in ["", "e"]:
                    if len(wordnet.synsets(_stemmed+_suffix)) != 0:
                        stemmed.append(_stemmed + _suffix)
        return stemmed

    def is_verb_deviant(name):
        if (
            name not in list(itertools.chain(
                *all_attr_of_synsets(name, "lemma_names")))
            or intersect(all_attr_of_synsets(name, "pos"), {"a", "s"})
        ):
            return False

        synset = [name] + list(map(
            lambda x: x.lemma_names()[0],
            wordnet.synsets(name)
        ))[:2]
        for synonym in synset:
            stemmed = stemming(synonym)
            for _stemmed in stemmed:
                if "v" in all_attr_of_synsets(_stemmed, "pos"):
                    return True
        return False

    def are_synonyms(word1, word2=None):
        if word2 is None:
            assert isinstance(word1, tuple)
            word1, word2 = word1
        # return intersect(stemming(word1), synset_by_LM(word2)) or \
        #     intersect(stemming(word2), synset_by_LM(word1))
        for _word1, _word2 in ((word1, word2), (word2, word1)):
            if _word1 in (
                [_word2] + stemming(_word2) +
                list(itertools.chain(
                    *all_attr_of_synsets(_word2, "lemma_names", "n")[:2]
                )) +
                list(itertools.chain(
                    *all_attr_of_synsets(_word2, "lemma_names", "v")
                ))
            ):
                return True
            # or _word1 in ". ".join(
            #     all_attr_of_synsets(_word2, "definition")
            # ):
            # if _word1 in list(map(
            #     lambda x: x[0],
            #     all_attr_of_synsets(_word2, "lemma_names", "v")
            # )):
            #     return True
        return False

    def find_synonyms(word, group):
        return list(filter(lambda x: are_synonyms(word, x), group))

    for _name, _weight in tqdm(full_ranking.items()):
        if is_verb_deviant(_name):
            _lemmatized = lemmatizer.lemmatize(_name)
            synset = find_synonyms(_lemmatized, selected_names) + \
                find_synonyms(_name, selected_names)
            if len(synset) == 0:
                if len(selected_names) >= args.num_select_names:
                    continue
                selected_names[_lemmatized] = {
                    "lemma_names":
                    list(set(stemming(_lemmatized))
                         .difference(all_selected_names)),
                    "weight": _weight,
                }
                all_selected_names.update(selected_names[_lemmatized])
            else:
                selected_names[synset[-1]]["lemma_names"].extend(
                    {_name, _lemmatized}.difference(all_selected_names)
                )
                selected_names[synset[-1]]["weight"] += _weight
                all_selected_names.update(selected_names[synset[-1]])

    selected_names = dict(sorted(
        selected_names.items(), key=lambda x: x[1]["weight"]
    ))

    with open(args.selected_names_file, 'w') as f:
        print(selected_names.keys())
        json.dump(selected_names, f, indent=4)


def prompt_relevant_sentences(corpora, events, args):
    type_category = "INCIDENT: TYPE"
    with open(args.selected_names_file, 'r') as f:
        selected_names = json.load(f)
    all_selected_names_index = {
        _fine_grained: _name
        for _name, _group in selected_names.items()
        for _fine_grained in _group
    }
    all_sentences = get_all_sentences(corpora)

    if not os.path.exists(args.by_type_cache_file):
        groupby_type = {
            _type: list(filter(lambda x: x[type_category][0] == _type, events))
            for _type in all_types
        }
        type_sentence_samples = {
            _type: list(filter(
                lambda x: x is not None,
                map(
                    lambda _event: extract_relevant_sentences(
                        _event, corpora[_event["MESSAGE: ID"]])[0][0],
                    np.random.choice(
                        _group, min(args.num_samples, len(_group)), False)
                )
            ))
            for _type, _group in groupby_type.items()
        }

        with open(args.raw_output, 'r') as f:
            prompted_lists = json.load(f)
        type_prompts = {
            _type: [
                prompted_lists[all_sentences.index(_sentence) // 2]
                for _sentence in _sentences
            ]
            for _type, _sentences in type_sentence_samples.items()
        }
        with open(args.by_type_cache_file, 'w') as f:
            json.dump(type_prompts, f)
    else:
        with open(args.by_type_cache_file, 'r') as f:
            type_prompts = json.load(f)

    type_ranked_list = {
        _type: list(filter(
            lambda x: x in selected_names,
            merge_ranked_list(map(
                lambda x: x[0],
                _prompts
            )).keys()
        ))[:args.list_len]
        for _type, _prompts in type_prompts.items()
    }
    type_ranks_by_name = defaultdict(
        lambda: defaultdict(lambda: {_i: 0 for _i in range(30)})
    )
    for _type, _prompts in type_prompts.items():
        for _prompt in _prompts:
            for _i, _name in enumerate(_prompt[0]):
                _name = _name.strip(characters_to_strip)
                if _name in all_selected_names_index:
                    type_ranks_by_name[_type][
                        all_selected_names_index[_name]
                    ][_i] += 1
    with open(args.by_type_output, "w") as f:
        json.dump((type_ranked_list, type_ranks_by_name), f, indent=4)


def retrieve_by_type(corpora, args):
    with open(args.selected_names_file, 'r') as f:
        selected_names = json.load(f)
    all_selected_names_index = {
        _fine_grained: _name
        for _name, _group in selected_names.items()
        for _fine_grained in _group["lemma_names"]
    }
    all_sentences = get_all_sentences(corpora)
    with open(args.raw_output, 'r') as f:
        prompted_lists = json.load(f)
    num_sentences = len(all_sentences)
    prompted_lists = list(map(
        lambda _i: rebalance_by_weight(
            merge_ranked_list(prompted_lists[2*_i: 2*_i+2]),
            selected_names, all_selected_names_index
        ),
        range(num_sentences)
    ))
    prompted_top_names = [
        all_selected_names_index.get(
            list(_list.keys())[0], None)
        for _list in prompted_lists
    ]
    retrieved_indexes = {
        _type: np.where(np.array(prompted_top_names) == _keyword)[0]
        for _type, _keyword in all_types.items()
    }
    with open(args.retrieval_output, 'w') as f:
        json.dump((
            {
                _type: [all_sentences[_i]
                        for _i in np.random.choice(
                            _indexes,
                            min(args.num_samples, len(_indexes)),
                            False)
                        ]
                for _type, _indexes in retrieved_indexes.items()
            },
            dict(sorted(Counter(prompted_top_names).items(),
                        key=lambda x: x[1], reverse=True))
        ), f, indent=4)


def main(args):
    dev_corpora, dev_events, tst_corpora, tst_events, proper_nouns = \
        load_muc4()

    if args.task_name == "all":
        prompt_all_sentences(dev_corpora, args)
    elif args.task_name == "by-type":
        dev_corpora_dict = corpora_to_dict(dev_corpora)
        prompt_relevant_sentences(
            dev_corpora_dict, dev_events, args
        )
    elif args.task_name == "retrieve-by-type":
        retrieve_by_type(dev_corpora, args)


if __name__ == "__main__":
    args = get_args()
    main(args)
