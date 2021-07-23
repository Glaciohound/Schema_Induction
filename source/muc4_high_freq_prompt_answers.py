import os
import argparse
import numpy as np
import itertools
import json
from collections import defaultdict
from tqdm import tqdm

from nltk.corpus import wordnet

from components.load_muc4 import load_muc4
from components.muc4_tools import \
    get_all_sentences, corpora_to_dict, merge_ranked_list, \
    characters_to_strip, get_all_type_sentences
from components.wordnet_tools import \
    all_attr_of_synsets, are_synonyms, stemming,\
    lemmatizer
from components.constants import event_prompt_sentences
from components.logic_tools import intersect


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="roberta-large")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--list-len", type=int, default=10)
    parser.add_argument("--all-prompt-results",
                        default="data/muc34/outputs/all-prompt-results.json")
    parser.add_argument("--selected-names-file",
                        default="data/muc34/outputs/selected-names.json")
    parser.add_argument("--by-type-output", type=str,
                        default="data/muc34/outputs/by-type-top-names.json")
    parser.add_argument("--task-name", type=str, default="all",
                        choices=["all", "by-type"])
    parser.add_argument("--num-selected-names", type=int, default=60)
    parser.add_argument("--top-num-synset-LM", type=int, default=10)
    args = parser.parse_args()
    return args


def prompt_all_sentences(corpora, args):
    if os.path.exists(args.all_prompt_results):
        with open(args.all_prompt_results, 'r') as f:
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
            for prompt_sentence in event_prompt_sentences
        ]
        all_prompt_answers = LM_prompt(
            all_prompts,
            tokenizer, maskedLM,
            args.model_name, args.top_k
        )
        prompted_lists = list(map(lambda x: x[0], all_prompt_answers))
        with open(args.all_prompt_results, 'w') as f:
            json.dump(prompted_lists, f)

    full_ranking = merge_ranked_list(prompted_lists)
    selected_names = dict()
    all_selected_names = set()

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

    def find_synonyms(word, group):
        return list(filter(lambda x: are_synonyms(word, x), group))

    for _name, _weight in tqdm(full_ranking.items()):
        if is_verb_deviant(_name):
            _lemmatized = lemmatizer.lemmatize(_name)
            synset = find_synonyms(_lemmatized, selected_names) + \
                find_synonyms(_name, selected_names)
            if len(synset) == 0:
                if len(selected_names) >= args.num_selected_names:
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
        selected_names.items(), key=lambda x: x[1]["weight"], reverse=True
    ))

    with open(args.selected_names_file, 'w') as f:
        print(selected_names.keys())
        json.dump(selected_names, f, indent=4)


def prompt_relevant_sentences(corpora, events, args):
    with open(args.selected_names_file, 'r') as f:
        selected_names = json.load(f)
    all_selected_names_index = {
        _fine_grained: _name
        for _name, _group in selected_names.items()
        for _fine_grained in _group
    }
    all_sentences = get_all_sentences(corpora)
    type_sentences = get_all_type_sentences(events, corpora)
    with open(args.all_prompt_results, 'r') as f:
        prompted_lists = json.load(f)
    type_prompts = {
        _type: [
            merge_ranked_list(
                prompted_lists[all_sentences.index(_sentence[0]) * 2]
            )
            for _sentences in _group
            for _sentence in _sentences[:2]
        ]
        for _type, _group in type_sentences.items()
    }

    type_ranked_list = {
        _type: list(filter(
            lambda x: x in selected_names,
            merge_ranked_list(_prompts).keys()
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


if __name__ == "__main__":
    args = get_args()
    main(args)
