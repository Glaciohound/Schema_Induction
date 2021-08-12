import os
import itertools
import json
from collections import defaultdict, Counter
from tqdm import tqdm

from nltk.corpus import wordnet

from components.load_muc4 import load_muc4
from components.muc4_tools import \
    get_all_sentences, get_all_paragraphs, \
    corpora_to_dict, get_all_type_contents, load_selected_names
from components.logic_tools import merge_ranked_list
from components.wordnet_tools import \
    all_attr_of_synsets, are_synonyms, stemming,\
    lemmatizer
from components.logic_tools import intersect
from components.prompt_all import \
    prompt_all_with_cache, get_all_sentence_prompts, \
    get_all_paragraph_prompts, get_all_paragraph_split_prompts
from components.get_args import get_args
from components.logging import logger
from components.constants import all_event_types


def high_freq_in_all_sentences(corpora, args):
    if args.event_element == "sentence":
        logger.info("Prompt from sentences")
        prompted_lists = prompt_all_with_cache(
            args, corpora, get_all_sentence_prompts,
            args.prompt_all_sentences_results,
            tokens_only=True,
            overwrite_prompt_cache=args.overwrite_prompt_cache,
        )
        logger.info(f"len = {len(prompted_lists)}")
    elif args.event_element == "paragraph":
        logger.info("Prompt from paragraphs")
        prompted_lists = prompt_all_with_cache(
            args, corpora, get_all_paragraph_prompts,
            args.prompt_all_paragraphs_results,
            tokens_only=True,
            overwrite_prompt_cache=args.overwrite_prompt_cache,
        )
        logger.info(f"len = {len(prompted_lists)}")
    elif args.event_element == "paragraph-split":
        logger.info("Prompt from paragraph splits")
        prompted_lists = prompt_all_with_cache(
            args, corpora, get_all_paragraph_split_prompts,
            args.prompt_all_paragraphs_split_results,
            tokens_only=True,
            overwrite_prompt_cache=args.overwrite_prompt_cache,
        )
        logger.info(f"len = {len(prompted_lists)}")

    if not args.overwrite_top_events and os.path.exists(args.top_names_file):
        with open(args.top_names_file, 'r') as f:
            selected_names = json.load(f)
        logger.info(str(dict(enumerate(selected_names.keys()))))
        return

    logger.info("Prompting names")
    full_ranking = merge_ranked_list(
        [merge_ranked_list(_list, "max") for _list in prompted_lists],
        "power"
    )
    logger.info(f"shrinking from list of length {len(full_ranking)}")
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
                if len(selected_names) >= args.num_selected_events:
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
    for _i, _item in enumerate(selected_names.values()):
        _item["rank"] = _i
    logger.info("Got selected names:")
    logger.info(str(dict(enumerate(selected_names.keys()))))

    with open(args.top_names_file, 'w') as f:
        json.dump(selected_names, f, indent=4)
        logger.info(f"Dump selected names to {args.top_names_file}")


def prompt_relevant_sentences(corpora, events, args):
    selected_names, all_selected_names_index = load_selected_names(
        args.top_names_file, all_event_types)

    if args.event_element == "sentence":
        all_sentences = list(get_all_sentences(corpora).values())
        all_prompt_results_file = args.prompt_all_sentences_results
    elif args.event_element == "paragraph":
        all_sentences = list(get_all_paragraphs(corpora).values())
        all_prompt_results_file = args.prompt_all_paragraphs_results
    elif args.event_element == "paragraph-split":
        all_sentences = list(get_all_paragraphs(corpora).values())
        all_prompt_results_file = args.prompt_all_paragraphs_split_results
    else:
        raise NotImplementedError()
    type_sentences = get_all_type_contents(
        events, corpora, args.event_element)
    with open(all_prompt_results_file, 'r') as f:
        prompted_lists = json.load(f)

    type_prompts = {
        _type: [
            merge_ranked_list(
                [
                    _list
                    for _sentence in _sentences[:args.num_contents_each_event]
                    for _list in
                    prompted_lists[all_sentences.index(_sentence["sentence"])]
                ],
                args.merge_single_policy
            )
            for _sentences in _group
        ]
        for _type, _group in type_sentences.items()
    }
    type_ranks_by_name = defaultdict(
        lambda: dict(enumerate([Counter() for _ in range(args.top_k)]))
    )
    for _type, _prompts in type_prompts.items():
        for _prompt_list in _prompts:
            for _i, _name in enumerate(_prompt_list.keys()):
                if _i >= args.top_k:
                    break
                if _name in all_selected_names_index:
                    type_ranks_by_name[_type][_i].update(
                        [all_selected_names_index[_name]])
    type_ranks_by_name = {
        _type: {
            _i: dict(_counter.most_common())
            for _i, _counter in _group.items()
        }
        for _type, _group in type_ranks_by_name.items()
    }
    with open(args.top_names_by_type_file, "w") as f:
        logger.info("Dumping top names by type to "
                    f"{args.top_names_by_type_file}")
        json.dump(type_ranks_by_name, f, indent=4)


def main(args):
    dev_corpora, dev_events, tst_corpora, tst_events, proper_nouns = \
        load_muc4(args=args)
    dev_corpora_dict = corpora_to_dict(dev_corpora)

    high_freq_in_all_sentences(dev_corpora, args)
    prompt_relevant_sentences(
        dev_corpora_dict, dev_events, args
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
