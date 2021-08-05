import itertools
import json
from collections import defaultdict
from tqdm import tqdm

from nltk.corpus import wordnet

from components.load_muc4 import load_muc4
from components.muc4_tools import \
    get_all_sentences, corpora_to_dict, get_all_type_sentences
from components.logic_tools import merge_ranked_list
from components.constants import characters_to_strip
from components.wordnet_tools import \
    all_attr_of_synsets, are_synonyms, stemming,\
    lemmatizer
from components.logic_tools import intersect
from components.prompt_all import \
    prompt_all_with_cache, get_all_sentence_prompts, get_all_paragraph_prompts
from components.get_args import get_args
from components.logging import getLogger


logger = getLogger("top-events")


def high_freq_in_all_sentences(corpora, args):
    if args.select_names_from == "sentence":
        logger.info("prompt from sentences")
        prompted_lists = prompt_all_with_cache(
            args, corpora, get_all_sentence_prompts,
            args.prompt_all_sentences_results
        )
        logger.info(f"len = {len(prompted_lists)}")
    elif args.select_names_from == "paragraph":
        logger.info("prompt from paragraphs")
        prompted_lists = prompt_all_with_cache(
            args, corpora, get_all_paragraph_prompts,
            args.prompt_all_paragraphs_results
        )
        logger.info(f"len = {len(prompted_lists)}")
    else:
        raise NotImplementedError()

    prompted_lists = list(itertools.chain(*prompted_lists))
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
    logger.info(f"got selected names: {selected_names.keys()}")

    with open(args.top_names_file, 'w') as f:
        json.dump(selected_names, f, indent=4)
        logger.info(f"dump selected names to {args.top_names_file}")


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
        load_muc4(args.data_dir, args.loading_cache_file,
                  args.overwrite_loading)
    dev_corpora_dict = corpora_to_dict(dev_corpora)

    high_freq_in_all_sentences(dev_corpora, args)
    exit()
    prompt_relevant_sentences(
        dev_corpora_dict, dev_events, args
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
