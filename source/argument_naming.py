import os
import json
import argparse
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from collections import defaultdict, Counter
import itertools

from components.muc4_tools import \
    corpora_to_dict, group_events_by_type, \
    extract_relevant_sentences_from_events
from components.load_muc4 import load_muc4
from components.np_extractors import SpacyNPExtractor
from components.constants import \
    argument_name_prompt_sentences, all_types, all_arguments
from components.wordnet_tools import \
    all_attr_of_synsets, are_synonyms, synsets
from components.logic_tools import \
    intersect, merge_ranked_list, sort_rank


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="roberta-large")
    parser.add_argument("--all-noun-phrases", type=str,
                        default="data/muc34/outputs/all-arguments.json")
    # parser.add_argument("--argument-prompts", type=str,
    #                     default="data/muc34/outputs/gt-argument-naming.json")
    parser.add_argument("--argument-prompts", type=str,
                        default="data/muc34/outputs/argument-naming.json")
    parser.add_argument("--selected-arguments", type=str,
                        default="data/muc34/outputs/selected-arguments.json")
    parser.add_argument("--overwrite-arguments", action="store_true")
    parser.add_argument("--overwrite-argument-naming", action="store_true")
    parser.add_argument("--overwrite-argument-filtering", action="store_true")
    parser.add_argument("--num-selected-names", type=int, default=30)
    parser.add_argument("--only-tackling-top", type=int, default=50)
    args = parser.parse_args()
    return args


def get_arguments_by_type(args, events, corpora):
    if args.overwrite_arguments or not os.path.exists(args.all_noun_phrases):
        parser = SpacyNPExtractor()

        events_by_type = group_events_by_type(events)
        sentences_by_type = {
            _type: extract_relevant_sentences_from_events(
                _events, corpora)
            for _type, _events in events_by_type.items()
        }

        num_sentences = sum(map(len, sentences_by_type.values()))
        pbar = tqdm(total=num_sentences)
        for _type, _sentences in sentences_by_type.items():
            for _sentence in _sentences:
                pbar.update()
                _sentence["extracted-noun-phrases"] = list(map(
                    lambda x: x.text,
                    parser.extract(_sentence["sentence"])
                ))

        with open(args.all_noun_phrases, 'w') as f:
            json.dump(sentences_by_type, f, indent=2)
    else:
        with open(args.all_noun_phrases, 'r') as f:
            sentences_by_type = json.load(f)
        num_sentences = sum(map(len, sentences_by_type.values()))

    return sentences_by_type, num_sentences


def prompt_argument_naming(args, arguments_by_type):
    if args.overwrite_argument_naming or \
            not os.path.exists(args.argument_prompts):
        from components.LM_prompt import LM_prompt, get_LM
        tokenizer, maskedLM = get_LM(args.model_name)
        lemmatizer = WordNetLemmatizer()

        def get_all_naming_prompts():
            for _type, _sentences in arguments_by_type.items():
                for _sentence in _sentences:
                    _sentence["argument-naming"] = {}
                    # _sentence.pop("extracted-noun-phrases")
                    # for _noun_phrase in :
                    for _noun_phrase in list(_sentence["arguments"]) + \
                            _sentence["extracted-noun-phrases"]:
                        if lemmatizer.lemmatize(_noun_phrase) == _noun_phrase:
                            copula = "is"
                        else:
                            copula = "are"
                        if _noun_phrase[0].islower():
                            _noun_phrase = "The " + _noun_phrase
                        prompt = _sentence["sentence"] + " " + \
                            argument_name_prompt_sentences[0].format(
                                _noun_phrase, copula, all_types[_type][0]
                            )
                        yield prompt

        all_naming_prompts = list(get_all_naming_prompts())
        prompted_result = LM_prompt(
            all_naming_prompts, tokenizer, maskedLM, strip=True
        )
        yield_prompt = iter(prompted_result)

        for _type, _sentences in arguments_by_type.items():
            for _sentence in _sentences:
                for _noun_phrase in list(_sentence["arguments"]) +\
                        _sentence["extracted-noun-phrases"]:
                    naming = list(map(
                        lemmatizer.lemmatize,
                        next(yield_prompt)[0][:5],
                    ))
                    _category = _sentence["arguments"].get(_noun_phrase, None)
                    _sentence["argument-naming"][_noun_phrase] = (
                        naming,
                        naming[0] == all_arguments[_category][0]
                        if _category is not None else None
                    )

        with open(args.argument_prompts, "w") as f:
            json.dump(arguments_by_type, f, indent=2)

    else:
        with open(args.argument_prompts, "r") as f:
            arguments_by_type = json.load(f)

    return arguments_by_type


def filter_namings(ranked_all_namings, num_selected_names):
    output = {}
    for naming, weight in tqdm(
            list(ranked_all_namings.items())[:args.only_tackling_top]
    ):
        output = sort_rank(output, key=lambda x: x[1]["weight"])
        if len(synsets(naming)) == 0 or not naming.islower():
            continue
        if intersect(all_attr_of_synsets(naming, "pos"), {"a"}) or \
                not intersect(all_attr_of_synsets(naming, "pos"), {"n"}):
            continue
        found_synonym = None
        for _existed, _group in output.items():
            for _lemma in _group["synonyms"]:
                if are_synonyms(_lemma, naming, "v", True):
                    if found_synonym is None:
                        print(f"found synonym: {naming} -> {_existed}")
                        output[_existed]["weight"] += weight
                        output[_existed]["synonyms"].append(naming)
                        found_synonym = _existed
                        break
        if found_synonym is not None:
            continue
        output[naming] = {
            "weight": weight,
            "synonyms": [naming],
        }
    output = dict(list(sort_rank(
        output, key=lambda x: x[1]["weight"]
    ).items())[:num_selected_names])
    for i, (_, _group) in enumerate(output.items()):
        _group["rank"] = i
    return output


def argument_filtering(args, argument_naming):
    if args.overwrite_argument_filtering or not os.path.exists(
            args.selected_arguments):
        all_namings = [
            _naming[0] for _sentences in argument_naming.values()
            for _sentence in _sentences
            for _noun, _naming in _sentence["argument-naming"].items()
            if _noun in _sentence["extracted-noun-phrases"]
        ]
        ranked_all_namings = merge_ranked_list(all_namings)
        ranked_all_namings = filter_namings(
            ranked_all_namings, args.num_selected_names)

        with open(args.selected_arguments, 'w') as f:
            json.dump(ranked_all_namings, f, indent=4)
    else:
        with open(args.selected_arguments, 'r') as f:
            ranked_all_namings = json.load(f)

    return ranked_all_namings


def evaluate_gt_accuracy(argument_naming, selected_arguments):
    result = defaultdict(lambda: Counter())
    selected_arguments_index = {
        _syn: _name
        for _name, _group in selected_arguments.items()
        for _syn in _group["synonyms"]
    }
    all_gt_argument_names = list(itertools.chain(
        *list(all_arguments.values())))

    for _type, _sentences in argument_naming.items():
        for _sentence in _sentences:
            for _argument, _naming in _sentence["gt-argument-naming"].items():
                _naming = list(filter(lambda x: x in selected_arguments_index,
                                      _naming[0]))
                _naming = _naming[0] if len(_naming) > 0 else None
                gt_cat = all_arguments[_sentence["arguments"][_argument]][0]
                if _naming == gt_cat:
                    result[f"{gt_cat}"].update(["hit"])
                result[f"{gt_cat}"].update(["gt"])
                if _naming in all_gt_argument_names:
                    result[f"{_naming}"].update(["pred"])

    for _category, _counter in result.items():
        print(_category,
              "precision:", _counter["hit"] / _counter["pred"],
              "recall:", _counter["hit"] / _counter["gt"],
              "F1:", _counter["hit"] / (_counter["pred"] + _counter["gt"]) * 2
              )


def main(args):
    dev_corpora, dev_events, _, _, _ = load_muc4()
    dev_corpora = corpora_to_dict(dev_corpora)

    arguments_by_type, num_sentences = get_arguments_by_type(
        args, dev_events, dev_corpora)

    argument_naming = prompt_argument_naming(args, arguments_by_type)
    ranked_all_namings = argument_filtering(args, argument_naming)
    print(ranked_all_namings.keys())

    # evaluate_gt_accuracy(argument_naming, ranked_all_namings)


if __name__ == "__main__":
    args = parse_args()
    main(args)
