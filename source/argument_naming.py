import os
import json
import numpy as np
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer

from components.muc4_tools import \
    corpora_to_dict, get_all_type_contents, \
    get_all_sentences, get_all_paragraphs, get_element_from_index
from components.load_muc4 import load_muc4
from components.np_extractors import SpacyNPExtractor
from components.constants import \
    argument_name_prompt_sentences, all_event_types, all_arguments_types
from components.wordnet_tools import \
    all_attr_of_synsets, are_synonyms, synsets
from components.logic_tools import \
    intersect, merge_ranked_list, sort_rank
from components.get_args import get_args
from components.logging import getLogger

logger = getLogger("argument-naming")


def get_all_arguments(args, events, corpora):
    if args.overwrite_ner or not os.path.exists(args.all_arguments):
        parser = SpacyNPExtractor()
        if args.event_element == "sentence":
            all_sentences_dict = get_all_sentences(corpora, write_in=True)
            retrieval_output = args.sentence_retrieval_output
        elif args.event_element == "paragraph":
            all_sentences_dict = get_all_paragraphs(corpora, write_in=True)
            retrieval_output = args.paragraph_retrieval_output
        all_sentences = np.array(list(all_sentences_dict.values()))
        all_indexes = np.array(list(all_sentences_dict.keys()))
        num_sentences = all_sentences.shape[0]
        with open(retrieval_output, 'r') as f:
            retrieved_indexes = json.load(f)
        sentences_by_type = get_all_type_contents(
            events, corpora, args.event_element,
            True, args.num_contents_each_event)

        all_arguments = corpora
        for _article in all_arguments.values():
            _article.pop("content")
            # _article.pop("content-cased-split")
            _article["elements"] = {
                _element: {"gt-arguments": []}
                for _element in _article["elements"]}

        for _type, _group in sentences_by_type.items():
            for _element in _group:
                _title = _element["article"]
                content = _element["sentence"]
                element_value = all_arguments[_title]["elements"][content]
                element_value["gt-arguments"].extend(list(zip(
                    [_type]*len(_element["arguments"]),
                    _element["arguments"].keys(),
                    _element["arguments"].values(),
                )))
        for _type, _indices in retrieved_indexes.items():
            for _index in _indices:
                _title, _inner_index = all_indexes[_index]
                content = get_element_from_index(corpora[_title], _inner_index)
                corpora[_title]["elements"][content]["pred-event-type"] = _type
        pbar = tqdm(total=num_sentences)
        for _type, _article in all_arguments.items():
            for _sentence, _value in _article["elements"].items():
                pbar.update()
                _value["extracted-ner"] = list(map(
                    lambda x: x.text, parser.extract(_sentence)))
        for _article in all_arguments.values():
            _article.pop("content-cased-split")

        with open(args.all_arguments, 'w') as f:
            json.dump(all_arguments, f, indent=2)
    else:
        with open(args.all_arguments, 'r') as f:
            all_arguments = json.load(f)

    return all_arguments


def prompt_argument_naming(args, all_arguments):
    if args.overwrite_argument_naming or \
            not os.path.exists(args.argument_prompts):
        lemmatizer = WordNetLemmatizer()

        def get_all_naming_prompts():
            for _article in all_arguments.values():
                for _sentence, _group in _article["elements"].items():
                    _group["argument-naming"] = {}
                    if "pred-event-type" in _group:
                        _event_type = all_event_types[
                            _group["pred-event-type"]][0]
                    else:
                        _event_type = "incident"
                    for _noun_phrase in _group["extracted-ner"]:
                        if lemmatizer.lemmatize(_noun_phrase) == _noun_phrase:
                            copula = "is"
                        else:
                            copula = "are"
                        if _noun_phrase[0].islower():
                            _noun_phrase = "The " + _noun_phrase
                        prompt = _sentence + " " + \
                            argument_name_prompt_sentences[0].format(
                                _noun_phrase, copula, _event_type
                            )
                        yield prompt

        all_naming_prompts = list(get_all_naming_prompts())

        from components.LM_prompt import LM_prompt, get_LM
        tokenizer, maskedLM = get_LM(args.model_name)
        prompted_result = LM_prompt(
            all_naming_prompts, tokenizer, maskedLM, strip=True
        )
        yield_prompt = iter(prompted_result)

        for _article in all_arguments.values():
            for _sentence, _group in _article["elements"].items():
                for _noun_phrase in _group["extracted-ner"]:
                    naming = list(map(
                        lemmatizer.lemmatize,
                        next(yield_prompt)[0][:args.argument_list_len],
                    ))
                    _group["argument-naming"][_noun_phrase] = naming

        with open(args.argument_prompts, "w") as f:
            json.dump(all_arguments, f, indent=2)

    else:
        with open(args.argument_prompts, "r") as f:
            all_arguments = json.load(f)

    all_namings = [
        _naming for _article in all_arguments.values()
        for _element in _article["elements"].values()
        for _noun, _naming in _element["argument-naming"].items()
    ]

    return all_namings


def argument_filtering(args, all_namings):
    if args.overwrite_argument_filtering or not os.path.exists(
            args.top_arguments):
        ranked_all_namings = merge_ranked_list(all_namings)
        output = {}
        for naming, weight in tqdm(
            list(ranked_all_namings.items())[:200]
        ):
            output = sort_rank(output, key=lambda x: x[1]["weight"])
            if len(synsets(naming)) == 0 or not naming.islower():
                continue
            if intersect(all_attr_of_synsets(naming, "pos"), {"a"}) or \
                    not intersect(all_attr_of_synsets(naming, "pos"), {"n"}):
                continue
            found_synonym = None
            for _existed, _group in output.items():
                for _lemma in _group["lemma_names"]:
                    if are_synonyms(_lemma, naming, "v", True):
                        if found_synonym is None:
                            logger.info(f"synonym: {naming} -> {_existed}")
                            output[_existed]["weight"] += weight
                            output[_existed]["lemma_names"].append(naming)
                            found_synonym = _existed
                            break
            if found_synonym is not None:
                continue
            output[naming] = {
                "weight": weight,
                "lemma_names": [naming],
            }
        output = dict(list(sort_rank(
            output, key=lambda x: x[1]["weight"]
        ).items())[:args.num_selected_arguments])
        for i, (_, _group) in enumerate(output.items()):
            _group["rank"] = i

        with open(args.top_arguments, 'w') as f:
            json.dump(output, f, indent=4)
    else:
        with open(args.top_arguments, 'r') as f:
            output = json.load(f)

    return output


def main(args):
    dev_corpora, dev_events, _, _, _ = load_muc4(args=args)
    dev_corpora = corpora_to_dict(dev_corpora)

    all_arguments = get_all_arguments(args, dev_events, dev_corpora)
    argument_naming = prompt_argument_naming(args, all_arguments)
    ranked_all_namings = argument_filtering(args, argument_naming)
    logger.info(dict(enumerate(ranked_all_namings.keys())))


if __name__ == "__main__":
    args = get_args()
    main(args)
