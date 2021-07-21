import os
import json
import argparse
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer

from components.muc4_tools import \
    corpora_to_dict, group_events_by_type, \
    extract_relevant_sentences_from_events
from components.load_muc4 import load_muc4
from components.np_extractors import SpacyNPExtractor
from components.constants import \
    argument_name_prompt_sentences, all_types, all_arguments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="roberta-large")
    parser.add_argument("--all-noun-phrases", type=str,
                        default="data/muc34/outputs/all-arguments.json")
    parser.add_argument("--argument-prompts", type=str,
                        default="data/muc34/outputs/argument-naming.json")
    parser.add_argument("--overwrite-arguments", action="store_true")
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


def main(args):
    dev_corpora, dev_events, _, _, _ = load_muc4()
    dev_corpora = corpora_to_dict(dev_corpora)

    arguments_by_type, num_sentences = get_arguments_by_type(
        args, dev_events, dev_corpora)

    from components.LM_prompt import LM_prompt, get_LM
    tokenizer, maskedLM = get_LM(args.model_name)
    lemmatizer = WordNetLemmatizer()

    def get_all_gt_naming_prompts():
        for _type, _sentences in arguments_by_type.items():
            for _sentence in _sentences:
                _sentence["gt-argument-naming"] = {}
                _sentence.pop("extracted-noun-phrases")
                for _noun_phrase in _sentence["arguments"]:
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

    all_gt_naming_prompts = list(get_all_gt_naming_prompts())
    prompted_result = LM_prompt(
        all_gt_naming_prompts[:10], tokenizer, maskedLM, strip=True
    )

    yield_prompt = iter(prompted_result)

    for _type, _sentences in arguments_by_type.items():
        for _sentence in _sentences:
            for _noun_phrase, _category in _sentence["arguments"].items():
                naming = lemmatizer.lemmatize(next(yield_prompt)[0][0])
                _sentence["gt-argument-naming"][_noun_phrase] = (
                    naming,
                    naming == all_arguments[_category][0]
                )

    with open(args.argument_prompts, "w") as f:
        json.dump(arguments_by_type, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    main(args)
