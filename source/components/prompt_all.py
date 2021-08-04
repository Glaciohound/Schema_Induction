import os
import json
import itertools

from components.muc4_tools import \
    get_all_sentences, get_all_paragraphs
from components.logic_tools import random_choice
from components.constants import event_prompt_sentences


def prompt_all_with_cache(args, corpora,
                          all_prompt_sentences_getter, cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            prompted_lists = json.load(f)
    else:
        all_prompt_sentences = all_prompt_sentences_getter(corpora)
        all_prompt_sentences_expand = list(
            itertools.chain(*all_prompt_sentences)
        )

        from LM_prompt import get_LM, LM_prompt
        tokenizer, maskedLM = get_LM(args.model_name)
        all_prompt_answers = LM_prompt(
            all_prompt_sentences_expand,
            tokenizer, maskedLM,
            args.model_name, args.top_k, tokens_only=True
        )
        all_prompt_answers = iter(all_prompt_answers)
        prompted_lists = [
            [next(all_prompt_answers) for _ in len(_group)]
            for _group in all_prompt_sentences
        ]
        with open(cache_file, 'w') as f:
            json.dump(prompted_lists, f)

    return prompted_lists


def get_all_sentence_prompts(args, corpora):
    all_sentences = random_choice(
        get_all_sentences(corpora), args.num_samples)
    all_prompts = [
        [_sentence+" "+prompt_sentence
         for prompt_sentence in event_prompt_sentences]
        for _sentence in all_sentences
    ]
    return all_prompts


def get_all_paragraph_prompts(args, corpora):
    all_paragraphs = random_choice(
        get_all_paragraphs(corpora), args.num_samples)
    all_prompts = [
        [_sentence+" "+prompt_sentence
         for prompt_sentence in event_prompt_sentences]
        for _sentence in all_paragraphs
    ]
    return all_prompts